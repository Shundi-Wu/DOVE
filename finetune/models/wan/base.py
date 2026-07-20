from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanPipeline,
    WanTransformer3DModel,
)
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, UMT5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model


class WanTrainerBase(Trainer):
    """Shared Wan2.1 plumbing for the one-step DOVE trainers."""

    UNLOAD_LIST = ["text_encoder", "vae"]
    MAX_SEQUENCE_LENGTH = 512

    @override
    def get_vae_dtype(self) -> torch.dtype:
        # Diffusers recommends fp32 for AutoencoderKLWan. Keeping this explicit
        # is also important for the pixel-space Stage-2 loss.
        return torch.float32

    @override
    def load_components(self) -> Components:
        model_path = str(self.args.model_path)
        components = Components()
        components.pipeline_cls = WanPipeline
        components.tokenizer = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )
        components.text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )
        components.transformer = WanTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )
        components.vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        components.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return components

    @override
    def initialize_pipeline(self) -> WanPipeline:
        return WanPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )

    def _latent_stats(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(
            self.components.vae.config.latents_mean,
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, self.components.vae.config.z_dim, 1, 1, 1)
        std = torch.as_tensor(
            self.components.vae.config.latents_std,
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, self.components.vae.config.z_dim, 1, 1, 1)
        return mean, std

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent = vae.encode(video).latent_dist.sample()
        mean, std = self._latent_stats(latent)
        latent = (latent - mean) / std
        return latent.to(dtype=self.state.weight_dtype)

    def decode_latents(self, latent: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        latent = latent.to(vae.device, dtype=vae.dtype)
        mean, std = self._latent_stats(latent)
        latent = latent * std + mean
        return vae.decode(latent, return_dict=False)[0]

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        text_inputs = self.components.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.accelerator.device)
        attention_mask = text_inputs.attention_mask.to(self.accelerator.device)
        sequence_lengths = attention_mask.gt(0).sum(dim=1).long()
        prompt_embedding = self.components.text_encoder(
            input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        # Match WanPipeline: discard padded token activations, then pad them
        # back with exact zeros to the fixed sequence length.
        padded_embeddings = []
        for embedding, sequence_length in zip(prompt_embedding, sequence_lengths):
            embedding = embedding[:sequence_length]
            padding = embedding.new_zeros(
                self.MAX_SEQUENCE_LENGTH - embedding.shape[0], embedding.shape[1]
            )
            padded_embeddings.append(torch.cat([embedding, padding], dim=0))
        return torch.stack(padded_embeddings, dim=0)

    def predict_clean_latent(
        self,
        lq_latent: torch.Tensor,
        prompt_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x0 from an observed LQ latent with Wan flow matching.

        For Wan's flow prediction parameterization, x0 = x_sigma - sigma*v.
        ``flow_sigma`` is the actual shifted sigma presented to the pretrained
        transformer; unlike CogVideoX, no DDIM velocity conversion is used.
        """
        sigma_value = float(self.args.flow_sigma)
        if not 0.0 < sigma_value <= 1.0:
            raise ValueError(f"flow_sigma must be in (0, 1], got {sigma_value}")

        batch_size = lq_latent.shape[0]
        prompt_embedding = prompt_embedding.to(
            device=lq_latent.device,
            dtype=lq_latent.dtype,
        )
        timestep = torch.full(
            (batch_size,),
            sigma_value * self.components.scheduler.config.num_train_timesteps,
            device=lq_latent.device,
            dtype=torch.float32,
        )
        predicted_flow = self.components.transformer(
            hidden_states=lq_latent,
            timestep=timestep,
            encoder_hidden_states=prompt_embedding,
            return_dict=False,
        )[0]
        sigma = lq_latent.new_tensor(sigma_value)
        return lq_latent - sigma * predicted_flow

    @override
    def validation_step(
        self,
        eval_data: Dict[str, Any],
        pipe: WanPipeline,
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        del pipe
        video = eval_data["video_tensor"]
        if isinstance(video, tuple):
            video = video[0]

        height, width = video.shape[-2:]
        video = F.interpolate(
            video,
            size=(height * 4, width * 4),
            mode="bilinear",
            align_corners=False,
        )
        frame_transform = transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        video = torch.stack([frame_transform(frame) for frame in video], dim=0)
        video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

        with torch.no_grad():
            self.components.vae.to(self.accelerator.device)
            latent = self.encode_video(video)
            self.components.text_encoder.to(self.accelerator.device)
            prompt_embedding = self.encode_text(eval_data["prompt"])
            clean_latent = self.predict_clean_latent(latent, prompt_embedding)
            generated_video = self.decode_latents(clean_latent)
            generated_video = (generated_video * 0.5 + 0.5).clamp(0.0, 1.0)

        return [("video", generated_video[0])]
