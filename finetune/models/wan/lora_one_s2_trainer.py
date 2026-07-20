import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from typing_extensions import override

from .base import WanTrainerBase
from ..utils import register


class WanS2Trainer(WanTrainerBase):
    """Stage-2 Wan trainer: pixel and temporal refinement through the VAE."""

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "hq_videos": torch.stack([sample["hq_video"] for sample in samples]),
            "lq_videos": torch.stack([sample["lq_video"] for sample in samples]),
            "hq_images": torch.stack([sample["hq_image"] for sample in samples]),
            "lq_images": torch.stack([sample["lq_image"] for sample in samples]),
            "prompt_embeddings": torch.stack(
                [sample["prompt_embedding"] for sample in samples]
            ),
        }

    def _encode_frames_independently(self, video: torch.Tensor) -> torch.Tensor:
        latents = [
            self.encode_video(video[:, :, frame_index : frame_index + 1])
            for frame_index in range(video.shape[2])
        ]
        return torch.cat(latents, dim=2)

    def _decode_frames_independently(self, latent: torch.Tensor) -> torch.Tensor:
        frames = [
            self.decode_latents(latent[:, :, frame_index : frame_index + 1])
            for frame_index in range(latent.shape[2])
        ]
        return torch.cat(frames, dim=2)

    def _compute_perceptual_loss(
        self,
        generated_video: torch.Tensor,
        hq_video: torch.Tensor,
    ) -> torch.Tensor:
        loss = generated_video.new_zeros((), dtype=torch.float32)
        for frame_index in range(generated_video.shape[2]):
            predicted_frame = generated_video[:, :, frame_index].float()
            target_frame = hq_video[:, :, frame_index].float()

            if self.args.ea_dists_weight > 0:
                loss = loss + self.dists_loss(predicted_frame, target_frame)
                loss = loss + self.dists_loss(
                    self.edge_detection_model(predicted_frame),
                    self.edge_detection_model(target_frame),
                )
            elif self.args.dists_weight > 0:
                loss = loss + self.dists_loss(predicted_frame, target_frame)
            elif self.args.ea_lpips_weight > 0:
                loss = loss + self.lpips_loss(predicted_frame, target_frame)
                loss = loss + self.lpips_loss(
                    self.edge_detection_model(predicted_frame),
                    self.edge_detection_model(target_frame),
                )
            elif self.args.lpips_weight > 0:
                loss = loss + self.lpips_loss(predicted_frame, target_frame)

        frame_count = generated_video.shape[2]
        if self.args.ea_dists_weight > 0:
            return loss / (frame_count * 2) * self.args.ea_dists_weight
        if self.args.dists_weight > 0:
            return loss / frame_count * self.args.dists_weight
        if self.args.ea_lpips_weight > 0:
            return loss / (frame_count * 2) * self.args.ea_lpips_weight
        if self.args.lpips_weight > 0:
            return loss / frame_count * self.args.lpips_weight
        return loss

    @override
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.args.is_latent:
            raise ValueError("Wan Stage-2 does not support cached latents")

        is_image_batch = random.random() < self.args.image_ratio
        if is_image_batch:
            lq_video = batch["lq_images"]
            hq_video = batch["hq_images"]
        else:
            lq_video = batch["lq_videos"]
            hq_video = batch["hq_videos"]

        with torch.no_grad():
            self.components.vae.to(self.accelerator.device)
            lq_video = lq_video.to(self.accelerator.device)
            lq_latent = self._encode_frames_independently(lq_video)

        hq_video = (hq_video * 0.5 + 0.5).clamp(0.0, 1.0)
        hq_video = hq_video.to(self.accelerator.device)

        predicted_hq_latent = self.predict_clean_latent(
            lq_latent,
            batch["prompt_embeddings"],
        )
        generated_video = self._decode_frames_independently(predicted_hq_latent)
        generated_video = (generated_video * 0.5 + 0.5).clamp(0.0, 1.0)

        mse_loss = F.mse_loss(
            generated_video.float(),
            hq_video.float(),
            reduction="mean",
        )
        perceptual_loss = self._compute_perceptual_loss(generated_video, hq_video)

        if generated_video.shape[2] > 1:
            generated_difference = (
                generated_video[:, :, 1:] - generated_video[:, :, :-1]
            )
            target_difference = hq_video[:, :, 1:] - hq_video[:, :, :-1]
            frame_difference_loss = (
                F.l1_loss(
                    generated_difference.float(),
                    target_difference.float(),
                )
                * self.args.frame_diff_weight
            )
        else:
            frame_difference_loss = generated_video.new_zeros((), dtype=torch.float32)

        loss = mse_loss + perceptual_loss + frame_difference_loss
        logs = {
            "perceptual_loss": perceptual_loss.detach().item(),
            "mse_loss": mse_loss.detach().item(),
            "frame_diff_loss": frame_difference_loss.detach().item(),
            "loss": loss.detach().item(),
        }
        return loss, logs


register("wan-s2", "lora", WanS2Trainer)
