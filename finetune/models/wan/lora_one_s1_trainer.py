from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from typing_extensions import override

from .base import WanTrainerBase
from ..utils import register


class WanS1Trainer(WanTrainerBase):
    """Stage-1 Wan trainer: one-step LQ-to-HQ regression in latent space."""

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "hq_videos": torch.stack([sample["hq_video"] for sample in samples]),
            "lq_videos": torch.stack([sample["lq_video"] for sample in samples]),
            "prompt_embeddings": torch.stack(
                [sample["prompt_embedding"] for sample in samples]
            ),
        }
        encoded_hq = [sample["encoded_hq_video"] for sample in samples]
        encoded_lq = [sample["encoded_lq_video"] for sample in samples]
        if all(latent is not None for latent in encoded_hq):
            batch["encoded_hq_videos"] = torch.cat(encoded_hq, dim=0)
        if all(latent is not None for latent in encoded_lq):
            batch["encoded_lq_videos"] = torch.cat(encoded_lq, dim=0)
        return batch

    @override
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.args.is_latent:
            lq_latent = batch["encoded_lq_videos"].to(self.accelerator.device)
            hq_latent = batch["encoded_hq_videos"].to(self.accelerator.device)
        else:
            with torch.no_grad():
                self.components.vae.to(self.accelerator.device)
                mixed_video = torch.cat(
                    [batch["lq_videos"], batch["hq_videos"]], dim=0
                ).to(self.accelerator.device)
                mixed_latent = self.encode_video(mixed_video)
                lq_latent, hq_latent = mixed_latent.chunk(2, dim=0)

        predicted_hq_latent = self.predict_clean_latent(
            lq_latent,
            batch["prompt_embeddings"],
        )
        return F.mse_loss(
            predicted_hq_latent.float(),
            hq_latent.float(),
            reduction="mean",
        )


register("wan-s1", "lora", WanS1Trainer)
