from .lora_one_s2_trainer import WanS2Trainer
from ..utils import register


class WanS2SFTTrainer(WanS2Trainer):
    pass


register("wan-s2", "sft", WanS2SFTTrainer)
