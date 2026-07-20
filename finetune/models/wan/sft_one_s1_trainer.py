from .lora_one_s1_trainer import WanS1Trainer
from ..utils import register


class WanS1SFTTrainer(WanS1Trainer):
    pass


register("wan-s1", "sft", WanS1SFTTrainer)
