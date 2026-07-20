import sys
from pathlib import Path


finetune_dir = Path(__file__).resolve().parent
project_root = finetune_dir.parent

# Keep the local ``finetune/datasets`` package from shadowing the
# Hugging Face ``datasets`` package imported by Accelerate.
sys.path[:] = [
    entry for entry in sys.path if Path(entry or ".").resolve() != finetune_dir
]
sys.path.insert(0, str(project_root))

from finetune.models.utils import get_model_cls
from finetune.schemas import Args


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
