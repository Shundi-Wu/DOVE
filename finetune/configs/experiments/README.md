# Experiment configurations

Each YAML file is a complete, self-contained experiment definition. There is no
base/model/stage inheritance: opening one file shows every value supplied to the trainer.

By default, `runtime.tee: true` mirrors stdout and stderr to both the terminal and:

```text
<output_dir>/logs/train-YYYYMMDD-HHMMSS.log
```

The launcher uses Bash `pipefail`, so an Accelerate/training failure remains a failed
process even when `tee` exits successfully. Pass `--no-tee` for a one-off launch without
the file log.

Launch through the existing wrappers:

```bash
bash finetune/train_ddp_wan_s1.sh
```

Override a value without copying the launcher:

```bash
bash finetune/train_ddp_wan_s1.sh --set learning_rate=1e-5
```

Inspect the command without starting training:

```bash
bash finetune/train_ddp_wan_s1.sh --print-config --dry-run
```

Network implementations remain under `finetune/models/`. A new architecture needs a
registry name and one new experiment YAML, not another shell launcher.
