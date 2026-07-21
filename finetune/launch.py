#!/usr/bin/env python3
"""Validate an experiment config and launch it through Accelerate."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


FINETUNE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FINETUNE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetune.schemas import Args


ALLOWED_TOP_LEVEL_KEYS = {
    "name",
    "runtime",
    "environment",
    "args",
    "launch_metadata",
}
REQUIRED_TRAIN_ARGS = {
    "model_path",
    "model_name",
    "model_type",
    "training_type",
    "output_dir",
    "report_to",
    "data_root",
    "video_column",
    "train_resolution",
}


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def set_nested(config: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"Override must use key=value syntax: {assignment}")
    key, raw_value = assignment.split("=", 1)
    parts = key.split(".") if "." in key else ["args", key]
    if any(not part for part in parts):
        raise ValueError(f"Invalid override key: {key}")

    target = config
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot assign below non-mapping key: {part}")
        target = child
    target[parts[-1]] = yaml.safe_load(raw_value)


def value_to_cli(value: Any) -> list[str]:
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        raise TypeError("Nested mappings cannot be passed as trainer arguments")
    return [str(value)]


def args_to_cli(train_args: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in train_args.items():
        if value is None:
            continue
        result.append(f"--{key}")
        result.extend(value_to_cli(value))
    return result


def resolve_from_finetune(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (FINETUNE_DIR / path).resolve()


def find_passthrough_value(arguments: list[str], option: str) -> str | None:
    value = None
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            value = arguments[index + 1]
        elif argument.startswith(f"{option}="):
            value = argument.split("=", 1)[1]
    return value


def get_output_dir(config: dict[str, Any], passthrough: list[str]) -> Path:
    output_value = find_passthrough_value(passthrough, "--output_dir")
    if output_value is None:
        output_value = str(config["args"]["output_dir"])
    return resolve_from_finetune(output_value)


def find_accelerate() -> str:
    sibling = Path(sys.executable).with_name("accelerate")
    if sibling.is_file():
        return str(sibling)
    executable = shutil.which("accelerate")
    if executable is None:
        raise FileNotFoundError(
            "Could not find accelerate next to the selected Python interpreter or on PATH"
        )
    return executable


def with_tee(command: list[str], log_path: Path) -> list[str]:
    tee = shutil.which("tee")
    if tee is None:
        raise FileNotFoundError("tee is required for runtime log capture")
    pipeline = (
        f"{shlex.join(command)} 2>&1 | "
        f"{shlex.quote(tee)} {shlex.quote(str(log_path))}"
    )
    return ["/bin/bash", "-o", "pipefail", "-c", pipeline]


def validate_config(config: dict[str, Any]) -> None:
    unknown = set(config) - ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    if not isinstance(config.get("runtime"), dict):
        raise ValueError("Config must contain a runtime mapping")
    if not isinstance(config.get("args"), dict):
        raise ValueError("Config must contain an args mapping")

    missing = REQUIRED_TRAIN_ARGS - set(config["args"])
    if missing:
        raise ValueError(f"Missing required trainer args: {sorted(missing)}")

    unknown_args = set(config["args"]) - set(Args.model_fields)
    if unknown_args:
        raise ValueError(f"Unknown trainer args in config: {sorted(unknown_args)}")

    # Args.parse_args performs this conversion after argparse. Mirror it here
    # so every composed YAML is type-checked before Accelerate starts ranks.
    validation_args = copy.deepcopy(config["args"])
    resolution = validation_args.get("train_resolution")
    if isinstance(resolution, str):
        try:
            validation_args["train_resolution"] = tuple(
                int(part) for part in resolution.split("x")
            )
        except ValueError as error:
            raise ValueError(
                "train_resolution must use framesxheightxwidth syntax"
            ) from error
    Args.model_validate(validation_args)

    accelerate_config = config["runtime"].get("accelerate_config")
    if not accelerate_config:
        raise ValueError("runtime.accelerate_config is required")
    if not resolve_from_finetune(accelerate_config).is_file():
        raise FileNotFoundError(f"Accelerate config not found: {accelerate_config}")


def write_snapshot(
    config: dict[str, Any],
    config_path: Path,
    command: list[str],
    output_dir: Path,
    launched_at: dt.datetime,
    log_path: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = copy.deepcopy(config)
    snapshot["launch_metadata"] = {
        "source_config": str(config_path.resolve()),
        "launched_at": launched_at.isoformat(),
        "log_file": str(log_path) if log_path is not None else None,
    }
    (output_dir / "launch_config.resolved.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "launch_command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )


def parse_cli() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch a self-contained DOVE experiment configuration."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value; unqualified keys are placed under args.",
    )
    parser.add_argument("--entrypoint", help="Override runtime.entrypoint")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument(
        "--no-tee",
        action="store_true",
        help="Do not mirror stdout/stderr into output_dir/logs.",
    )
    return parser.parse_known_args()


def main() -> None:
    cli, passthrough = parse_cli()
    config = load_config(cli.config)
    for assignment in cli.overrides:
        set_nested(config, assignment)

    environment_entrypoint = os.environ.get("TRAIN_ENTRYPOINT")
    if cli.entrypoint:
        config.setdefault("runtime", {})["entrypoint"] = cli.entrypoint
    elif environment_entrypoint:
        config.setdefault("runtime", {})["entrypoint"] = environment_entrypoint

    validate_config(config)
    runtime = config["runtime"]
    entrypoint = resolve_from_finetune(runtime.get("entrypoint", "train.py"))
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Training entrypoint not found: {entrypoint}")

    accelerate_command = [
        find_accelerate(),
        "launch",
        "--config_file",
        str(resolve_from_finetune(runtime["accelerate_config"])),
        str(entrypoint),
        *args_to_cli(config["args"]),
        *passthrough,
    ]

    launched_at = dt.datetime.now().astimezone()
    output_dir = get_output_dir(config, passthrough)
    tee_enabled = bool(runtime.get("tee", True)) and not cli.no_tee
    log_path = None
    execution_command = accelerate_command
    if tee_enabled:
        log_path = output_dir / "logs" / f"train-{launched_at:%Y%m%d-%H%M%S}.log"
        execution_command = with_tee(accelerate_command, log_path)

    if cli.print_config:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    print(shlex.join(execution_command), flush=True)
    if cli.dry_run:
        return

    if not cli.no_snapshot:
        write_snapshot(
            config,
            cli.config,
            execution_command,
            output_dir,
            launched_at,
            log_path,
        )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    for key, value in config.get("environment", {}).items():
        environment[str(key)] = str(value)
    os.chdir(FINETUNE_DIR)
    os.execvpe(execution_command[0], execution_command, environment)


if __name__ == "__main__":
    main()
