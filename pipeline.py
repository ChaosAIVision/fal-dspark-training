#!/usr/bin/env python3
"""Small command builder around the pinned TorchSpec checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TORCHSPEC = ROOT / "upstream" / "TorchSpec"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("materialize", "train", "export"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--data")
    parser.add_argument("--offline-data")
    parser.add_argument("--load-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TORCHSPEC) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MC_STORE_MEMCPY", "0")
    return env


def latest_checkpoint(root: Path) -> Path:
    tracker = root / "latest_checkpointed_iteration.txt"
    iteration = int(tracker.read_text().strip())
    model = root / f"iter_{iteration:07d}" / "model"
    if not (model / ".metadata").is_file():
        raise RuntimeError(f"invalid checkpoint: {model}")
    return model


def main() -> None:
    value = args()
    if value.stage in {"materialize", "train"} and (not value.data or not value.offline_data):
        raise SystemExit("--data and --offline-data are required")
    if value.stage == "materialize":
        command = [
            sys.executable, "-m", "torchspec.offline.generate", "--config",
            str(ROOT / "configs/single_gpu/materialize.yaml"), "--output", value.offline_data,
            f"model.target_model_path={value.target}", f"dataset.train_data_path={value.data}",
            f"training.max_seq_length={value.max_seq_length}",
        ]
        if value.overwrite:
            command.append("--overwrite")
    elif value.stage == "train":
        if not value.load_path or not value.output_dir:
            raise SystemExit("--load-path and --output-dir are required")
        command = [
            sys.executable, "-m", "torchspec.train_entry", "--config",
            str(ROOT / "configs/single_gpu/dspark.yaml"),
            f"model.target_model_path={value.target}", f"dataset.train_data_path={value.data}",
            f"inference.offline.data_path={value.offline_data}",
            f"training.load_path={value.load_path}", f"output_dir={value.output_dir}",
            f"training.max_seq_length={value.max_seq_length}",
        ]
    else:
        if not value.load_path or not value.output_dir:
            raise SystemExit("--load-path and --output-dir are required")
        command = [
            sys.executable, str(TORCHSPEC / "tools/convert_to_hf.py"),
            "--input-dir", str(latest_checkpoint(Path(value.load_path))),
            "--output-dir", value.output_dir,
            "--config", str(ROOT / "configs/single_gpu/dspark_draft_config.json"),
            "--target-model-path", value.target,
            "--embedding-key", "model.language_model.embed_tokens", "--trust-remote-code",
        ]
    print("+", __import__("shlex").join(command), flush=True)
    if not value.dry_run:
        subprocess.run(command, cwd=ROOT, env=environment(), check=True)


if __name__ == "__main__":
    main()
