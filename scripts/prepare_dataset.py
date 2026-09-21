#!/usr/bin/env python3
"""Validate private supervision data and write deterministic JSONL shards."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument("--local", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty {key!r}")
    return value.strip()


def canonical_json_output(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing non-empty 'output'")
    text = value.strip()
    match = re.fullmatch(r"<final_answer>\s*([\s\S]*?)\s*</final_answer>", text, re.I)
    if match:
        text = match.group(1).strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if match:
        text = match.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("output JSON must be an array")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def build_record(source: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")[-40:] or "source"
    return {
        "id": f"{slug}-{index:08d}",
        "conversations": [
            {"role": "system", "content": required_text(row, "system_prompt")},
            {"role": "user", "content": required_text(row, "input")},
            {
                "role": "assistant",
                "reasoning_content": required_text(row, "thinking"),
                "content": canonical_json_output(row.get("output")),
            },
        ],
    }


def source_rows(repos: Iterable[str], paths: Iterable[str]) -> Iterator[tuple[str, int, dict]]:
    for value in paths:
        path = Path(value).expanduser().resolve()
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if line.strip():
                    yield path.stem, index, json.loads(line)
    for repo in repos:
        from datasets import load_dataset

        for index, row in enumerate(load_dataset(repo, split="train", token=True)):
            yield repo.rsplit("/", 1)[-1], index, dict(row)


def write_shards(records: list[dict], output: Path, size: int, seed: int) -> dict:
    if size <= 0:
        raise ValueError("shard-size must be positive")
    random.Random(seed).shuffle(records)
    output.mkdir(parents=True, exist_ok=True)
    shards = []
    for number, start in enumerate(range(0, len(records), size)):
        rows = records[start : start + size]
        name = f"shard-{number:05d}.jsonl"
        temporary = output / f".{name}.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output / name)
        shards.append({"file": name, "rows": len(rows)})
    manifest = {"version": 1, "seed": seed, "shard_size": size, "rows": len(records), "shards": shards}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"refusing to replace non-empty directory: {output}")
    if args.overwrite and output.exists():
        for path in output.glob("shard-*.jsonl"):
            path.unlink()
    repos = list(args.repo)
    if not repos:
        repos = [item.strip() for item in __import__("os").environ.get("DATASET_REPOS", "").split(",") if item.strip()]
    if not repos and not args.local:
        raise RuntimeError("provide --local, --repo, or DATASET_REPOS")
    records = [build_record(source, row, index) for source, index, row in source_rows(repos, args.local)]
    manifest = write_shards(records, output, args.shard_size, args.seed)
    print(f"Wrote {manifest['rows']} rows in {len(manifest['shards'])} shards to {output}")


if __name__ == "__main__":
    main()
