#!/usr/bin/env python3
"""Compare two OpenAI-compatible endpoints with identical requests."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--dspark-url", required=True)
    parser.add_argument("--model", default="target")
    parser.add_argument("--input", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def prompts(path: str, limit: int) -> list[str]:
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        value = row.get("input") or row.get("prompt")
        if isinstance(value, str) and value.strip():
            result.append(value)
        if len(result) >= limit:
            break
    if not result:
        raise RuntimeError("no prompts found")
    return result


def request(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    tokens = int(body["usage"]["completion_tokens"])
    return {"seconds": elapsed, "tokens": tokens, "tok_s": tokens / elapsed,
            "text": body["choices"][0]["message"]["content"]}


def summarize(runs: list[dict]) -> dict:
    speeds = [item["tok_s"] for item in runs]
    latencies = sorted(item["seconds"] for item in runs)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    return {"mean_tok_s": statistics.mean(speeds), "median_tok_s": statistics.median(speeds),
            "p50_seconds": statistics.median(latencies), "p95_seconds": p95,
            "tokens": sum(item["tokens"] for item in runs)}


def main() -> None:
    args = arguments()
    rows = prompts(args.input, args.requests)
    baseline = [request(args.baseline_url, args.model, p, args.max_tokens) for p in rows]
    dspark = [request(args.dspark_url, args.model, p, args.max_tokens) for p in rows]
    report = {"baseline": summarize(baseline), "dspark": summarize(dspark)}
    report["speedup"] = report["dspark"]["mean_tok_s"] / report["baseline"]["mean_tok_s"]
    report["exact_match_rate"] = sum(a["text"] == b["text"] for a, b in zip(baseline, dspark)) / len(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
