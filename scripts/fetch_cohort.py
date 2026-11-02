"""Download per-instance results and patches for the mini-SWE-agent v2.0.0 cohort.

Usage: python -I scripts/fetch_cohort.py <output_dir>

Reads public objects anonymously. Re-running skips files that already exist.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset

BUCKET = "https://swe-bench-submissions.s3.amazonaws.com"
MODELS = [
    "20260217_mini-v2.0.0_claude-4-5-haiku-high",
    "20260217_mini-v2.0.0_claude-4-5-opus-high",
    "20260217_mini-v2.0.0_claude-4-5-sonnet-high",
    "20260217_mini-v2.0.0_claude-4-6-opus",
    "20260217_mini-v2.0.0_deepseek-3-2-high",
    "20260217_mini-v2.0.0_gemini-3-flash-high",
    "20260217_mini-v2.0.0_glm-5-high",
    "20260217_mini-v2.0.0_gpt-5-2-high",
    "20260217_mini-v2.0.0_gpt-5-mini",
    "20260217_mini-v2.0.0_kimi-k2-5-high",
    "20260217_mini-v2.0.0_minimax-2-5-high",
    "20260219_mini-v2.0.0_gpt-5-2-codex",
    "20260226_mini-v2.0.0_gemini-3-pro-high",
]
FILES = ["report.json", "patch.diff"]


def fetch(job):
    url, dest = job
    if dest.exists():
        return "cached"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                body = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(body)
            tmp.replace(dest)
            return "ok"
        except urllib.error.HTTPError as err:
            if err.code in (403, 404):
                return f"missing:{err.code}"
            time.sleep(2**attempt)
        except OSError:
            time.sleep(2**attempt)
    return "failed"


def main():
    out = Path(sys.argv[1])
    dataset_path = out / "dataset.jsonl"
    rows = list(load_dataset("SWE-bench/SWE-bench_Verified", split="test"))
    out.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"dataset instances: {len(rows)}", flush=True)

    jobs = []
    for model in MODELS:
        for row in rows:
            iid = row["instance_id"]
            for name in FILES:
                url = f"{BUCKET}/bash-only/{model}/logs/{iid}/{name}"
                jobs.append((url, out / "runs" / model / iid / name))
    print(f"objects: {len(jobs)}", flush=True)

    counts = {}
    missing = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, (job, status) in enumerate(zip(jobs, pool.map(fetch, jobs)), 1):
            counts[status] = counts.get(status, 0) + 1
            if status.startswith("missing") or status == "failed":
                missing.append({"url": job[0], "status": status})
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} {counts}", flush=True)

    (out / "fetch_log.json").write_text(
        json.dumps({"counts": counts, "missing": missing}, indent=1), encoding="utf-8"
    )
    print(f"done {counts}", flush=True)


if __name__ == "__main__":
    main()
