"""Sanity checks and follow-ups that were NOT in ANALYSIS_PLAN.md. Report them as exploratory.

Usage: python -I scripts/exploratory.py <raw_dir> <results_dir>
Does not touch the E3 hold-out evaluation.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("analyze", Path(__file__).with_name("analyze.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

# Published "resolved" percentages from each entry's metadata.yaml in SWE-bench/experiments.
LEADERBOARD = {"claude-4-5-haiku-high": 66.6, "claude-4-5-opus-high": 76.8, "claude-4-5-sonnet-high": 71.4,
               "claude-4-6-opus": 75.6, "deepseek-3-2-high": 70.0, "gemini-3-flash-high": 75.8,
               "gemini-3-pro-high": 69.6, "glm-5-high": 72.8, "gpt-5-2-codex": 72.8, "gpt-5-2-high": 72.8,
               "gpt-5-mini": 56.2, "kimi-k2-5-high": 70.8, "minimax-2-5-high": 75.8}


def main():
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = pd.read_csv(out / "patches.csv")
    for col in ["official_resolved"]:
        df[col] = df[col].map({True: True, False: False, "True": True, "False": False}).astype("boolean").fillna(False)
    green = df[df.regression_green]
    ev = df[df.evaluable]

    zero = df[df.p2p_expected == 0].instance_id.unique()
    bands = []
    for low, high, label in [(1, 9, "1-9"), (10, 49, "10-49"), (50, 199, "50-199"), (200, 10**7, "200+")]:
        s = green[green.p2p_expected.between(low, high)]
        bands.append({"regression_tests": label, "green_patches": len(s), "issues": int(s.instance_id.nunique()),
                      "false_acceptance": float(s.wrong_but_green.mean())})
    kept = ev[~ev.instance_id.isin(zero)]
    per_issue = green.groupby("instance_id").wrong_but_green.agg(["sum", "count"])
    wrong = green[green.wrong_but_green]

    edits = []
    for row in ev[ev.edits_tests == 1].itertuples():
        text = (raw / "runs").glob(f"*_{row.model}/{row.instance_id}/patch.diff")
        files = a.parse_diff(next(text).read_text(encoding="utf-8", errors="replace"))
        tests = [f for f in files if a.is_test(f["path"])]
        added, removed = sum(f["add"] for f in tests), sum(f["del"] for f in tests)
        edits.append({"model": row.model, "instance_id": row.instance_id, "files": [f["path"] for f in tests],
                      "lines_added": added, "lines_removed": removed, "intent_pass": bool(row.intent_pass),
                      "hidden_file": bool(row.edits_hidden_tests)})

    board = []
    for model, published in LEADERBOARD.items():
        ours = int(df[df.model == model].official_resolved.sum())
        board.append({"model": model, "ours": ours, "published": round(published * 5),
                      "difference": ours - round(published * 5)})

    prompts = {}
    for path in sorted((raw / "sample_traj").glob("*.json")):
        messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
        text = next(m["content"] for m in messages if m["role"] == "user")
        text = text if isinstance(text, str) else json.dumps(text)
        prompts[path.stem.split("_", 2)[2]] = {
            "hash": hashlib.sha256(text.encode()).hexdigest()[:12],
            "forbids_test_edits": "DO NOT MODIFY: Tests" in text}

    results = {
        "false_acceptance_by_regression_tests": bands,
        "issues_without_regression_tests": len(zero),
        "green_patches_on_those": int(green.instance_id.isin(zero).sum()),
        "false_acceptance_excluding_them": float(kept.wrong_but_green.sum() / kept.regression_green.sum()),
        "issues_in_green_population": int(green.instance_id.nunique()),
        "issues_with_any_wrong_but_green": int((per_issue["sum"] > 0).sum()),
        "issues_where_all_green_patches_fail": int((per_issue["sum"] == per_issue["count"]).sum()),
        "wrong_but_green_all_intent_tests_fail": int((wrong.f2p_ok == 0).sum()),
        "wrong_but_green_some_intent_tests_fail": int((wrong.f2p_ok > 0).sum()),
        "issues_per_difficulty": {k: int(v) for k, v in green.groupby("difficulty").instance_id.nunique().items()},
        "test_edits": edits,
        "test_edits_removing_lines": sum(e["lines_removed"] > 0 for e in edits),
        "test_edits_add_only": sum(e["lines_removed"] == 0 for e in edits),
        "leaderboard": board,
        "prompt_sample_instance": "django__django-11099",
        "prompts": prompts,
        "new_file_patches": sum("new file mode" in p.read_text(encoding="utf-8", errors="replace")
                                for p in (raw / "runs").glob("*/*/patch.diff")),
    }
    (out / "exploratory.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("test_edits", "prompts")}, indent=1))
    for e in edits:
        print(e["model"].ljust(24), e["instance_id"].ljust(28), f"+{e['lines_added']}/-{e['lines_removed']}",
              "pass" if e["intent_pass"] else "FAIL", e["files"][:2])


if __name__ == "__main__":
    main()
