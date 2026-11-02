"""Run E1 to E3 from ANALYSIS_PLAN.md on the downloaded cohort.

Usage: python -I scripts/analyze.py <raw_dir> <results_dir>
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261101
N_BOOT = 2000
TEST_DIR = re.compile(r"(^|/)(test|tests|testing)(/|$)")
TEST_NAME = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|tests\.py|conftest\.py)$")
STRAY_NAME = re.compile(r"reproduce|debug|scratch|tmp|temp|verify", re.I)
CONFIG = re.compile(r"(^|/)(setup\.py|requirements[^/]*|[^/]*\.toml|[^/]*\.cfg|[^/]*\.ini)$")
SIGNALS = ["log_lines", "src_files", "hunks", "edits_tests", "adds_tests",
           "stray_file", "config_touch", "consensus"]


def is_test(path):
    return bool(TEST_DIR.search(path) or TEST_NAME.search(path))


def parse_diff(text):
    """Return per-file change records from a unified diff."""
    files = []
    current = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+) b/(.+)$", line)
            current = {"path": match.group(2) if match else line, "new": False,
                       "deleted": False, "add": 0, "del": 0, "hunks": 0}
            files.append(current)
        elif current is None:
            continue
        elif line.startswith("new file mode"):
            current["new"] = True
        elif line.startswith("deleted file mode"):
            current["deleted"] = True
        elif line.startswith("@@"):
            current["hunks"] += 1
        elif line.startswith("+") and not line.startswith("+++"):
            current["add"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            current["del"] += 1
    return files


def patch_features(files, gold_test_files):
    src = [f for f in files if not is_test(f["path"])]
    tests = [f for f in files if is_test(f["path"])]
    stray = any(f["new"] and ("/" not in f["path"] or STRAY_NAME.search(f["path"].rsplit("/", 1)[-1]))
                for f in files)
    return {
        "src_set": frozenset(f["path"] for f in src),
        "log_lines": math.log1p(sum(f["add"] + f["del"] for f in src)),
        "src_files": len(src),
        "hunks": sum(f["hunks"] for f in src),
        "edits_tests": int(any(not f["new"] for f in tests)),
        "adds_tests": int(any(f["new"] for f in tests)),
        "edits_hidden_tests": int(any(f["path"] in gold_test_files for f in tests)),
        "stray_file": int(stray),
        "config_touch": int(any(CONFIG.search(f["path"]) for f in files)),
    }


def test_count(value):
    return len(json.loads(value) if isinstance(value, str) else value)


def integrity(df):
    """Check that reported outcomes agree with their own test lists and the dataset."""
    ran = df[df.status == "applied"]
    derived = (ran.f2p_fail == 0) & (ran.p2p_fail == 0)
    return {
        "resolved_disagrees_with_test_lists": int((derived != ran.official_resolved.astype(bool)).sum()),
        "f2p_count_differs_from_dataset": int(((ran.f2p_ok + ran.f2p_fail) != ran.f2p_expected).sum()),
        "p2p_count_differs_from_dataset": int(((ran.p2p_ok + ran.p2p_fail) != ran.p2p_expected).sum()),
        "applied_reports": len(ran),
        "nested_reports": int(ran.report_nested.astype(bool).sum()),
        "status_by_model": {m: {str(k): int(v) for k, v in g.status.value_counts().items()}
                            for m, g in df.groupby("model")},
    }


def load(raw):
    dataset = [json.loads(l) for l in (raw / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = {}
    for row in dataset:
        gold_tests = {f["path"] for f in parse_diff(row["test_patch"])}
        meta[row["instance_id"]] = {"repo": row["repo"], "difficulty": row["difficulty"],
                                    "gold_tests": gold_tests,
                                    "f2p_expected": test_count(row["FAIL_TO_PASS"]),
                                    "p2p_expected": test_count(row["PASS_TO_PASS"])}
    records = []
    for model_dir in sorted((raw / "runs").iterdir()):
        model = model_dir.name.split("_", 2)[2]
        for iid, info in meta.items():
            rec = {"model": model, "instance_id": iid, "repo": info["repo"],
                   "difficulty": info["difficulty"], "status": "no_report",
                   "f2p_expected": info["f2p_expected"], "p2p_expected": info["p2p_expected"]}
            report_path = model_dir / iid / "report.json"
            diff_path = model_dir / iid / "patch.diff"
            text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
            files = parse_diff(text)
            rec.update(patch_features(files, info["gold_tests"]))
            rec["no_patch"] = not files
            if report_path.exists():
                data = json.loads(report_path.read_text(encoding="utf-8"))
                # Published reports are either flat or nested under the instance id.
                report = data.get(iid, data)
                status = report.get("tests_status") or {}
                f2p = status.get("FAIL_TO_PASS", {"success": [], "failure": []})
                p2p = status.get("PASS_TO_PASS", {"success": [], "failure": []})
                applied = report.get("patch_successfully_applied", False)
                rec.update({
                    "status": "not_applied" if not applied else "applied" if status else "no_test_status",
                    "report_nested": iid in data,
                    "official_resolved": bool(report.get("resolved")),
                    "f2p_fail": len(f2p["failure"]), "f2p_ok": len(f2p["success"]),
                    "p2p_fail": len(p2p["failure"]), "p2p_ok": len(p2p["success"]),
                })
            records.append(rec)
    df = pd.DataFrame(records)
    ok = df.status == "applied"
    df["evaluable"] = ok & ~df.no_patch
    df["regression_green"] = df.evaluable & (df.p2p_fail.fillna(0) == 0)
    df["intent_pass"] = df.evaluable & (df.f2p_fail.fillna(1) == 0)
    df["wrong_but_green"] = df.regression_green & ~df.intent_pass
    df["holdout"] = df.instance_id.map(
        lambda i: int(hashlib.sha256(i.encode()).hexdigest(), 16) % 10 < 3)
    return df


def add_consensus(df):
    values = np.full(len(df), np.nan)
    for _, group in df[df.evaluable].groupby("instance_id"):
        sets = list(group.src_set)
        for pos, idx in enumerate(group.index):
            others = [s for j, s in enumerate(sets) if j != pos]
            sims = [len(sets[pos] & s) / len(sets[pos] | s) if sets[pos] | s else 1.0 for s in others]
            values[df.index.get_loc(idx)] = float(np.mean(sims)) if sims else np.nan
    df["consensus"] = values
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - r) / d, (c + r) / d)


def cluster_boot(df, stat, rng):
    """Percentile CI for stat(df), resampling instances with replacement."""
    groups = {k: v for k, v in df.groupby("instance_id")}
    keys = list(groups)
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        sample = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
        try:
            value = stat(sample)
        except ValueError:
            continue
        if value == value:
            out.append(value)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def far(frame):
    return frame.wrong_but_green.sum() / frame.regression_green.sum()


def e1(df, rng):
    green = df[df.regression_green]
    per_model = []
    for model, group in df.groupby("model"):
        k, n = int(group.wrong_but_green.sum()), int(group.regression_green.sum())
        low, high = wilson(k, n)
        per_model.append({"model": model, "patches": len(group), "evaluable": int(group.evaluable.sum()),
                          "regression_green": n, "wrong_but_green": k,
                          "false_acceptance": k / n if n else None, "ci_low": low, "ci_high": high,
                          "official_resolved": int(group.official_resolved.fillna(False).astype(bool).sum())})
    def strata(col):
        return [{col: key, "regression_green": len(g), "wrong_but_green": int(g.wrong_but_green.sum()),
                 "false_acceptance": float(g.wrong_but_green.mean())}
                for key, g in green.groupby(col)]
    return {
        "pooled_false_acceptance": float(far(df[df.evaluable])),
        "pooled_ci": cluster_boot(df[df.evaluable], far, rng),
        "pooled_regression_green": int(df.regression_green.sum()),
        "pooled_wrong_but_green": int(df.wrong_but_green.sum()),
        "per_model": per_model, "by_difficulty": strata("difficulty"), "by_repo": strata("repo"),
        "status_counts": {str(k): int(v) for k, v in df.status.value_counts().items()},
        "empty_patches": int(df.no_patch.sum()),
    }


def e2(df):
    ev = df[df.evaluable]
    rows = []
    for name in ["edits_tests", "adds_tests", "edits_hidden_tests"]:
        with_, without = ev[ev[name] == 1], ev[ev[name] == 0]
        diffs = []
        for _, group in ev.groupby("instance_id"):
            a, b = group[group[name] == 1], group[group[name] == 0]
            if len(a) and len(b):
                diffs.append(a.intent_pass.mean() - b.intent_pass.mean())
        rows.append({"behaviour": name, "share_of_patches": float(ev[name].mean()), "n_with": len(with_),
                     "intent_pass_with": float(with_.intent_pass.mean()) if len(with_) else None,
                     "intent_pass_without": float(without.intent_pass.mean()) if len(without) else None,
                     "within_instance_diff": float(np.mean(diffs)) if diffs else None,
                     "instances_mixed": len(diffs)})
    per_model = ev.groupby("model")[["edits_tests", "adds_tests", "edits_hidden_tests"]].mean()
    return {"overall": rows, "per_model": per_model.reset_index().to_dict("records")}


def within_auc(frame, score_col):
    """Share of wrong/right pairs on one issue where the wrong patch ranks riskier."""
    wins = ties = pairs = 0
    for _, group in frame.groupby("instance_id"):
        wrong = group[group.wrong_but_green][score_col].to_numpy()
        right = group[~group.wrong_but_green][score_col].to_numpy()
        if len(wrong) and len(right):
            delta = wrong[:, None] - right[None, :]
            wins += int((delta > 0).sum())
            ties += int((delta == 0).sum())
            pairs += delta.size
    return (wins + ties / 2) / pairs if pairs else float("nan")


def recall_at(frame, score_col, share=0.2):
    k = max(1, int(round(len(frame) * share)))
    # Random tie-break, so discrete scores do not borrow from row order.
    jitter = np.random.default_rng(SEED).random(len(frame))
    order = np.lexsort((jitter, -frame[score_col].to_numpy()))
    return frame.wrong_but_green.to_numpy()[order[:k]].sum() / frame.wrong_but_green.sum()


def evaluate(frame, score_col, rng):
    plain = lambda s: roc_auc_score(s.wrong_but_green, s[score_col])
    within = lambda s: within_auc(s, score_col)
    recall = lambda s: recall_at(s, score_col)
    return {"auc": float(plain(frame)), "auc_ci": cluster_boot(frame, plain, rng),
            "within_auc": float(within(frame)), "within_auc_ci": cluster_boot(frame, within, rng),
            "recall_top20": float(recall(frame)), "recall_top20_ci": cluster_boot(frame, recall, rng)}


def fit(train, columns):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(train[columns], train.wrong_but_green.astype(int))


def e3(df, rng):
    pop = df[df.regression_green].dropna(subset=SIGNALS).copy()
    dev, hold = pop[~pop.holdout].copy(), pop[pop.holdout].copy()
    model = fit(dev, SIGNALS)
    hold["risk"] = model.predict_proba(hold[SIGNALS])[:, 1]
    # Consensus is scored so that low overlap means high risk.
    hold["consensus_risk"] = -hold.consensus
    results = {
        "dev_patches": len(dev), "dev_wrong": int(dev.wrong_but_green.sum()),
        "holdout_patches": len(hold), "holdout_wrong": int(hold.wrong_but_green.sum()),
        "holdout_instances": int(hold.instance_id.nunique()),
        "coefficients": dict(zip(SIGNALS, map(float, model[-1].coef_[0]))),
        "full_model": evaluate(hold, "risk", rng),
        "consensus_only": evaluate(hold, "consensus_risk", rng),
    }
    no_consensus = [s for s in SIGNALS if s != "consensus"]
    hold["risk_static"] = fit(dev, no_consensus).predict_proba(hold[no_consensus])[:, 1]
    results["single_patch_signals"] = evaluate(hold, "risk_static", rng)
    per_model = []
    for name in sorted(pop.model.unique()):
        target = hold[hold.model == name].copy()
        target["risk_lomo"] = fit(dev[dev.model != name], SIGNALS).predict_proba(target[SIGNALS])[:, 1]
        if target.wrong_but_green.nunique() == 2:
            per_model.append({"model": name, "n": len(target), "wrong": int(target.wrong_but_green.sum()),
                              "auc": float(roc_auc_score(target.wrong_but_green, target.risk_lomo))})
    results["unseen_model"] = per_model
    return results


def main():
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = add_consensus(load(raw))
    results = {"cohort": {"models": int(df.model.nunique()), "instances": int(df.instance_id.nunique()),
                          "patches": len(df)},
               "integrity": integrity(df), "E1": e1(df, rng), "E2": e2(df), "E3": e3(df, rng)}
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    df.drop(columns=["src_set"]).to_csv(out / "patches.csv", index=False)
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
