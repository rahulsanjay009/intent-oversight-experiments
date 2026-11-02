# When the tests are green, is the patch right?

An exploratory reanalysis of 6,500 published AI coding agent patches: 13 models, one agent harness
(mini-SWE-agent v2.0.0), and the 500 issues of SWE-bench Verified. It asks which checks give a reviewer
real evidence that a patch does what the issue asked. Not peer reviewed.

**Start with [the results notebook](notebooks/results.ipynb).** It renders on GitHub, with no need to run anything.

## Main results

- 23.1% of patches that kept regression tests green still failed the tests written for the issue
  (95% CI 20.0–26.4%). This is an upper bound, because the benchmark runs only part of each project's test suite.
- Being green barely moved the pass rate, from 75.4% of all patches to 76.9%.
- Cheap signals about a patch found hard issues, but did not reliably separate good from bad patches on the
  same issue (within-issue AUC 0.56, 95% CI 0.44–0.70). The planned hypotheses here were not supported.

The planned follow-up, tests written from the issue alone (E4), has not been run.

## Contents

| Path | What it is |
|---|---|
| `ANALYSIS_PLAN.md` | Hypotheses, metrics and hold-out split, written before the download, with a log of deviations |
| `scripts/fetch_cohort.py` | Downloads published reports and patches |
| `scripts/analyze.py` | Runs the planned analyses, E1 to E3 |
| `scripts/test_analyze.py` | Checks the analysis code on hand-computed inputs |
| `scripts/exploratory.py` | Unplanned follow-ups and integrity checks |
| `scripts/build_notebook.py` | Builds the notebook from `results/` |
| `results/` | Per-patch table, results files and figures |

Reproduction steps and limitations are in the notebook. Raw benchmark data is not included. It comes from the
[SWE-bench experiments](https://github.com/SWE-bench/experiments) records.

Analysis code and text were written with help from Claude Code.
