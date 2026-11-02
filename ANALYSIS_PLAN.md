# Analysis plan: what can check an AI coding agent's patch?

Written before the cohort was downloaded. Seen beforehand: published leaderboard
resolve rates, one sample `report.json`/`patch.diff` (django__django-11099, Claude 4.6
Opus), the dataset's field names, and Wang & Pradel's "Are 'Solved Issues' in SWE-bench
Really Solved Correctly?" Any departure from this plan is logged at the bottom and
reported as exploratory.

## Question

When an agent's patch has to be accepted or rejected, which checks carry real evidence
that it does what the issue asked, and which only look like evidence?

## Data

- Benchmark: SWE-bench Verified (500 human-curated issues from 12 Python projects).
- Patches: published runs of a single harness, mini-SWE-agent v2.0.0, across 13 models,
  so model differences are not confounded with agent differences.
- Per-patch outcome: `report.json` from the official evaluation.
- No patch is generated or re-evaluated by us in E1 to E3.

## Definitions

- `regression_green`: the patch applied, and no `PASS_TO_PASS` test failed. These tests
  passed before the fix, so they stand in for "existing tests stayed green".
- `intent_pass`: no `FAIL_TO_PASS` test failed. These tests were written with the human
  fix and encode the behaviour the issue asked for.
- `wrong_but_green`: `regression_green` and not `intent_pass`.
- Empty or unappliable patches are counted and reported, and excluded from gate rates.

## E1: How much does a green regression run overstate correctness?

- Metric: false-acceptance rate, `P(not intent_pass | regression_green)`, per model and
  pooled, with the pooled figure also split by the dataset's `difficulty` and by repo.
- Per-model CI: Wilson 95%. Pooled CI: cluster bootstrap over instances, 2,000 resamples.
- H1: the pooled 95% CI excludes 0.
- H1b: false-acceptance rate is higher for harder difficulty strata.

## E2: Do agents touch the tests that are supposed to judge them?

- A file is a test file if its path has a `test`, `tests` or `testing` directory, or its
  name matches `test_*.py`, `*_test.py`, `tests.py` or `conftest.py`.
- Metrics, per model and pooled: share of non-empty patches that edit an existing test
  file; that add a new test file; that edit a file also changed by the benchmark's
  hidden `test_patch` (the harness overwrites those edits).
- Comparison: `intent_pass` rate with and without each behaviour, overall and within
  instance. This is descriptive: agents choose to edit tests, so it is not causal.

## E3: Can cheap, deterministic signals direct a human reviewer's attention?

Population: non-empty `regression_green` patches. Target: `wrong_but_green`.

Signals, computed only from what is available without hidden tests:

1. `log_lines`: log(1 + added + deleted lines), excluding test files
2. `src_files`: number of non-test files changed
3. `hunks`: number of hunks, excluding test files
4. `edits_tests`: edits an existing test file
5. `adds_tests`: adds a new test file
6. `stray_file`: adds a new file at the repo root, or a new file whose name contains
   `reproduce`, `debug`, `scratch`, `tmp`, `temp`, or `verify`
7. `config_touch`: changes packaging or config (`setup.py`, `setup.cfg`, `pyproject.toml`,
   `tox.ini`, `requirements*`, `*.toml`, `*.cfg`, `*.ini`)
8. `consensus`: mean Jaccard overlap between this patch's non-test files and those of
   the other 12 models on the same issue

Protocol:

- Hold-out: an instance is in the hold-out if `int(sha256(instance_id), 16) % 10 < 3`.
  All tuning uses the other instances. Hold-out metrics are computed once.
- Model: standardised logistic regression, default L2, trained on development instances.
- Metrics on hold-out: AUC; recall of `wrong_but_green` when reviewing the top 20% of
  patches by risk, against the 20% expected from random review; and within-instance AUC
  (over pairs on one issue with one wrong and one right patch, how often is the wrong one
  ranked riskier). Within-instance AUC is the headline metric because plain AUC can be
  inflated by signals that merely detect hard issues.
- CI: cluster bootstrap over instances, 2,000 resamples.
- Unseen-model check: for each model, retrain on development instances without that
  model, and score that model on hold-out instances.
- H3: the within-instance AUC 95% CI for the full model excludes 0.5.
- H3b: `consensus` alone has within-instance AUC above 0.5, with a CI excluding 0.5.

## E4: Do tests written from the issue alone catch what the green run missed?

Needs an LLM and Docker. Fixed before any E1 to E3 analysis so those results cannot
shape the question; sample details are finalised only by counts of eligible patches.

- For sampled issues, generate tests from the issue text and repository at the base
  commit, with no access to any patch, and freeze them.
- Run the frozen tests on the human patch and on agent patches.
- False-alarm rate: frozen tests that fail on the human patch. These are discarded per
  test, and the discard rate is reported rather than hidden.
- Catch rate: `wrong_but_green` patches failing at least one surviving test.
- Over-rejection rate: `intent_pass` patches failing at least one surviving test. Each is
  a disagreement to inspect, not automatically an error.
- Fixed comparison: tests from issue text alone against tests that may also read the
  agent's patch, to measure whether exposure to the implementation weakens the oracle.

## Threats to validity, stated in advance

- `PASS_TO_PASS` covers only test files touched by the human test patch, not the project's
  full suite. A real CI run may catch more, so E1 is an upper bound on what a full CI
  would let through. It must be described that way.
- Hidden tests are an imperfect oracle in both directions. Wang & Pradel show that some
  patches pass them and are still wrong; others may be over-fitted to the human fix.
- Benchmark tasks are open-source Python bug fixes, not industrial work.
- Thirteen models answering the same 500 issues are not independent, hence clustering
  by instance.
- Published runs are taken as reported. We do not re-execute them.

## Deviations log

1. Parser only, no change to any hypothesis or metric. Published reports come in two
   layouts (flat, and nested under the instance id), which the first run did not handle.
   Added integrity checks against the dataset and the leaderboard at the same time. This
   crash happened before any outcome was computed.
2. E2 cannot answer its question as written. Every model received an identical prompt
   saying "DO NOT MODIFY: Tests, configuration files" and telling it to diff only the source
   files it changed (checked on one issue per model). So edit rates measure breaches of an
   explicit instruction, not spontaneous behaviour. The numbers are reported under that
   meaning. Found after the E1 to E3 run.
3. Consequence for E3: `adds_tests`, `stray_file` and `config_touch` have no variance in the
   development data for the same reason, so the model effectively has five signals. The
   hold-out evaluation was not re-run and no signal was changed.
4. Added `scripts/exploratory.py` after seeing results: integrity follow-ups, and false
   acceptance by number of regression tests. All of it is exploratory, not confirmatory.
5. Added after seeing results, in `scripts/build_notebook.py`: the pass rate of all, green and
   red patches. It is exploratory, including where the summary and the outreach email quote it.
6. Headline E2 count corrected after seeing results. The planned path rule gives 34 patches,
   but 3 of them only change `django/test/testcases.py`, which is framework source, so 31 are
   reported. Both numbers are shown.

The SHA-256 in `ANALYSIS_PLAN.sha256` covers this file as first written, with LF line
endings and this section reading only "None yet."
