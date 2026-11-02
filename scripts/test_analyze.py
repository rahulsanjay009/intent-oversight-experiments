"""Checks for analyze.py on hand-computed inputs. Run: python -I scripts/test_analyze.py"""
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("analyze", Path(__file__).with_name("analyze.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

DIFF = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,2 +1,3 @@
 keep
-old
+new
+extra
@@ -9 +10 @@
-a
+b
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-x
+y
diff --git a/reproduce_bug.py b/reproduce_bug.py
new file mode 100644
--- /dev/null
+++ b/reproduce_bug.py
@@ -0,0 +1 @@
+print(1)
diff --git a/setup.cfg b/setup.cfg
--- a/setup.cfg
+++ b/setup.cfg
@@ -1 +1 @@
-p
+q
"""

files = a.parse_diff(DIFF)
assert [f["path"] for f in files] == ["pkg/core.py", "tests/test_core.py", "reproduce_bug.py", "setup.cfg"]
assert (files[0]["add"], files[0]["del"], files[0]["hunks"]) == (3, 2, 2)
assert files[2]["new"] and not files[0]["new"]

assert a.is_test("tests/test_core.py") and a.is_test("a/test_x.py") and a.is_test("a/x_test.py")
assert a.is_test("django/test/client.py") and a.is_test("conftest.py")
assert not a.is_test("src/latest.py") and not a.is_test("pkg/contest.py") and not a.is_test("protests/a.py")

f = a.patch_features(files, {"tests/test_core.py"})
assert f["src_set"] == {"pkg/core.py", "reproduce_bug.py", "setup.cfg"}
assert (f["src_files"], f["hunks"]) == (3, 4)
assert (f["edits_tests"], f["adds_tests"], f["edits_hidden_tests"], f["stray_file"], f["config_touch"]) == (1, 0, 1, 1, 1)
assert a.patch_features(files, set())["edits_hidden_tests"] == 0
assert a.patch_features([], set())["src_files"] == 0

low, high = a.wilson(5, 10)
assert abs(low - 0.2366) < 1e-3 and abs(high - 0.7634) < 1e-3

# Issue i1 has one wrong patch against two right ones: scores 3 vs (1, 3), so 1 win and 1 tie.
# Issue i2 is all right and must contribute no pairs.
frame = pd.DataFrame({"instance_id": ["i1", "i1", "i1", "i2"],
                      "wrong_but_green": [True, False, False, False],
                      "s": [3.0, 1.0, 3.0, 9.0]})
assert a.within_auc(frame, "s") == 0.75

ranked = pd.DataFrame({"wrong_but_green": [True] + [False] * 4 + [True] * 1, "s": [9.0, 1, 2, 3, 4, 0]})
assert a.recall_at(ranked, "s", share=0.2) == 0.5

cons = pd.DataFrame({"instance_id": ["i", "i", "i"], "evaluable": [True, True, False],
                     "src_set": [frozenset({"a", "b"}), frozenset({"a"}), frozenset({"z"})]})
out = a.add_consensus(cons)
assert out.consensus[0] == 0.5 and out.consensus[1] == 0.5 and pd.isna(out.consensus[2])
print("all checks passed")
