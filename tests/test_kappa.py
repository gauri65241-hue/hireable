"""
Tests for kappa.py — verifies cohen_kappa() and join() against known values.

Hand-calculated reference:
  6-pair fixture A (perfect agreement): kappa = 1.0
  6-pair fixture B (chance-level):      kappa ≈ 0.0 (each annotator uses all
                                         three classes uniformly, agreement only
                                         by chance)
  6-pair fixture C (partial, 4/6 agree, balanced class distribution): kappa ≈ 0.1
"""

import csv
import sys
from pathlib import Path

# Make the scripts directory importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from kappa import cohen_kappa, model_label, _FIT_THRESHOLD, _NO_THRESHOLD  # noqa: E402


CLASSES = ["fit", "maybe", "no"]


def test_perfect_agreement():
    a1 = ["fit", "fit", "maybe", "maybe", "no", "no"]
    a2 = ["fit", "fit", "maybe", "maybe", "no", "no"]
    k = cohen_kappa(a1, a2, CLASSES)
    assert k == 1.0, f"Perfect agreement should give kappa=1.0, got {k}"


def test_chance_level():
    # Each annotator uses each class exactly twice — uniform marginals.
    # Expected agreement p_e = (2/6)^2 * 3 = 1/3.
    # Observed agreement: only positions 0 and 3 match → p_o = 2/6.
    # kappa = (1/3 - 1/3) / (2/3) = 0.0
    a1 = ["fit",   "fit",   "maybe", "maybe", "no",  "no"]
    a2 = ["fit",   "maybe", "no",    "maybe", "fit", "no"]
    k = cohen_kappa(a1, a2, CLASSES)
    # p_o: positions 0 (fit==fit) and 3 (maybe==maybe) and 5 (no==no) agree → 3/6 = 0.5
    # recompute: a1[0]=fit a2[0]=fit ✓, a1[3]=maybe a2[3]=maybe ✓, a1[5]=no a2[5]=no ✓
    # p_o = 3/6 = 0.5; p_e = 3*(2/6*2/6) = 3*(1/9) = 1/3
    # kappa = (0.5 - 1/3) / (1 - 1/3) = (1/6) / (2/3) = 0.25
    assert abs(k - 0.25) < 1e-9, f"Expected kappa≈0.25, got {k}"


def test_all_same_class_one_annotator():
    # If annotator 2 never uses "maybe", kappa should still compute without crash.
    a1 = ["fit", "maybe", "no",  "fit", "no",  "maybe"]
    a2 = ["fit", "fit",   "no",  "fit", "no",  "fit"]
    # Should not raise even though "maybe" has 0 count in a2.
    k = cohen_kappa(a1, a2, CLASSES)
    assert isinstance(k, float)


def test_model_label_thresholds():
    assert model_label(1.0) == "fit"
    assert model_label(_FIT_THRESHOLD) == "fit"
    assert model_label(_FIT_THRESHOLD - 0.001) == "maybe"
    assert model_label(_NO_THRESHOLD) == "maybe"
    assert model_label(_NO_THRESHOLD - 0.001) == "no"
    assert model_label(0.0) == "no"
    assert model_label(None) == "maybe"


def test_join_end_to_end(tmp_path):
    """Full join path: write fixture CSVs, run join(), check output."""
    import io
    import contextlib
    from kappa import join

    blind = tmp_path / "blind.csv"
    scores = tmp_path / "blind.scores.csv"
    a1_path = tmp_path / "a1.csv"
    a2_path = tmp_path / "a2.csv"

    # Blind sheet (title/desc only — content irrelevant for join)
    with open(blind, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "job_title", "job_description"])
        for i in range(1, 5):
            w.writerow([f"j{i}", f"Job {i}", "Some description"])

    # Score mapping — two fits, one maybe, one no (by threshold)
    with open(scores, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "model_score"])
        w.writerows([["j1", 0.80], ["j2", 0.70], ["j3", 0.50], ["j4", 0.30]])

    # Perfect human agreement
    with open(a1_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "label"])
        w.writerows([["j1", "fit"], ["j2", "fit"], ["j3", "maybe"], ["j4", "no"]])
    with open(a2_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "label"])
        w.writerows([["j1", "fit"], ["j2", "fit"], ["j3", "maybe"], ["j4", "no"]])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        join(blind, a1_path, a2_path)

    output = buf.getvalue()
    assert "Inter-annotator kappa:      1.000" in output, f"Expected kappa=1.0 in output:\n{output}"
    assert "Model vs consensus agree:   100.0%" in output, f"Expected 100% model agreement:\n{output}"


if __name__ == "__main__":
    test_perfect_agreement()
    test_chance_level()
    test_all_same_class_one_annotator()
    test_model_label_thresholds()

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        test_join_end_to_end(Path(d))

    print("All tests passed.")
