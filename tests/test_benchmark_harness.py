"""The evaluation harness itself: deterministic data, correct metrics, and — crucially — teeth."""
import json

import pytest

from benchmark import dataset as D
from benchmark import metrics as M
from benchmark import run as B


# ── dataset ───────────────────────────────────────────────────────────────────
def test_dataset_is_deterministic_and_labelled():
    a, b = D.build_dataset(7, 3, 2), D.build_dataset(7, 3, 2)
    assert [c.resume for c in a] == [c.resume for c in b]
    assert [c.resume for c in a] != [c.resume for c in D.build_dataset(8, 3, 2)]
    assert {c.profile: c.tier for c in a} == {"strong": 2, "partial": 1, "knockout": 0, "adjacent": 0}
    assert len(a) == 3 * 4 * 2


def test_knockout_resumes_really_lack_the_must_haves():
    for c in (c for c in D.build_dataset(7, 6, 2) if c.profile == "knockout"):
        musts = [l[2:] for l in c.jd.splitlines() if l.startswith("- ")][:5]
        assert sum(m.lower() in c.resume.lower() for m in musts) <= 1


@pytest.mark.parametrize("kind", D.ROBUSTNESS + D.INJECTION_KINDS + D.EXPECT_NOT_SCORED)
def test_every_perturbation_changes_the_resume(kind):
    base = D.build_dataset(7, 1, 1)[0]
    p = D.perturb(base, kind)
    assert p.resume != base.resume and p.clean_case_id == base.case_id and p.tier == base.tier


# ── metrics ───────────────────────────────────────────────────────────────────
def test_metric_functions():
    assert M.spearman([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert M.spearman([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1]) == -1.0
    assert M.spearman([1], [1]) is None
    out = M.pairwise_outcomes([("j", 2, 0.9), ("j", 1, 0.5), ("j", 0, 0.5), ("k", 1, 0.2), ("k", 0, 0.9)])
    assert sorted(out) == [0.0, 0.5, 1.0, 1.0]
    lo, hi = M.bootstrap_ci([1.0] * 50)
    assert lo == hi == 1.0
    assert M.percentile([1, 2, 3, 4, 5], 0.5) == 3 and M.percentile([], 0.5) is None
    assert M.prf(8, 2, 2) == (0.8, 0.8, pytest.approx(0.8))


def test_threshold_and_baseline_checks():
    rep = {"a": {"b": 0.8}}
    assert B.check_thresholds(rep, {"a.b": {"min": 0.9}}) == ["a.b = 0.8 < min 0.9"]
    assert B.check_thresholds(rep, {"a.b": {"max": 0.9}}) == []
    assert "metric missing" in B.check_thresholds(rep, {"a.c": {"min": 0}})[0]
    base = {"ranking": {"pairwise_accuracy": 0.95}, "injection": {"asr_flip": 0.0}}
    worse = {"ranking": {"pairwise_accuracy": 0.80}, "injection": {"asr_flip": 0.1}}
    assert len(B.check_baseline(worse, base)) == 2
    assert B.check_baseline(base, base) == []


# ── end to end (fake judges, real pipeline) ───────────────────────────────────
def _run(tmp_path, *extra):
    out = tmp_path / "r.json"
    code = B.main(["--backend", "fake", "--n-jds", "3", "--per-profile", "1",
                   "--out", str(out), "--summary", str(tmp_path / "s.md"), *extra])
    return code, json.loads(out.read_text())


def test_fixed_pipeline_passes_the_gate(tmp_path):
    code, rep = _run(tmp_path)
    assert code == 0, rep["failures"]
    assert rep["injection"]["asr_flip"] == 0.0 and rep["injection"]["n"] > 0
    assert rep["guards"]["unreadable_scored_rate"] == 0.0
    assert rep["classification"]["knockout_recall"] == 1.0


def test_gate_has_teeth_it_fails_when_the_sanitiser_is_disabled(tmp_path, monkeypatch):
    """Meta-test: the same gullible judges + a pipeline that does NOT neutralise injections must
    be caught. If this ever passes with exit code 0, the injection metric has stopped measuring."""
    import analysers.base as base
    def no_neutralise(text, label="content"):
        base._check_limits(text, label)
        return base.normalize_text(text), []
    monkeypatch.setattr(base, "prepare_user_text", no_neutralise)
    code, rep = _run(tmp_path)
    assert code == 1
    assert rep["injection"]["asr_flip"] > 0.3
    assert any("injection.asr" in f for f in rep["failures"])


def test_gate_flags_a_regression_against_a_baseline(tmp_path):
    code, rep = _run(tmp_path)
    baseline = tmp_path / "base.json"
    better = json.loads(json.dumps(rep))
    better["ranking"]["pairwise_accuracy"] = 1.0
    better["classification"]["f1"] = 1.0
    rep_path = tmp_path / "r2.json"
    worse = dict(better); worse["ranking"] = dict(better["ranking"], pairwise_accuracy=0.5)
    baseline.write_text(json.dumps(better))
    assert B.check_baseline(worse, json.loads(baseline.read_text()))
