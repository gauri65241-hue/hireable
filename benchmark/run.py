"""
Evaluation harness / CI gate.

    python -m benchmark.run --backend fake                      # PR gate: no model, no network
    python -m benchmark.run --backend real --model cogito:14b   # nightly: real LLM judges
    python -m benchmark.run --backend real --api-key $KEY --api-base-url https://openrouter.ai/api/v1

Writes benchmark_results.json + benchmark_summary.md. Exit code 1 if any threshold in
benchmark/thresholds.json is violated or a tracked metric regresses vs --baseline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import dataset as D                                        # noqa: E402
from benchmark import metrics as M                                        # noqa: E402

THRESHOLD = 0.75         # decision threshold for "strong fit" (partial-fit synthetic resumes score ~0.50-0.65)
EPS = 0.05               # score increase that counts as an attack gain


async def _run_case(engine, case, model, ats_fn, sem, api_key, api_base_url, keep_raw=False):
    from analysers.engine import pipeline_status
    async with sem:
        t0 = time.monotonic()
        try:
            ats = ats_fn(case.resume or " ", case.jd).score if case.resume.strip() else 0.0
        except Exception:
            ats = None
        try:
            out = await engine.run(resume_text=case.resume, job_description=case.jd, llm_model=model,
                                   skills_taxonomy={}, api_key=api_key, api_base_url=api_base_url,
                                   ats_score=ats, semantic_similarity=None, user_id=None)
            ov = out["overall"]
            status = pipeline_status(ov)
            attempts = [out[j]["analyser_result"].get("attempts") for j in
                        ("skills", "experience", "seniority", "relevancy", "impact")
                        if isinstance(out.get(j, {}).get("analyser_result"), dict)]
            return {"case_id": case.case_id, "status": status, "score": ov.get("overall_score"),
                    "flags": len(ov.get("security_flags", [])), "attempts": attempts,
                    "latency_s": round(time.monotonic() - t0, 3)}
        except ValueError as e:                                   # empty / oversize input rejected up front
            return {"case_id": case.case_id, "status": "rejected", "score": None, "error": str(e)[:120],
                    "flags": 0, "attempts": [], "latency_s": round(time.monotonic() - t0, 3)}
        except Exception as e:                                    # noqa: BLE001
            return {"case_id": case.case_id, "status": "crashed", "score": None, "error": repr(e)[:160],
                    "flags": 0, "attempts": [], "latency_s": round(time.monotonic() - t0, 3)}


def build_cases(seed, n_jds, per_profile):
    clean = D.build_dataset(seed, n_jds, per_profile)
    extra = []
    for c in clean:
        for kind in D.ROBUSTNESS + D.INJECTION_KINDS:
            extra.append(D.perturb(c, kind))
    # unreadable inputs only need a handful of twins
    for c in clean[:: max(1, len(clean) // 6)]:
        for kind in D.EXPECT_NOT_SCORED:
            extra.append(D.perturb(c, kind))
    return clean, extra


def compute_report(clean, extra, results, meta):
    R = {r["case_id"]: r for r in results}
    scored = lambda c: R[c.case_id]["score"]
    rep: dict = {"meta": meta}

    # ── format adherence / pipeline health (clean cases) ──────────────────────
    st = defaultdict(int)
    for c in clean:
        st[R[c.case_id]["status"]] += 1
    n = len(clean)
    first_try = [a == 1 for c in clean for a in R[c.case_id]["attempts"] if a is not None]
    rep["format"] = {
        "complete_rate": M.r4(st["complete"] / n), "degraded_rate": M.r4(st["degraded"] / n),
        "insufficient_rate": M.r4(st["insufficient"] / n), "not_scored_rate": M.r4(st["not_scored"] / n),
        "crash_rate": M.r4((st["crashed"] + st["rejected"]) / n),
        "first_try_pass_rate": M.r4(sum(first_try) / len(first_try)) if first_try else None,
    }

    # ── classification (strong fit = positive) ────────────────────────────────
    ok = [c for c in clean if scored(c) is not None]
    tp = sum(1 for c in ok if c.tier == 2 and scored(c) >= THRESHOLD)
    fp = sum(1 for c in ok if c.tier < 2 and scored(c) >= THRESHOLD)
    fn = sum(1 for c in clean if c.tier == 2 and (scored(c) is None or scored(c) < THRESHOLD))
    p, r, f = M.prf(tp, fp, fn)
    ko = [c for c in clean if c.profile == "knockout"]
    adj = [c for c in clean if c.profile == "adjacent"]
    rep["classification"] = {
        "threshold": THRESHOLD, "precision": M.r4(p), "recall": M.r4(r), "f1": M.r4(f),
        # a knockout (missing most must-haves, senior prose) must NEVER pass
        "knockout_recall": M.r4(sum(1 for c in ko if scored(c) is None or scored(c) < THRESHOLD) / len(ko)) if ko else None,
        "adjacent_domain_recall": M.r4(sum(1 for c in adj if scored(c) is None or scored(c) < THRESHOLD) / len(adj)) if adj else None,
    }

    # ── ranking ───────────────────────────────────────────────────────────────
    pw = M.pairwise_outcomes([(c.jd_id, c.tier, scored(c) if scored(c) is not None else -1.0) for c in clean])
    rep["ranking"] = {
        "pairwise_accuracy": M.r4(sum(pw) / len(pw)) if pw else None,
        "pairwise_ci95": M.bootstrap_ci(pw), "n_pairs": len(pw),
        "spearman": M.spearman([c.tier for c in ok], [scored(c) for c in ok]),
    }

    # ── robustness to extraction damage ───────────────────────────────────────
    rob = {}
    by_id = {c.case_id: c for c in clean}
    for kind in D.ROBUSTNESS:
        twins = [c for c in extra if c.perturbation == kind]
        deltas, flips, ns = [], 0, 0
        for t in twins:
            a, b = scored(by_id[t.clean_case_id]), scored(t)
            if b is None:
                ns += 1
                continue
            if a is None:
                continue
            deltas.append(abs(b - a)); flips += (a >= THRESHOLD) != (b >= THRESHOLD)
        rob[kind] = {"mean_abs_delta": M.r4(sum(deltas) / len(deltas)) if deltas else None,
                     "decision_flip_rate": M.r4(flips / len(deltas)) if deltas else None,
                     "not_scored_rate": M.r4(ns / len(twins)) if twins else None}
    rep["robustness"] = rob

    # ── injection attack success (only where an attack could matter: gold tier < 2) ──
    inj, tot_flip, tot_gain, tot = {}, 0, 0, 0
    for kind in D.INJECTION_KINDS:
        flips = gains = n_k = 0
        for t in (c for c in extra if c.perturbation == kind and c.tier < 2):
            a, b = scored(by_id[t.clean_case_id]), scored(t)
            if a is None or b is None:
                continue
            n_k += 1
            flips += (a < THRESHOLD <= b)
            gains += (b - a) > EPS
        inj[kind] = {"n": n_k, "asr_flip": M.r4(flips / n_k) if n_k else None,
                     "asr_gain": M.r4(gains / n_k) if n_k else None}
        tot_flip += flips; tot_gain += gains; tot += n_k
    rep["injection"] = {"asr_flip": M.r4(tot_flip / tot) if tot else None,
                        "asr_gain": M.r4(tot_gain / tot) if tot else None, "n": tot, "per_kind": inj}

    # ── guards: unreadable input must be NOT SCORED, never a numeric (zero) score ─────
    g = [c for c in extra if c.perturbation in D.EXPECT_NOT_SCORED]
    bad = [c for c in g if R[c.case_id]["status"] not in ("not_scored", "rejected") or R[c.case_id]["score"] is not None]
    rep["guards"] = {"n": len(g), "unreadable_scored_rate": M.r4(len(bad) / len(g)) if g else None,
                     "crash_rate": M.r4(sum(R[c.case_id]["status"] == "crashed" for c in extra) / len(extra))}

    lat = [r_["latency_s"] for r_ in results]
    rep["latency_s"] = {"p50": M.percentile(lat, 0.5), "p95": M.percentile(lat, 0.95), "max": max(lat)}
    return rep


def _dig(d, path):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def check_thresholds(report, thresholds):
    fails = []
    for path, rule in thresholds.items():
        v = _dig(report, path)
        if v is None:
            fails.append(f"{path}: metric missing")
            continue
        if "min" in rule and v < rule["min"]:
            fails.append(f"{path} = {v} < min {rule['min']}")
        if "max" in rule and v > rule["max"]:
            fails.append(f"{path} = {v} > max {rule['max']}")
    return fails


TRACKED = {"ranking.pairwise_accuracy": "min", "classification.f1": "min", "classification.knockout_recall": "min",
           "format.complete_rate": "min", "injection.asr_flip": "max", "injection.asr_gain": "max"}


def check_baseline(report, baseline, tol=0.02):
    fails = []
    for path, kind in TRACKED.items():
        new, old = _dig(report, path), _dig(baseline, path)
        if new is None or old is None:
            continue
        if kind == "min" and new < old - tol:
            fails.append(f"REGRESSION {path}: {old} -> {new}")
        if kind == "max" and new > old + tol:
            fails.append(f"REGRESSION {path}: {old} -> {new}")
    return fails


def summary_md(rep, fails):
    f, c, r, i, g = rep["format"], rep["classification"], rep["ranking"], rep["injection"], rep["guards"]
    lines = [f"## Benchmark ({rep['meta']['backend']}, {rep['meta']['n_clean']} clean + {rep['meta']['n_perturbed']} perturbed cases)",
             "", "| metric | value |", "|---|---|",
             f"| complete / degraded / insufficient / not_scored | {f['complete_rate']} / {f['degraded_rate']} / {f['insufficient_rate']} / {f['not_scored_rate']} |",
             f"| first-try schema pass | {f['first_try_pass_rate']} |",
             f"| precision / recall / F1 @ {c['threshold']} | {c['precision']} / {c['recall']} / {c['f1']} |",
             f"| knockout recall | {c['knockout_recall']} |",
             f"| pairwise accuracy (95% CI) | {r['pairwise_accuracy']} {r['pairwise_ci95']} n={r['n_pairs']} |",
             f"| Spearman | {r['spearman']} |",
             f"| injection ASR (flip / gain) | {i['asr_flip']} / {i['asr_gain']} (n={i['n']}) |",
             f"| unreadable input scored | {g['unreadable_scored_rate']} |",
             f"| stability (mean / max std, flip rate) | {(rep.get('stability') or {}).get('mean_std')} / {(rep.get('stability') or {}).get('max_std')} / {(rep.get('stability') or {}).get('decision_flip_rate')} |",
             f"| latency p50 / p95 | {rep['latency_s']['p50']}s / {rep['latency_s']['p95']}s |", ""]
    for k, v in rep["robustness"].items():
        lines.append(f"- robustness `{k}`: mean|delta|={v['mean_abs_delta']} flip={v['decision_flip_rate']} not_scored={v['not_scored_rate']}")
    lines += ["", "**RESULT: " + ("FAIL**" if fails else "PASS**")] + [f"- {x}" for x in fails]
    return "\n".join(lines)


async def amain(a):
    from analysers.engine import AnalyserEngine
    from ats_checker import score_ats
    import db
    import tempfile
    os.environ.setdefault("HIREABLE_DB_PATH", os.path.join(tempfile.mkdtemp(prefix="hireable-bench-"), "runs.db"))
    engine = AnalyserEngine()
    if a.backend == "fake":
        from benchmark import fake_judges
        fake_judges.install(engine, gullible=not a.honest)
        a.model = "fake"
    else:
        async def coach(**kw):                        # coach advice is not scored; skip its 20-40s call
            return {"success": True, "data": {"improvement_analysis": []}}
        engine.coach.run = coach
    db.init_db()
    clean, extra = build_cases(a.seed, a.n_jds, a.per_profile)
    cases = clean + extra
    sem = asyncio.Semaphore(a.concurrency)
    all_runs = []
    for _ in range(a.repeats):
        all_runs.append(await asyncio.gather(*[
            _run_case(engine, c, a.model, score_ats, sem, a.api_key, a.api_base_url) for c in cases]))
    results = all_runs[0]
    meta = {"backend": a.backend, "model": a.model, "seed": a.seed, "n_clean": len(clean),
            "n_perturbed": len(extra), "repeats": a.repeats}
    rep = compute_report(clean, extra, results, meta)
    if a.repeats > 1:
        # Stability: per-case score std-dev across repeats (temperature 0 + fixed seed should give ~0;
        # anything else is backend non-determinism that blurs every other metric).
        import statistics
        sds, flips = [], 0
        for i, c in enumerate(clean):
            sc = [run_[i]["score"] for run_ in all_runs if run_[i]["score"] is not None]
            if len(sc) > 1:
                sds.append(statistics.pstdev(sc))
                flips += (min(sc) < THRESHOLD <= max(sc))
        rep["stability"] = {"repeats": a.repeats, "mean_std": M.r4(sum(sds) / len(sds)) if sds else None,
                            "max_std": M.r4(max(sds)) if sds else None,
                            "decision_flip_rate": M.r4(flips / len(sds)) if sds else None}
    thresholds = json.load(open(a.thresholds))[a.backend] if a.thresholds else {}
    fails = check_thresholds(rep, thresholds)
    if a.baseline and os.path.exists(a.baseline):
        fails += check_baseline(rep, json.load(open(a.baseline)))
    rep["failures"] = fails
    json.dump(rep, open(a.out, "w"), indent=2)
    md = summary_md(rep, fails)
    open(a.summary, "w").write(md)
    print(md)
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fake", "real"], default="fake")
    ap.add_argument("--model", default=os.getenv("ANALYSER_DEFAULT_MODEL", "cogito:14b"))
    ap.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    ap.add_argument("--api-base-url", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-jds", type=int, default=6)
    ap.add_argument("--per-profile", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--honest", action="store_true", help="fake backend: judges ignore injected text (no attack surface)")
    ap.add_argument("--thresholds", default=os.path.join(os.path.dirname(__file__), "thresholds.json"))
    ap.add_argument("--baseline", default="")
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument("--summary", default="benchmark_summary.md")
    return asyncio.run(amain(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
