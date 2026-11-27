"""
Deterministic, rule-based stand-ins for the five LLM judges.

Purpose: exercise the REAL pipeline (sanitisation, ATS gate, scoring math, gates, quorum,
weights, caching, status handling) in CI with no model and no network. They do NOT measure LLM
quality — that is what `--backend real` (nightly) is for.

`gullible=True` simulates the worst-case LLM: if instruction-like text reaches it, it obeys and
marks everything perfect. Whatever the sanitiser lets through is therefore visible as attack
success, which is exactly the property the injection metrics gate on.
"""
from __future__ import annotations

import re

GULLIBLE_RE = re.compile(
    r"(mark all|10/10|ideal|perfect|maximum|ignore all previous|evaluator note|rate 100|assign max)", re.I)


def _section(jd: str, header: str) -> list[str]:
    out, on = [], False
    for line in jd.splitlines():
        if line.strip().lower().startswith(header.lower()):
            on = True
            continue
        if on:
            if line.startswith("- "):
                out.append(line[2:].strip())
            elif line.strip() == "" and out:
                break
    return out


def _years(text: str, default: int = 0) -> int:
    m = re.search(r"(\d+)\s+years", text)
    return int(m.group(1)) if m else default


def _has(text: str, skill: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text, re.I) is not None


def _bullets(resume: str) -> list[str]:
    return [l[2:].strip() for l in resume.splitlines() if l.startswith("- ") and len(l) > 12][:12] or ["n/a"]


def install(engine, gullible: bool = True) -> None:
    async def skills(resume_text, jd, *a, **k):
        musts = _section(jd, "requirements")
        pwn = gullible and GULLIBLE_RE.search(resume_text)
        items = [{"skill": s, "is_present": 1 if (pwn or _has(resume_text, s)) else 0,
                  "evidence_strength": "strong", "resume_evidence": s if (pwn or _has(resume_text, s)) else ""}
                 for s in musts]
        if not items:
            return {"success": False, "error": "no requirements parsed", "data": {}}
        return {"success": True, "data": {"required_skills": musts, "present_skills": [], "skill_analysis": items,
                                          "confidence": 90}}

    async def experience(resume_text, jd, *a, **k):
        musts = _section(jd, "requirements") or ["x"]
        cover = sum(_has(resume_text, s) for s in musts) / len(musts)
        req = max(1, _years(jd, 5))
        base = min(1.0, _years(resume_text) / req) * (0.4 + 0.6 * cover)
        base = 1.0 if (gullible and GULLIBLE_RE.search(resume_text)) else base
        resp = _section(jd, "responsibilities") or ["x"]
        return {"success": True, "data": {"responsibility_analysis": [
            {"responsibility": r, "experience_score": base} for r in resp], "confidence": 90}}

    async def seniority(resume_text, jd, *a, **k):
        low = resume_text.lower()
        pwn = gullible and GULLIBLE_RE.search(resume_text)
        sig = ["led", "mentor", "team", "own", "architect"]
        return {"success": True, "data": {"seniority_analysis": [
            {"signal": s, "is_demonstrated": 1 if (pwn or s in low) else 0} for s in sig], "confidence": 90}}

    async def relevancy(resume_text, jd, *a, **k):
        musts = _section(jd, "requirements")
        pwn = gullible and GULLIBLE_RE.search(resume_text)
        return {"success": True, "data": {"statement_analysis": [
            {"resume_statement": b, "is_relevant": 1 if (pwn or any(_has(b, s) for s in musts)) else 0}
            for b in _bullets(resume_text)], "confidence": 90}}

    async def impact(resume_text, jd, *a, **k):
        pwn = gullible and GULLIBLE_RE.search(resume_text)
        return {"success": True, "data": {"impact_analysis": [
            {"resume_statement": b, "is_quantified": 1 if (pwn or re.search(r"\d", b)) else 0}
            for b in _bullets(resume_text)], "confidence": 90}}

    async def coach(**kw):
        return {"success": True, "data": {"improvement_analysis": []}}

    engine.skills.run, engine.experience.run, engine.seniority.run = skills, experience, seniority
    engine.relevancy.run, engine.impact.run, engine.coach.run = relevancy, impact, coach
