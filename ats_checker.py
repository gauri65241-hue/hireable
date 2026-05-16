"""
ATS compliance pre-flight checker.
Ported from resume-ats-optimizer — 9 heuristic criteria, no LLM needed.
Runs before the 5 LLM judges as an instant structural check.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

SECTION_HEADINGS = [
    "experience", "work experience", "professional experience", "employment",
    "education", "skills", "technical skills", "summary", "objective",
    "profile", "projects", "certifications", "achievements", "publications",
    "awards", "languages", "interests", "references",
]

ACTION_VERBS = [
    "achieved", "built", "created", "designed", "developed", "drove",
    "engineered", "established", "executed", "generated", "implemented",
    "improved", "increased", "launched", "led", "managed", "optimised",
    "optimized", "owned", "reduced", "scaled", "shipped", "spearheaded",
    "streamlined", "delivered", "deployed", "automated", "architected",
    "coordinated", "mentored", "negotiated", "partnered", "produced",
    "resolved", "transformed",
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "it",
    "its", "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "they", "them", "their", "what", "which",
    "who", "how", "not", "no", "so", "if", "then", "than", "about", "also",
}


@dataclass
class ATSCriterion:
    id: str
    label: str
    weight: int
    passed: bool
    detail: str


@dataclass
class ATSResult:
    score: float
    grade: str
    criteria: list[ATSCriterion] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _check_contact_info(text: str) -> ATSCriterion:
    has_email = bool(re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text))
    has_phone = bool(re.search(r'[\+\(]?\d[\d\s\-\(\)]{7,}\d', text))
    passed = has_email and has_phone
    missing = []
    if not has_email:
        missing.append("no email found")
    if not has_phone:
        missing.append("no phone found")
    return ATSCriterion("contact_info", "Contact Info", 10, passed,
                        "Email and phone detected" if passed else "; ".join(missing))


def _check_section_headings(text: str) -> ATSCriterion:
    lower = text.lower()
    found = [h for h in SECTION_HEADINGS if re.search(rf'\b{re.escape(h)}\b', lower)]
    passed = len(found) >= 3
    return ATSCriterion("section_headings", "Section Headings", 15, passed,
                        f"{len(found)} standard headings found: {', '.join(found[:6])}")


def _check_bullet_points(text: str) -> ATSCriterion:
    bullet_lines = [l for l in text.splitlines() if re.match(r'^\s*[-•*▪◦→]\s+\S', l)]
    passed = len(bullet_lines) >= 5
    return ATSCriterion("bullet_points", "Bullet Points", 10, passed,
                        f"{len(bullet_lines)} bullet-formatted lines (need ≥5)")


def _check_action_verbs(text: str) -> ATSCriterion:
    lower = text.lower()
    found = [v for v in ACTION_VERBS if re.search(rf'\b{v}\b', lower)]
    passed = len(found) >= 4
    return ATSCriterion("action_verbs", "Action Verbs", 10, passed,
                        f"{len(found)} action verbs: {', '.join(found[:8])}")


def _check_quantification(text: str) -> ATSCriterion:
    hits = re.findall(
        r'\d+\s*(?:%|x\b|times\b|million\b|k\b|thousand\b|billion\b|hrs?\b|days?\b|weeks?\b|months?\b|years?\b)',
        text, re.IGNORECASE
    )
    passed = len(hits) >= 2
    return ATSCriterion("quantification", "Quantified Results", 15, passed,
                        f"{len(hits)} quantified data points (need ≥2)")


def _check_dates(text: str) -> ATSCriterion:
    date_ranges = re.findall(
        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+\d{4}|(?:19|20)\d{2}\s*[–\-—]\s*(?:(?:19|20)\d{2}|[Pp]resent|[Cc]urrent)',
        text
    )
    passed = len(date_ranges) >= 2
    return ATSCriterion("dates", "Employment Dates", 10, passed,
                        f"{len(date_ranges)} date ranges detected (need ≥2)")


def _check_length(text: str) -> ATSCriterion:
    words = len(text.split())
    passed = 200 <= words <= 1200
    return ATSCriterion("length", "Resume Length", 10, passed,
                        f"{words} words (optimal: 200–1200)")


def _check_no_tables(text: str) -> ATSCriterion:
    pipe_lines = sum(1 for l in text.splitlines() if l.count('|') >= 2)
    passed = pipe_lines < 3
    return ATSCriterion("no_tables", "No Table Formatting", 10, passed,
                        "Table formatting detected — ATS often fails to parse pipe-separated content"
                        if not passed else "No table formatting detected")


def _check_keywords(text: str, job_description: Optional[str]) -> ATSCriterion:
    if not job_description:
        return ATSCriterion("keywords", "JD Keyword Match", 10, True,
                            "No job description provided — skipped")
    jd_tokens = {
        w.lower().strip('.,;:!?')
        for w in re.split(r'\s+', job_description)
        if len(w) > 3 and w.lower() not in STOPWORDS
    }
    resume_lower = text.lower()
    matched = {t for t in jd_tokens if re.search(rf'\b{re.escape(t)}\b', resume_lower)}
    if not jd_tokens:
        return ATSCriterion("keywords", "JD Keyword Match", 10, True, "No meaningful JD keywords extracted")
    pct = len(matched) / len(jd_tokens)
    passed = pct >= 0.40
    return ATSCriterion("keywords", "JD Keyword Match", 10, passed,
                        f"{round(pct * 100)}% JD keywords matched ({len(matched)}/{len(jd_tokens)})")


def score_ats(resume_text: str, job_description: Optional[str] = None) -> ATSResult:
    checks = [
        _check_contact_info(resume_text),
        _check_section_headings(resume_text),
        _check_bullet_points(resume_text),
        _check_action_verbs(resume_text),
        _check_quantification(resume_text),
        _check_dates(resume_text),
        _check_length(resume_text),
        _check_no_tables(resume_text),
        _check_keywords(resume_text, job_description),
    ]
    total_weight = sum(c.weight for c in checks)
    earned = sum(c.weight for c in checks if c.passed)
    score = round(earned / total_weight * 100, 1)
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D" if score >= 40 else "F"
    issues = [f"{c.label}: {c.detail}" for c in checks if not c.passed]
    return ATSResult(score=score, grade=grade, criteria=checks, issues=issues)


def ats_to_dict(result: ATSResult) -> dict:
    return {
        "score": result.score,
        "grade": result.grade,
        "issues": result.issues,
        "criteria": [
            {"id": c.id, "label": c.label, "weight": c.weight, "passed": c.passed, "detail": c.detail}
            for c in result.criteria
        ],
    }
