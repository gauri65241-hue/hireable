"""
ATS compliance pre-flight checker.
9 heuristic criteria, no LLM needed.
Runs before the 5 LLM judges as an instant structural check.

PDF-aware: handles plain-text extraction where bullet characters and
section-heading line breaks are stripped by pypdf/pdfminer.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

SECTION_HEADINGS = [
    "experience", "work experience", "professional experience", "employment",
    "education", "skills", "technical skills", "summary", "objective",
    "profile", "about me", "about", "projects", "certifications",
    "achievements", "publications", "awards", "languages", "interests",
    "references",
]

ACTION_VERBS = [
    "achieved", "architected", "automated", "built", "created", "designed",
    "developed", "drove", "engineered", "established", "executed",
    "generated", "implemented", "improved", "increased", "launched", "led",
    "managed", "optimised", "optimized", "owned", "reduced", "scaled",
    "shipped", "spearheaded", "streamlined", "delivered", "deployed",
    "coordinated", "mentored", "negotiated", "partnered", "produced",
    "resolved", "transformed", "researched", "analyzed", "analysed",
    "integrated", "migrated", "refactored", "maintained",
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
    """
    PDF-aware detection: checks three signals.
    1. Heading word appears anywhere in text (broad match — handles merged lines)
    2. Short ALL-CAPS lines (PDF heading style)
    3. Title-case short standalone lines
    Need ≥3 distinct heading matches.
    """
    lower = text.lower()

    # Signal 1: known heading word anywhere (no strict trailing \b — handles run-together text)
    word_found = [h for h in SECTION_HEADINGS
                  if re.search(rf'\b{re.escape(h)}', lower)]

    # Signal 2: short ALL-CAPS lines (e.g. "EXPERIENCE", "EDUCATION", "SKILLS")
    caps_lines = set()
    for line in text.splitlines():
        stripped = line.strip()
        if 2 <= len(stripped.split()) <= 5 and stripped.isupper() and len(stripped) >= 4:
            caps_lines.add(stripped.lower())

    # Merge unique hits
    all_found = list(dict.fromkeys(word_found + [c for c in caps_lines
                                                  if any(h in c for h in SECTION_HEADINGS)]))
    passed = len(all_found) >= 3
    return ATSCriterion("section_headings", "Section Headings", 15, passed,
                        f"{len(all_found)} standard headings found: {', '.join(all_found[:6])}"
                        if all_found else "0 standard headings found")


def _check_bullet_points(text: str) -> ATSCriterion:
    """
    PDF-aware bullet detection — four signals, best count wins.

    Signal 1 — Explicit chars: lines starting with •, -, *, ▪ etc.
    Signal 2 — Line-proxy: medium-length lines starting with action verb
               (when PDF preserves newlines between bullets).
    Signal 3 — Sentence-proxy: split on sentence boundaries and count
               action-verb-starting segments (when PDF merges bullets into
               paragraphs but periods are preserved).
    Signal 4 — Action-verb count: count of distinct action verb occurrences
               in the full text. Final fallback for PDFs that merge bullets
               AND strip periods between them (e.g. "...Inc. Architected...
               Engineered... Developed"). If a resume has 5+ action verb
               hits it almost certainly has 5+ bullets.
    """
    action_re = re.compile(r'\b(' + '|'.join(ACTION_VERBS) + r')\b', re.IGNORECASE)
    action_start_re = re.compile(r'^(' + '|'.join(ACTION_VERBS) + r')\b', re.IGNORECASE)
    lines = text.splitlines()

    # Signal 1: explicit bullet chars
    explicit = [line for line in lines if re.match(r'^\s*[-•*▪◦→✓]\s+\S', line)]

    # Signal 2: line-level proxy
    line_proxy = []
    for line in lines:
        stripped = line.strip()
        words = stripped.split()
        if len(words) < 6 or len(words) > 60:
            continue
        if stripped and action_start_re.match(stripped):
            if not stripped.endswith(':'):
                line_proxy.append(stripped)

    # Signal 3: sentence-level proxy (split on '. ' or '; ' before capital)
    sentence_splits = re.split(r'(?<=[.;])\s+(?=[A-Z])', text)
    sentence_proxy = [s.strip() for s in sentence_splits
                      if action_start_re.match(s.strip()) and 5 < len(s.split()) <= 80]

    # Signal 4: raw action-verb occurrence count (most PDF-robust)
    verb_hits = action_re.findall(text.lower())
    verb_count = len(verb_hits)

    best_count = max(len(explicit), len(line_proxy), len(sentence_proxy), verb_count)
    passed = best_count >= 5

    if explicit:
        detail = f"{len(explicit)} explicit bullets detected"
    elif line_proxy:
        detail = f"{len(line_proxy)} bullet-style lines detected (PDF stripped bullet chars)"
    elif sentence_proxy:
        detail = f"{len(sentence_proxy)} action-verb sentences detected (PDF merged bullets into paragraphs)"
    elif verb_count >= 5:
        detail = f"{verb_count} action-verb instances detected across bullet content (PDF merged lines)"
    else:
        detail = "0 bullet-formatted lines (need ≥5)"

    return ATSCriterion("bullet_points", "Bullet Points", 10, passed, detail)


def _check_action_verbs(text: str) -> ATSCriterion:
    lower = text.lower()
    found = [v for v in ACTION_VERBS if re.search(rf'\b{v}\b', lower)]
    passed = len(found) >= 4
    return ATSCriterion("action_verbs", "Action Verbs", 10, passed,
                        f"{len(found)} action verbs: {', '.join(found[:8])}")


def _check_quantification(text: str) -> ATSCriterion:
    """
    PDF-aware: matches standard metric patterns PLUS:
    - Numbers with + suffix: "12+", "5,000+", "7+"
    - Numbers followed by domain nouns: metrics, datasets, tests, users, etc.
    - Percentages and multipliers
    """
    # Standard: number + unit/percentage
    standard = re.findall(
        r'\d{1,3}(?:,\d{3})*[+]?\s*(?:%|x\b|×|times\b|million\b|k\b|thousand\b|'
        r'billion\b|hrs?\b|days?\b|weeks?\b|months?\b|years?\b)',
        text, re.IGNORECASE
    )
    # Domain nouns common in tech resumes
    domain_noun = re.findall(
        r'\d{1,3}(?:,\d{3})*[+]?\s+'
        r'(?:custom\s+)?(?:rag\s+)?'
        r'(?:metrics?|datasets?|tests?|models?|products?|features?|pipelines?|'
        r'agents?|systems?|questions?|q&a|endpoints?|users?|clients?|customers?|'
        r'employees?|teams?|projects?|services?|apis?|repositories?|repos?|'
        r'languages?|libraries?|frameworks?|tools?|rules?|labels?|pairs?|'
        r'functions?|queries?|reports?|dashboards?|experiments?|runs?)',
        text, re.IGNORECASE
    )
    # Bare number+plus (e.g. "12+" or "5,000+" as standalone quantifier)
    bare_plus = re.findall(r'\d{1,3}(?:,\d{3})*\+', text)

    all_hits = standard + domain_noun + bare_plus
    # Deduplicate overlapping matches
    unique = list(dict.fromkeys(all_hits))
    passed = len(unique) >= 2
    return ATSCriterion("quantification", "Quantified Results", 15, passed,
                        f"{len(unique)} quantified data points (need ≥2): {', '.join(unique[:5])}"
                        if unique else "0 quantified data points (need ≥2)")


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
    pipe_lines = sum(1 for ln in text.splitlines() if ln.count('|') >= 2)
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
