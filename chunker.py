"""
Structured pre-processing for JDs and resumes.

Extracts only the semantically relevant sections for each judge type,
so models receive compact, signal-dense inputs instead of raw full text.

Judge data requirements:
  skills     → JD: required_skills, preferred_skills | Resume: skills, tech_in_experience
  experience → JD: responsibilities, experience_req   | Resume: work_history
  seniority  → JD: title, seniority_signals           | Resume: titles, progression
  relevancy  → JD: domain, core_requirements          | Resume: summary, experience_bullets
  impact     → Resume: achievement_bullets only (no JD)
  coach      → compact structured summary of all above
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class JDChunks:
    title: str = ""
    company: str = ""
    seniority_level: str = ""          # Junior / Mid / Senior / Lead / Staff
    years_required: str = ""           # "3-5 years"
    domain: str = ""                   # ML, Backend, Data, etc.
    required_skills: list = field(default_factory=list)
    preferred_skills: list = field(default_factory=list)
    responsibilities: list = field(default_factory=list)
    experience_requirements: list = field(default_factory=list)
    seniority_signals: list = field(default_factory=list)
    raw_truncated: str = ""            # fallback: first 3000 chars


@dataclass
class ResumeChunks:
    name: str = ""
    summary: str = ""
    skills: list = field(default_factory=list)
    work_history: list = field(default_factory=list)   # list of RoleBlock
    tech_in_experience: list = field(default_factory=list)
    achievement_bullets: list = field(default_factory=list)
    education: list = field(default_factory=list)
    certifications: list = field(default_factory=list)
    total_years: Optional[float] = None
    raw_truncated: str = ""


@dataclass
class RoleBlock:
    title: str = ""
    company: str = ""
    duration: str = ""
    bullets: list = field(default_factory=list)


# ── Section-header patterns ────────────────────────────────────────────────────

_JD_SECTION_PATTERNS = {
    "required": re.compile(
        r"(required\s+skills?|required|must.have|minimum qualifications?|basic qualifications?|"
        r"what you.ll need|you must|you should have|requirements?)\s*[:\-]?\s*[\n ]",
        re.I,
    ),
    "preferred": re.compile(
        r"(preferred|nice.to.have|bonus|plus|desired|advantage|"
        r"additional qualifications?|what.s a plus)\s*[:\-]?\s*[\n ]",
        re.I,
    ),
    "responsibilities": re.compile(
        r"(responsibilities|what you.ll do|your role|duties|"
        r"key responsibilities|job duties|about the role|you will)\s*[:\-]?\s*[\n ]",
        re.I,
    ),
    "experience": re.compile(
        r"(experience required|background|qualifications?|about you)\s*[:\-]?\s*[\n ]",
        re.I,
    ),
}

_RESUME_SECTION_PATTERNS = {
    "summary": re.compile(
        r"(summary|profile|objective|about me|professional summary|overview)\s*[:\-]?\s*\n",
        re.I,
    ),
    "skills": re.compile(
        r"(skills|technical skills|core competencies|technologies|tech stack|"
        r"expertise|tools|languages|frameworks)\s*[:\-]?\s*\n",
        re.I,
    ),
    "experience": re.compile(
        r"(experience|work experience|employment|career history|"
        r"professional experience|work history)\s*[:\-]?\s*\n",
        re.I,
    ),
    "education": re.compile(
        r"(education|academic|qualification|degree|university|college)\s*[:\-]?\s*\n",
        re.I,
    ),
    "certifications": re.compile(
        r"(certification|licence|award|achievement|course)\s*[:\-]?\s*\n",
        re.I,
    ),
}

_YEARS_PATTERN = re.compile(r"(\d+)\+?\s*(?:to|-)\s*(\d+)\s*years?|(\d+)\+\s*years?|(\d+)\s*years?\s*(?:of\s*)?experience", re.I)
_SENIORITY_PATTERN = re.compile(r"\b(junior|associate|mid.level|senior|lead|staff|principal|director|manager|head of)\b", re.I)
_BULLET_PATTERN = re.compile(r"^\s*[•\-\*\u2022\u2023\u25E6]\s+(.+)$", re.M)
_DATE_RANGE_PATTERN = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s*\d{4}\s*[-–—to]+\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s*\d{4}|(\d{4})\s*[-–—to]+\s*(\d{4}|present|current|now)", re.I)
_METRIC_PATTERN = re.compile(r"\d+[\.,]?\d*\s*(%|x|times|k\b|m\b|bn\b|ms\b|s\b|hrs?\b|days?\b|users?|customers?|records?|queries|requests?|models?|projects?|teams?)", re.I)
_JS_SIGNAL = re.compile(r"(window\.|\.js\b|NREUM|const\s+\w|function\s*\(|=>\s*\{|\.init\s*=|webpack|bundle\.js)", re.I)

_BOILERPLATE_HEADER = re.compile(
    r"^\s*(about\s+(us|the\s+company|revolut|\w+)|life\s+at\s+\w+|who\s+we\s+are|"
    r"our\s+(culture|values|mission|story|team|people)|"
    r"diversity|inclusion|equal\s+opportunity|"
    r"benefits?(\s+&\s+perks?)?|perks?(\s+&\s+benefits?)?|"
    r"compensation|why\s+(join|work\s+(at|for|with))|"
    r"what\s+we\s+offer|our\s+offer|employee\s+perks?)\s*[:\-]?\s*$",
    re.I | re.M,
)

def _strip_boilerplate(text: str) -> str:
    """Remove trailing employer-branding / D&I / benefits blocks from a JD."""
    match = _BOILERPLATE_HEADER.search(text)
    if match:
        # Keep everything before the first boilerplate header
        return text[:match.start()].rstrip()
    return text


# ── JD parser ─────────────────────────────────────────────────────────────────

def is_corrupt_jd(text: str) -> bool:
    """Detect scraped JS/HTML garbage masquerading as a JD."""
    if not text or len(text) < 50:
        return True
    sample = text[:2000]
    js_hits = len(_JS_SIGNAL.findall(sample))
    printable_ratio = sum(1 for c in sample if c.isprintable() and c not in '<>{}[]();') / max(len(sample), 1)
    return js_hits >= 3 or printable_ratio < 0.6


def parse_jd(text: str) -> JDChunks:
    chunks = JDChunks()

    if is_corrupt_jd(text):
        chunks.title = "[JD content unavailable — scraping error]"
        chunks.raw_truncated = text[:200]
        return chunks

    text = _strip_boilerplate(text)
    chunks.raw_truncated = text[:3000]
    lines = text.splitlines()

    # Title: first non-empty line under 120 chars
    for line in lines[:5]:
        stripped = line.strip()
        if stripped and len(stripped) < 120:
            chunks.title = stripped
            break

    # Seniority level from title or body
    seniority_match = _SENIORITY_PATTERN.search(text[:500])
    if seniority_match:
        chunks.seniority_level = seniority_match.group(1).title()

    # Years required
    years_match = _YEARS_PATTERN.search(text)
    if years_match:
        chunks.years_required = years_match.group(0)

    # Extract sections by finding headers + collecting bullets until next header
    sections = _split_into_sections(text, _JD_SECTION_PATTERNS)

    if "required" in sections:
        chunks.required_skills = _extract_bullets_and_sentences(sections["required"])[:15]

    if "preferred" in sections:
        chunks.preferred_skills = _extract_bullets_and_sentences(sections["preferred"])[:10]

    if "responsibilities" in sections:
        chunks.responsibilities = _extract_bullets_and_sentences(sections["responsibilities"])[:8]

    if "experience" in sections:
        chunks.experience_requirements = _extract_bullets_and_sentences(sections["experience"])[:8]

    # Seniority signals: ownership/scale/leadership/complexity phrases
    chunks.seniority_signals = _extract_seniority_signals(text)

    # Domain detection
    chunks.domain = _detect_domain(text)

    return chunks


# ── Resume parser ──────────────────────────────────────────────────────────────

_SECTION_HEADER = re.compile(r"^(ABOUT ME|SUMMARY|EXPERIENCE|SKILLS|EDUCATION|CERTIFICATIONS?|PROJECTS?|ACHIEVEMENTS?|AWARDS?|PUBLICATIONS?)\s*$", re.I | re.M)
_KNOWN_HEADERS = {"about me", "summary", "experience", "skills", "education", "certifications", "certification", "projects", "achievements", "awards"}


def parse_resume(text: str, name: str = "") -> ResumeChunks:
    chunks = ResumeChunks()
    chunks.raw_truncated = text[:3000]

    # Use provided name if available (from CSV metadata)
    if name:
        chunks.name = name
    else:
        lines = text.splitlines()
        for line in lines[:5]:
            stripped = line.strip()
            if (stripped
                    and len(stripped) < 60
                    and stripped.lower() not in _KNOWN_HEADERS
                    and not _DATE_RANGE_PATTERN.search(stripped)
                    and not re.search(r"@|http|linkedin|\d{5}", stripped, re.I)):
                chunks.name = stripped
                break

    # Split text into labelled sections using ALL-CAPS headers
    raw_sections = _split_by_caps_headers(text)

    summary_text = raw_sections.get("about me") or raw_sections.get("summary") or ""
    chunks.summary = summary_text.strip()[:500]

    skills_text = raw_sections.get("skills") or ""
    if not skills_text:
        # Inline format: "Skills: Python (statsmodels, scipy...), SQL, R" — content on same line as header
        m = re.search(r'\bskills?\s*:\s*(.+)', text, re.I)
        if m:
            skills_text = m.group(1).strip()
    if skills_text:
        chunks.skills = _split_skills(skills_text)
    else:
        # Fallback: extract tech keywords from full text
        chunks.skills = _extract_tech_mentions(text)

    exp_text = raw_sections.get("experience") or text  # fall back to full text
    chunks.work_history = _parse_work_history(exp_text)
    chunks.tech_in_experience = _extract_tech_mentions(exp_text)
    chunks.achievement_bullets = _extract_achievement_bullets(exp_text)

    edu_text = raw_sections.get("education") or ""
    chunks.education = _extract_bullets_and_sentences(edu_text)[:5]

    cert_text = raw_sections.get("certifications") or raw_sections.get("certification") or ""
    chunks.certifications = _extract_bullets_and_sentences(cert_text)[:6]

    chunks.total_years = _estimate_total_years(chunks.work_history, text)

    return chunks


def _split_by_caps_headers(text: str) -> dict[str, str]:
    """Split resume by ALL-CAPS section headers (EXPERIENCE, SKILLS, etc.)."""
    parts = {}
    current_key = None
    current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped.upper() == stripped and stripped.lower() in _KNOWN_HEADERS and len(stripped) < 30:
            if current_key is not None:
                parts[current_key] = "\n".join(current_lines).strip()
            current_key = stripped.lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_key is not None:
        parts[current_key] = "\n".join(current_lines).strip()

    return parts


# ── Formatters: judge-specific compact text ───────────────────────────────────

def format_for_skills_judge(jd: JDChunks, resume: ResumeChunks) -> tuple[str, str]:
    """Skills judge needs: JD required/preferred skills | Resume skills + tech in experience."""
    jd_block = _build_block("JOB REQUIREMENTS", [
        ("Title", jd.title),
        ("Domain", jd.domain),
        ("Required Skills", _bullet_list(jd.required_skills) or jd.raw_truncated[:800]),
        ("Preferred Skills", _bullet_list(jd.preferred_skills)),
    ])

    # Include work history bullets so the judge can find skills mentioned in
    # experience context (e.g. "GitHub Actions", "SOC2", specific AWS services)
    # that the keyword extractor may not surface in the skills list.
    history_lines = []
    for role in resume.work_history[:4]:
        history_lines.append(f"{role.title} at {role.company} ({role.duration})")
        history_lines.extend(f"  • {b}" for b in role.bullets[:5])

    resume_block = _build_block("CANDIDATE PROFILE", [
        ("Name", resume.name),
        ("Core Skills", _comma_list(resume.skills) or resume.raw_truncated[:600]),
        ("Technologies Used (from experience)", _comma_list(resume.tech_in_experience)),
        ("Summary", resume.summary),
        ("Work History", "\n".join(history_lines)),
    ])

    return jd_block, resume_block


def format_for_experience_judge(jd: JDChunks, resume: ResumeChunks) -> tuple[str, str]:
    """Experience judge: JD responsibilities/experience req | Resume work history."""
    jd_block = _build_block("JOB REQUIREMENTS", [
        ("Title", jd.title),
        ("Key Responsibilities", _bullet_list(jd.responsibilities) or jd.raw_truncated[:600]),
        ("Experience Required", _bullet_list(jd.experience_requirements)),
        ("Years Required", jd.years_required),
    ])

    history_lines = []
    for role in resume.work_history[:5]:
        history_lines.append(f"{role.title} at {role.company} ({role.duration})")
        history_lines.extend(f"  • {b}" for b in role.bullets[:4])

    resume_block = _build_block("CANDIDATE EXPERIENCE", [
        ("Name", resume.name),
        ("Total Experience", f"~{resume.total_years:.0f} years" if resume.total_years else "Unknown"),
        ("Work History", "\n".join(history_lines) or resume.raw_truncated[:800]),
    ])

    return jd_block, resume_block


def format_for_seniority_judge(jd: JDChunks, resume: ResumeChunks) -> tuple[str, str]:
    """Seniority judge: JD seniority signals | Resume titles + progression."""
    jd_block = _build_block("JOB SENIORITY REQUIREMENTS", [
        ("Title", jd.title),
        ("Seniority Level", jd.seniority_level),
        ("Years Required", jd.years_required),
        ("Seniority Signals in JD", _bullet_list(jd.seniority_signals) or _bullet_list(jd.responsibilities[:5])),
    ])

    titles_line = " → ".join(r.title for r in resume.work_history if r.title)[:3]
    resume_block = _build_block("CANDIDATE SENIORITY EVIDENCE", [
        ("Name", resume.name),
        ("Career Progression", titles_line),
        ("Total Experience", f"~{resume.total_years:.0f} years" if resume.total_years else "Unknown"),
        ("Key Achievements (scale/ownership/leadership)", _bullet_list(resume.achievement_bullets[:8]) or resume.raw_truncated[:600]),
    ])

    return jd_block, resume_block


def format_for_relevancy_judge(jd: JDChunks, resume: ResumeChunks) -> tuple[str, str]:
    """Relevancy judge: JD domain/core requirements | Resume summary + all experience bullets."""
    jd_block = _build_block("JOB DESCRIPTION (CORE)", [
        ("Title", jd.title),
        ("Domain", jd.domain),
        ("Core Requirements", _bullet_list(jd.required_skills[:8] + jd.responsibilities[:5])),
    ])

    all_bullets = []
    for role in resume.work_history[:5]:
        all_bullets.extend(f"[{role.title}] {b}" for b in role.bullets[:5])

    resume_block = _build_block("CANDIDATE EXPERIENCE STATEMENTS", [
        ("Summary", resume.summary),
        ("Experience Bullets", _bullet_list(all_bullets[:25]) or resume.raw_truncated[:800]),
    ])

    return jd_block, resume_block


def format_for_impact_judge(resume: ResumeChunks) -> str:
    """Impact judge only needs resume achievement bullets — no JD."""
    bullets = resume.achievement_bullets[:20]
    if not bullets:
        # Fallback: all bullets from work history
        for role in resume.work_history:
            bullets.extend(role.bullets)
        bullets = bullets[:20]

    if not bullets:
        return resume.raw_truncated[:2000]

    return "RESUME ACHIEVEMENT STATEMENTS:\n" + "\n".join(f"• {b}" for b in bullets)


def format_for_coach_judge(jd: JDChunks, resume: ResumeChunks) -> tuple[str, str]:
    """Coach judge: compact summary of both sides for synthesis."""
    jd_block = _build_block("TARGET ROLE", [
        ("Title", jd.title),
        ("Domain", jd.domain),
        ("Seniority", jd.seniority_level),
        ("Years Required", jd.years_required),
        ("Key Requirements", _bullet_list((jd.required_skills + jd.responsibilities)[:10])),
    ])

    resume_block = _build_block("CANDIDATE", [
        ("Name", resume.name),
        ("Summary", resume.summary),
        ("Total Experience", f"~{resume.total_years:.0f} years" if resume.total_years else "Unknown"),
        ("Core Skills", _comma_list(resume.skills[:20])),
        ("Recent Roles", " | ".join(f"{r.title} @ {r.company}" for r in resume.work_history[:3])),
    ])

    return jd_block, resume_block


# ── Internal helpers ───────────────────────────────────────────────────────────

def _split_into_sections(text: str, patterns: dict) -> dict[str, str]:
    """Find section headers and extract text until the next header."""
    found = []
    for name, pattern in patterns.items():
        for m in pattern.finditer(text):
            found.append((m.start(), m.end(), name))

    found.sort(key=lambda x: x[0])

    sections = {}
    for i, (start, end, name) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
        content = text[end:next_start].strip()
        if content:
            # Keep first occurrence of each section type
            sections.setdefault(name, content)

    return sections


def _extract_bullets_and_sentences(text: str) -> list[str]:
    """Extract bullet points; fall back to sentences if no bullets found."""
    bullets = _BULLET_PATTERN.findall(text)
    if bullets:
        return [b.strip() for b in bullets if len(b.strip()) > 10]

    # No bullets — try newlines first, then period-split sentences
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
    if lines:
        return lines[:15]

    # Flat paragraph — split on period+space
    sentences = [s.strip() for s in re.split(r'\.\s+', text) if len(s.strip()) > 15]
    return sentences[:15]


_ROLE_TITLE_PATTERN = re.compile(
    r"\b(engineer|developer|scientist|analyst|manager|architect|lead|intern|"
    r"consultant|specialist|director|officer|associate|researcher|designer)\b",
    re.I,
)


def _parse_work_history(text: str) -> list[RoleBlock]:
    """Extract role blocks from experience section."""
    roles = []
    lines = text.splitlines()
    n = len(lines)

    current: Optional[RoleBlock] = None
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        date_match = _DATE_RANGE_PATTERN.search(stripped)
        if date_match:
            if current:
                roles.append(current)

            duration = date_match.group(0)
            without_date = _DATE_RANGE_PATTERN.sub("", stripped).strip(" |@–-•,")
            company = without_date[:60] if without_date else ""

            # Look at surrounding lines for role title — check next 3 lines
            title = ""
            for offset in range(1, 4):
                if i + offset < n:
                    candidate = lines[i + offset].strip()
                    if candidate and _ROLE_TITLE_PATTERN.search(candidate) and len(candidate) < 80:
                        title = candidate
                        break

            # Also check previous non-empty line
            if not title:
                for back in range(i - 1, max(i - 3, -1), -1):
                    candidate = lines[back].strip()
                    if candidate and _ROLE_TITLE_PATTERN.search(candidate) and len(candidate) < 80:
                        title = candidate
                        break

            current = RoleBlock(title=title or company or duration, company=company, duration=duration)
            i += 1
            continue

        if current is not None:
            bullet_match = _BULLET_PATTERN.match(line)
            if bullet_match:
                current.bullets.append(bullet_match.group(1).strip())
            elif len(stripped) > 30 and not _DATE_RANGE_PATTERN.search(stripped):
                # Paragraph-style bullet — sentence starting with capital + action verb
                if re.match(r"^[A-Z][a-z]", stripped) and len(current.bullets) < 8:
                    current.bullets.append(stripped[:250])

        i += 1

    if current:
        roles.append(current)

    return roles[:8]


def _split_skills(text: str) -> list[str]:
    """Split a skills string on delimiters but preserve parenthetical groups intact.

    "Python (statsmodels, scipy, pandas), R, SQL" → ["Python (statsmodels, scipy, pandas)", "R", "SQL"]
    Plain comma-split would destroy "Python (statsmodels, scipy, pandas)" into fragments.
    """
    skills: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in ",|•·\n" and depth == 0:
            s = "".join(buf).strip()
            if 2 < len(s) < 80:
                skills.append(s)
            buf = []
        else:
            buf.append(ch)
    if buf:
        s = "".join(buf).strip()
        if 2 < len(s) < 80:
            skills.append(s)
    return skills[:40]


def _extract_tech_mentions(text: str) -> list[str]:
    """Extract technology names mentioned in experience text."""
    TECH_KEYWORDS = {
        "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#", "scala",
        "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "helm",
        "pytorch", "tensorflow", "scikit-learn", "sklearn", "huggingface", "langchain",
        "spark", "kafka", "airflow", "dbt", "snowflake", "bigquery", "databricks",
        "react", "vue", "angular", "next.js", "fastapi", "django", "flask",
        "git", "linux", "bash", "rest", "graphql", "grpc", "mlflow", "wandb",
        "pandas", "numpy", "matplotlib", "seaborn", "llm", "rag", "fine-tuning",
        "bert", "gpt", "transformers", "embeddings", "vector database", "pinecone", "weaviate",
    }
    text_lower = text.lower()
    found = sorted({kw for kw in TECH_KEYWORDS if kw in text_lower})
    return found


_ACTION_VERBS = re.compile(
    r"\b(led|built|designed|architected|improved|reduced|increased|deployed|"
    r"launched|delivered|optimised|optimized|saved|generated|drove|scaled|migrated|"
    r"integrated|automated|mentored|managed|trained|created|developed|implemented|"
    r"engineered|researched|established|spearheaded|owned|shipped|refactored|"
    r"analyzed|analysed|coordinated|produced|achieved|transformed)\b",
    re.I,
)


def _extract_achievement_bullets(text: str) -> list[str]:
    """Extract achievement statements — handles both bullet-symbol and paragraph styles."""
    candidates = []

    # Style 1: explicit bullet symbols (•, -, *)
    for b in _BULLET_PATTERN.findall(text):
        candidates.append(b.strip())

    # Style 2: paragraph sentences — lines > 60 chars starting with capital + action verb
    if len(candidates) < 3:
        for line in text.splitlines():
            stripped = line.strip()
            if (len(stripped) > 60
                    and re.match(r"^[A-Z][a-z]", stripped)
                    and _ACTION_VERBS.search(stripped)
                    and not _DATE_RANGE_PATTERN.search(stripped)
                    and stripped.lower() not in _KNOWN_HEADERS):
                candidates.append(stripped[:300])

    # Score by signal strength
    scored = []
    for b in candidates:
        score = 0
        if _METRIC_PATTERN.search(b):
            score += 2
        if _ACTION_VERBS.search(b):
            score += 1
        if score > 0:
            scored.append((score, b))

    scored.sort(key=lambda x: -x[0])
    return [b for _, b in scored[:20]]


def _extract_seniority_signals(text: str) -> list[str]:
    """Find sentences/phrases indicating seniority expectations."""
    SIGNAL_PATTERNS = [
        re.compile(r".{0,80}(lead|own|mentor|architect|drive|scale|cross.functional|end.to.end|ownership|strategy|decision.making).{0,80}", re.I),
        re.compile(r".{0,60}(\d+\+?\s*years?|senior|staff|principal|director).{0,60}", re.I),
    ]
    signals = []
    for pat in SIGNAL_PATTERNS:
        for m in pat.finditer(text):
            snippet = m.group(0).strip()
            if snippet not in signals:
                signals.append(snippet[:120])
    return signals[:8]


def _detect_domain(text: str) -> str:
    DOMAINS = {
        "Machine Learning / AI": [
            "machine learning", "deep learning", "llm", "nlp", "computer vision",
            "neural network", "ai engineer", "ml engineer", "data scientist",
            "generative ai", "large language model", "langchain", "langgraph",
            "huggingface", "transformers", "embeddings", "rag", "retrieval",
            "pytorch", "tensorflow", "foundation model", "agentic", "ai/ml",
        ],
        "Data Engineering": ["data pipeline", "etl", "spark", "kafka", "airflow", "dbt", "data warehouse", "lakehouse"],
        "Backend Engineering": ["api", "microservices", "backend", "server-side", "rest", "grpc", "database"],
        "Frontend Engineering": ["react", "vue", "angular", "frontend", "ui/ux", "css", "html"],
        "DevOps / MLOps": ["devops", "mlops", "ci/cd", "docker", "kubernetes", "terraform", "devsecops"],
        "Data Analysis": ["analyst", "tableau", "power bi", "excel", "sql", "reporting", "dashboard"],
        "Security / Pentesting": ["penetration testing", "pentest", "cybersecurity", "vulnerability", "ethical hacking"],
    }
    text_lower = text.lower()
    # Extract first 200 chars as title region — title keyword matches count 3x
    title_region = text_lower[:200]
    best = ("General", 0)
    for domain, keywords in DOMAINS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        title_bonus = sum(3 for kw in keywords if kw in title_region)
        total = score + title_bonus
        if total > best[1]:
            best = (domain, total)
    return best[0]


def _estimate_total_years(work_history: list, text: str) -> Optional[float]:
    """Rough estimate of total experience years from date ranges."""
    import datetime
    total = 0.0
    found_any = False

    for m in _DATE_RANGE_PATTERN.finditer(text):
        groups = m.groups()
        # groups: (start_month_word, end_month_word, start_year_num, end_year_num_or_present)
        try:
            if groups[0]:  # month-year format
                pass  # complex parsing, skip
            elif groups[2]:
                start_year = int(groups[2])
                end_str = groups[3].lower() if groups[3] else "present"
                end_year = datetime.date.today().year if end_str in ("present", "current", "now") else int(end_str)
                total += max(0, end_year - start_year)
                found_any = True
        except (ValueError, TypeError):
            continue

    return total if found_any else None


def _build_block(header: str, fields: list[tuple[str, str]]) -> str:
    lines = [f"=== {header} ==="]
    for label, value in fields:
        if value and str(value).strip():
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _bullet_list(items: list) -> str:
    return "\n".join(f"• {i}" for i in items if i) if items else ""


def _comma_list(items: list) -> str:
    return ", ".join(str(i) for i in items if i) if items else ""
