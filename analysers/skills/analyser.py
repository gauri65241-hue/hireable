from analysers.base import run_with_retry, load_calibration
from analysers.skills.schema import SkillsOutput
from analysers.skills import prompts


# ── Semantic taxonomy chunking (lives here — used only by skills) ─────────────

def extract_semantic_chunks(job_description: str, resume_text: str, skills_taxonomy: dict) -> dict:
    if not skills_taxonomy:
        return {}
    combined = f"{job_description} {resume_text}".lower()
    chunk_defs = {
        "technical_skills":   {"keywords": ["python","java","javascript","sql","aws","docker","kubernetes","react","node","api","database","cloud","devops","git","linux","machine learning","ai","data science","algorithm","programming"], "priority": 1},
        "business_skills":    {"keywords": ["strategy","management","leadership","project","product","marketing","sales","revenue","growth","business","customer","stakeholder","negotiation","planning","analysis","reporting","communication"], "priority": 2},
        "domain_expertise":   {"keywords": ["healthcare","finance","legal","education","retail","manufacturing","logistics","supply chain","ecommerce","fintech","biotech","pharmaceutical","insurance","real estate","energy"], "priority": 3},
        "process_methodology":{"keywords": ["agile","scrum","kanban","waterfall","lean","six sigma","itil","devops","cicd","testing","quality assurance","compliance","risk management","governance","documentation"], "priority": 4},
        "soft_skills":        {"keywords": ["teamwork","collaboration","communication","problem solving","critical thinking","adaptability","creativity","innovation","mentoring","leadership","time management","analytical","strategic"], "priority": 5},
    }
    scored = {}
    for name, cfg in chunk_defs.items():
        hits = [kw for kw in cfg["keywords"] if kw in combined]
        score = sum(combined.count(kw) for kw in hits)
        if score > 0:
            scored[name] = {"score": score, "priority": cfg["priority"]}
    top3 = sorted(scored, key=lambda n: (scored[n]["score"], -scored[n]["priority"]), reverse=True)[:3]

    def _tech(t, j, r):
        c = f"{j} {r}".lower()
        out = {"tools": {}, "concepts": {}}
        for dd in t.values():
            for cat, items in dd.get("tools", {}).items():
                m = [i for i in items if i.lower() in c]
                if m: out["tools"].setdefault(cat, []).extend(m)
            for cat, items in dd.get("concepts", {}).items():
                m = [i for i in items if i.lower() in c]
                if m: out["concepts"].setdefault(cat, []).extend(m)
        return out

    def _biz(t, j, r):
        c = f"{j} {r}".lower()
        out = {"tools": {}, "concepts": {}, "soft_skills": []}
        biz = {"sales_and_marketing", "executive_and_management", "hr_and_people_ops"}
        for name, data in t.items():
            if name not in biz: continue
            for cat, items in data.get("tools", {}).items():
                m = [i for i in items if i.lower() in c]
                if m: out["tools"].setdefault(cat, []).extend(m)
            for cat, items in data.get("concepts", {}).items():
                m = [i for i in items if i.lower() in c]
                if m: out["concepts"].setdefault(cat, []).extend(m)
            out["soft_skills"].extend(s for s in data.get("soft_skills", []) if s.lower() in c)
        return out

    def _domain(t, j, r):
        c = f"{j} {r}".lower()
        doms = {"healthcare_and_medical","finance_and_accounting","legal_and_compliance","supply_chain_and_logistics","culinary_and_kitchen"}
        out = {}
        for name, data in t.items():
            if name not in doms: continue
            titles = [x for x in data.get("titles", []) if x.lower() in c]
            concepts = {k: [i for i in v if i.lower() in c] for k, v in data.get("concepts", {}).items() if any(i.lower() in c for i in v)}
            tools = {k: [i for i in v if i.lower() in c] for k, v in data.get("tools", {}).items() if any(i.lower() in c for i in v)}
            if titles or concepts or tools: out[name] = {"titles": titles, "concepts": concepts, "tools": tools}
        return out

    def _process(t, j, r):
        c = f"{j} {r}".lower()
        out = {"concepts": {}}
        for data in t.values():
            for cat, items in data.get("concepts", {}).items():
                m = [i for i in items if i.lower() in c]
                if m: out["concepts"].setdefault(cat, []).extend(m)
        return out

    def _soft(t, j, r):
        c = f"{j} {r}".lower()
        found = set()
        for data in t.values():
            found.update(s for s in data.get("soft_skills", []) if s.lower() in c)
        return {"soft_skills": list(found)}

    extractors = {
        "technical_skills":    _tech,
        "business_skills":     _biz,
        "domain_expertise":    _domain,
        "process_methodology": _process,
        "soft_skills":         _soft,
    }
    return {name: extractors[name](skills_taxonomy, job_description, resume_text) for name in top3}


def _skill_weight(skill_name: str, frequencies: dict, total_jobs: int) -> float:
    """
    Return a demand weight in [1.0, 2.0].
    Skills absent from the jobs table get 1.0 (neutral).
    Skills required by every job get 2.0 (double penalised when missing).
    """
    if not frequencies or not total_jobs:
        return 1.0
    return 1.0 + (frequencies.get(skill_name.lower(), 0) / total_jobs)


def score(analyser_output: dict, skill_frequencies: dict | None = None) -> dict:
    analysis = list(analyser_output.get("skill_analysis", []))
    required = analyser_output.get("required_skills", [])
    present_set = set(analyser_output.get("present_skills", []))

    # LLM truncation: nemo runs out of tokens after generating top-level fields and
    # only writes absent-skill entries in skill_analysis. Synthesize missing entries
    # from present_skills (which the model populated correctly before truncation).
    if required and len(analysis) < len(required):
        analysed = {item.get("skill", item.get("skill_requirement", "")) for item in analysis}
        for skill in required:
            if skill not in analysed:
                analysis.append({
                    "skill": skill,
                    "is_present": 1 if skill in present_set else 0,
                    "evidence_strength": "synthesized",
                })

    if not analysis:
        return {"score": 0.0, "present": 0, "missing": 0, "total": 0, "missing_skills": []}

    from analysers.skills.semantic_filter import filter_and_reclassify
    analysis = filter_and_reclassify(analysis)

    if not analysis:
        return {"score": 0.0, "present": 0, "missing": 0, "total": 0, "missing_skills": []}

    total_jobs = sum(skill_frequencies.values()) if skill_frequencies else 0

    total = len(analysis)
    present = 0
    partial = 0
    partial_evidence_skills = []
    total_weight = 0.0
    present_weight = 0.0

    for item in analysis:
        skill_name = item.get("skill", item.get("skill_requirement", "unknown"))
        w = _skill_weight(skill_name, skill_frequencies, total_jobs)
        value = item.get("is_present")
        strength = str(item.get("evidence_strength", "")).lower()

        total_weight += w
        if value == 1:
            if strength in {"weak", "medium"}:
                partial += 1
                present_weight += w * 0.5
                partial_evidence_skills.append(skill_name)
            else:
                present += 1
                present_weight += w
        elif value == 0.5:
            partial += 1
            present_weight += w * 0.5
        # missing: contributes 0 to present_weight

    missing_skills = [
        s.get("skill", s.get("skill_requirement", "unknown"))
        for s in analysis if s.get("is_present") == 0
    ]

    # Use frequency-weighted score when frequencies available, simple ratio otherwise
    if skill_frequencies and total_weight > 0:
        computed_score = round(present_weight / total_weight, 3)
    else:
        computed_score = round((present + 0.5 * partial) / total, 3) if total else 0.0

    out = {
        "score": computed_score,
        "present": present, "partial": partial,
        "missing": total - present - partial,
        "total": total, "missing_skills": missing_skills,
        "frequency_weighted": bool(skill_frequencies and total_weight > 0),
    }
    if partial_evidence_skills:
        out["partial_evidence_skills"] = partial_evidence_skills

    if skill_frequencies and missing_skills:
        out["missing_skills_with_frequency"] = [
            {
                "skill": name,
                "market_frequency": round(skill_frequencies.get(name.lower(), 0) / (total_jobs or 1), 3),
                "frequency_count": skill_frequencies.get(name.lower(), 0),
            }
            for name in missing_skills
        ]

    return out


class SkillsAnalyser:
    name = "skills"

    def __init__(self, confidence_threshold: int = 75, max_retries: int = 3):
        self._ct = confidence_threshold
        self._mr = max_retries

    async def run(self, resume_text: str, job_description: str,
                  llm_model: str = "deepseek-r1:14b", skills_taxonomy: dict | None = None,
                  api_key: str = "", api_base_url: str = "",
                  arbitration_context: str = "") -> dict:
        calibration = load_calibration(self.name)

        def _build(resume, jd, taxonomy, *, attempt, prev_confidence, confidence_threshold, **kwargs):
            return prompts.build_prompt(resume, jd, taxonomy,
                                        attempt=attempt, prev_confidence=prev_confidence,
                                        confidence_threshold=confidence_threshold,
                                        calibration_note=calibration, **kwargs)

        return await run_with_retry(
            _build, (resume_text, job_description, skills_taxonomy),
            SkillsOutput, llm_model, self._mr, self._ct,
            extra_key="extracted_chunks", api_key=api_key, api_base_url=api_base_url,
            judge_name="skills", resume_text=resume_text,
            extra_system_context=arbitration_context,
        )
