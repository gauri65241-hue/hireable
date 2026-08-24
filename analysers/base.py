"""
Shared infrastructure for all analysers:
  - LLM calling (Ollama + OpenAI-compatible API)
  - Injection detection + content isolation
  - JSON extraction + Pydantic parse/validate
  - Retry runner
  - Calibration helpers
"""

import asyncio
import json
import logging
import os
import re
import time

import httpx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_FEEDBACK_FILE = os.path.join(BASE_DIR, "data", "judge_feedback.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("AnalyserEngine")

from analysers.config import cfg

# Semaphore size read from config so it's tunable without code changes
_OLLAMA_SEM = asyncio.Semaphore(cfg.ollama_semaphore_size)
# API backend semaphore — serialises calls to avoid free-tier 429s.
# Free-tier OpenRouter limits are ~10 req/min; 1 concurrent call + backoff keeps us safe.
# Set ANALYSER_API_SEM=3 if using a paid tier.
_API_SEM = asyncio.Semaphore(int(os.getenv("ANALYSER_API_SEM", "1")))


# ── Injection detection ───────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?|context)', re.I),
    re.compile(r'disregard\s+(the\s+)?(above|previous|prior)', re.I),
    re.compile(r'you\s+are\s+now\s+a', re.I),
    re.compile(r'output\s+(only\s+)?this\s+(exact\s+)?json', re.I),
    re.compile(r'assign\s+(maximum|perfect|highest)\s+scores?', re.I),
    re.compile(r'this\s+candidate\s+is\s+(a\s+)?(perfect|ideal|great)\s+match', re.I),
    re.compile(r'override\s+(all\s+)?(previous\s+)?instructions?', re.I),
    re.compile(r'system\s*:\s*(you\s+are|ignore|output)', re.I),
]


def sanitize_user_content(text: str, label: str = "content") -> None:
    """Raises ValueError on empty, oversized, or injected input. Call before any LLM invocation."""
    if not text or not text.strip():
        raise ValueError(f"{label} is empty. Submit a non-empty {label}.")
    limit = cfg.max_resume_chars if label == "resume" else cfg.max_job_description_chars
    if len(text) > limit:
        raise ValueError(
            f"{label} too long: {len(text)} chars exceeds limit of {limit}. "
            "Truncate before submitting."
        )
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"Prompt injection detected in {label}: "
                f"pattern '{pattern.pattern}' matched. Submission rejected."
            )


def wrap_user_content(text: str) -> str:
    return f"<candidate_supplied_content>\n{text}\n</candidate_supplied_content>"


# ── Post-chunker injection isolation ─────────────────────────────────────────

# RTL override and bidi formatting chars used to visually obscure injections
_RTL_CHARS = re.compile(r'[\u202e\u202b\u200f\u061c\u2067\u2066\u2068\u2069\u200b\u200c\u200d]')
# Closing wrapper tag appearing inside content = attempted context break-out
_WRAPPER_CLOSE = re.compile(r'</\s*candidate_supplied_content\s*>', re.I)


def sanitize_chunk(text: str, label: str = "chunk") -> str:
    """
    Called on each extracted chunk AFTER parse_jd/parse_resume, before prompt assembly.
    Strips RTL overrides, truncates at tag break-out, re-checks injection fragments.
    Returns cleaned text. Raises ValueError if an injection fragment is found.
    """
    if not text:
        return text
    text = _RTL_CHARS.sub('', text)
    m = _WRAPPER_CLOSE.search(text)
    if m:
        logger.warning("sanitize_chunk: closing wrapper tag in %s — truncating", label)
        text = text[:m.start()]
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"Injection fragment in {label}: pattern '{pattern.pattern}' matched."
            )
    return text.strip()


# ── JSON extraction helpers ───────────────────────────────────────────────────

# Pre-compiled: one alternation pass instead of 3 sequential re.sub calls.
# DOTALL|IGNORECASE handles mixed-case tags and multi-line think blocks.
_REASONING_TAG_RE = re.compile(
    r'<(think|thinking|reasoning)>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)


def _strip_reasoning_tags(text: str) -> str:
    # Fast path: skip regex entirely if no opening tag present.
    # str.find is a C-level substring search — O(n) but ~10x faster than re.search
    # for the negative case (most production outputs after first retry succeed).
    if '<think' not in text and '<reasoning' not in text:
        return text.strip()
    return _REASONING_TAG_RE.sub('', text).strip()


def _extract_outermost_json_object(text: str) -> str | None:
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    # Slice once so CPython iterates a compact char array (faster than text[i] indexing).
    for i, ch in enumerate(text[start:], start):
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _strip_markdown_fences(text: str) -> str:
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```")[0] if "```" in text else text
    elif text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 3 else text.replace("```", "")
    return text.strip()


def parse_and_validate(response: str, schema_class=None) -> dict:
    """
    Parse LLM response and validate against schema_class (Pydantic model) if provided.
    Raises ValueError on unrecoverable failure — never returns zeroed data with success=True.
    """
    from pydantic import ValidationError

    cleaned = _strip_reasoning_tags(response)
    cleaned = _strip_markdown_fences(cleaned)

    attempts = [
        cleaned,
        _extract_outermost_json_object(cleaned) or "",
        _extract_outermost_json_object(response) or "",
    ]

    for raw in attempts:
        if not raw:
            continue
        try:
            data = json.loads(raw)
            if schema_class:
                validated = schema_class.model_validate(data)
                out = validated.model_dump()
            else:
                out = data
            out["source"] = "structured"
            out["success"] = True
            return out
        except (json.JSONDecodeError, ValidationError):
            continue

    try:
        from json_repair import repair_json  # type: ignore
        repaired = repair_json(cleaned)
        data = json.loads(repaired)
        if schema_class:
            validated = schema_class.model_validate(data)
            out = validated.model_dump()
        else:
            out = data
        out["source"] = "repaired"
        out["success"] = True
        return out
    except Exception:
        pass

    raise ValueError(
        f"Unrecoverable parse failure. Response prefix: {cleaned[:300]!r}"
    )


# ── LLM calling ──────────────────────────────────────────────────────────────

def _call_api_model(system_prompt: str, user_prompt: str, model: str,
                    api_key: str, api_base_url: str = "https://openrouter.ai/api/v1",
                    temperature: float = 0.0) -> dict:
    url = api_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model, "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    # Exponential backoff on 429, respecting Retry-After header when present
    for attempt in range(4):
        try:
            with httpx.Client(timeout=620.0) as client:
                resp = client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("X-RateLimit-Reset-Requests")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 20 * (2 ** attempt)
                else:
                    wait = 20 * (2 ** attempt)  # 20, 40, 80, 160s
                logger.warning("429 rate limit on %s — retrying in %ds (attempt %d/4)", model, wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return {"success": True, "data": content}
        except Exception as e:
            if attempt == 3:
                return {"success": False, "error": str(e)}
            time.sleep(5 * (attempt + 1))
    return {"success": False, "error": "Exhausted rate-limit retries"}


def call_llm(system_prompt: str, user_prompt: str, model: str = "deepseek-r1:14b",
             temperature: float = 0.0, api_key: str = "", api_base_url: str = "") -> dict:
    if api_key:
        return _call_api_model(system_prompt, user_prompt, model, api_key,
                               api_base_url or "https://openrouter.ai/api/v1", temperature)
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
        total_chars = len(system_prompt) + len(user_prompt)
        num_ctx = min(131072, max(8192, (total_chars // 4) + 4096))
        logger.info("Calling %s | ctx=%d | base=%s", model, num_ctx, cfg.ollama_base_url)
        num_predict = int(os.getenv("ANALYSER_NUM_PREDICT", "3000"))
        llm = ChatOllama(
            model=model,
            temperature=0.0,
            seed=42,
            format="json",
            num_ctx=num_ctx,
            num_predict=num_predict,
            base_url=cfg.ollama_base_url,
        )
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return {"success": True, "data": response.content}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def call_llm_async(system_prompt: str, user_prompt: str, model: str = "deepseek-r1:14b",
                         temperature: float = 0.0, api_key: str = "", api_base_url: str = "") -> dict:
    sem = _API_SEM if api_key else _OLLAMA_SEM
    async with sem:
        return await asyncio.to_thread(
            call_llm, system_prompt, user_prompt, model, temperature, api_key, api_base_url
        )


# ── Post-validation warning layer ────────────────────────────────────────────

_POSITIVE_SIGNALS = re.compile(
    r'\b(strong|clearly|directly|explicitly|demonstrates?|proven|extensive|'
    r'solid|proficient|expert|advanced|senior|lead|owns?|built|shipped|'
    r'excellent|well.qualified|perfect match)\b', re.I
)
_NEGATIVE_SIGNALS = re.compile(
    r'\b(not found|absent|missing|no evidence|lacks?|never|none|'
    r'no mention|not present|not demonstrated|gap|insufficient)\b', re.I
)


def _check_citation(evidence: str, resume_text: str) -> bool:
    """
    Return True if evidence plausibly appears in resume_text.
    Tries exact substring match first; falls back to word overlap (≥65%).
    """
    if not evidence or not evidence.strip():
        return False
    ev = evidence.strip()
    if ev.lower() in resume_text.lower():
        return True
    ev_words = set(re.sub(r'[^\w\s]', ' ', ev.lower()).split())
    if not ev_words:
        return False
    res_words = set(re.sub(r'[^\w\s]', ' ', resume_text.lower()).split())
    return len(ev_words & res_words) / len(ev_words) >= 0.65


def post_validate(data: dict, judge: str, resume_text: str = "") -> dict:
    """
    Attach a _warnings list to the parsed judge output flagging observable anomalies.
    When resume_text is supplied, also validates resume_evidence citations and downgrades
    is_present/experience_score/is_demonstrated for items whose evidence can't be found.
    Called after every successful parse_and_validate in run_with_retry.
    """
    warnings = []

    score = data.get("score", data.get("overall_score", data.get("match_score")))
    confidence = data.get("confidence", 0)
    reasoning = data.get("overall_reasoning", data.get("reasoning", ""))

    if score is not None:
        score_f = float(score) if score <= 1.0 else float(score) / 100.0

        # High confidence but near-zero score — judge very certain about a very low score,
        # which is fine for Weak candidates but suspicious for others.
        if confidence > 85 and score_f < 0.15:
            warnings.append("high_conf_low_score")

        # Low confidence on a high score — judge unsure but scoring high.
        if confidence < 35 and score_f > 0.75:
            warnings.append("low_conf_high_score")

        # Reasoning text contradicts score direction.
        if reasoning and len(reasoning) > 20:
            has_positive = bool(_POSITIVE_SIGNALS.search(reasoning))
            has_negative = bool(_NEGATIVE_SIGNALS.search(reasoning))
            if has_positive and not has_negative and score_f < 0.20:
                warnings.append("reasoning_positive_score_low")
            if has_negative and not has_positive and score_f > 0.80:
                warnings.append("reasoning_negative_score_high")

    # Skills judge: flag individual skill_analysis items with empty evidence.
    if judge == "skills":
        empty_evidence_skills = [
            item.get("skill", item.get("skill_requirement", "?"))
            for item in data.get("skill_analysis", [])
            if not str(item.get("resume_evidence", "")).strip()
            or item.get("resume_evidence", "") == "Not found"
            and item.get("is_present", 0) == 1
        ]
        if empty_evidence_skills:
            warnings.append(f"claimed_present_no_evidence: {empty_evidence_skills[:3]}")

    # Citation validation — downgrade scores when evidence can't be found in resume text
    if resume_text:
        if judge == "skills":
            invalid = []
            for item in data.get("skill_analysis", []):
                if item.get("is_present") == 1:
                    ev = str(item.get("resume_evidence", ""))
                    if not _check_citation(ev, resume_text):
                        item["is_present"] = 0.5
                        item["_citation_invalid"] = True
                        invalid.append(item.get("skill", item.get("skill_requirement", "?")))
            if invalid:
                warnings.append(f"citation_invalid_skills: {invalid[:3]}")
                logger.info("post_validate [skills]: %d citation(s) invalid → downgraded to partial", len(invalid))

        elif judge == "experience":
            invalid = []
            for item in data.get("responsibility_analysis", []):
                if item.get("experience_score", 0) >= 0.75:
                    ev = str(item.get("resume_evidence", ""))
                    if not _check_citation(ev, resume_text):
                        item["experience_score"] = 0.5
                        item["_citation_invalid"] = True
                        invalid.append(item.get("responsibility", "?"))
            if invalid:
                warnings.append(f"citation_invalid_experience: {invalid[:3]}")
                logger.info("post_validate [experience]: %d citation(s) invalid → downgraded", len(invalid))

        elif judge == "seniority":
            invalid = []
            for item in data.get("seniority_analysis", []):
                if item.get("is_demonstrated") == 1:
                    ev = str(item.get("resume_evidence", ""))
                    if not _check_citation(ev, resume_text):
                        item["is_demonstrated"] = 0
                        item["_citation_invalid"] = True
                        invalid.append(item.get("signal", "?"))
            if invalid:
                warnings.append(f"citation_invalid_seniority: {invalid[:3]}")
                logger.info("post_validate [seniority]: %d citation(s) invalid → downgraded", len(invalid))

    if warnings:
        data["_warnings"] = warnings
        logger.warning("post_validate [%s]: %s", judge, warnings)

    return data


# ── Calibration + retry helpers ───────────────────────────────────────────────

def load_calibration(analyser_name: str) -> str:
    entries = []
    try:
        with open(JUDGE_FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    if e.get("judge") == analyser_name:
                        entries.append(e)
    except FileNotFoundError:
        return ""

    recent = entries[-5:]
    if len(recent) < 2:
        return ""

    total = len(recent)
    correct = sum(1 for e in recent if e["correct"])
    pos = [e for e in recent if e["outcome"]]
    neg = [e for e in recent if not e["outcome"]]
    avg_pos = sum(e["predicted_score"] for e in pos) / len(pos) if pos else None
    avg_neg = sum(e["predicted_score"] for e in neg) / len(neg) if neg else None

    lines = [f"\nCALIBRATION NOTE (from {total} real-world outcomes):"]
    lines.append(f"- Prediction accuracy: {correct/total:.0%} ({correct}/{total} correct)")
    if avg_pos is not None and avg_neg is not None:
        if avg_pos < 0.5:
            lines.append("- You UNDER-SCORE successful candidates. Be more generous when evidence is present but implicit.")
        elif avg_neg > 0.55:
            lines.append("- You OVER-SCORE unsuitable candidates. Apply stricter evidence standards before scoring high.")
        else:
            lines.append("- Scoring appears well-calibrated. Maintain current standards.")
    elif avg_pos is not None and avg_pos < 0.5:
        lines.append("- You UNDER-SCORE successful candidates. Be more generous when evidence is present but implicit.")
    return "\n".join(lines)


def _warnings_to_issues(warnings: list[str]) -> list[str]:
    """Translate post_validate warning codes into actionable retry guidance."""
    issues = []
    for w in warnings:
        if w == "high_conf_low_score":
            issues.append(
                "Your confidence is high but the score is very low — re-read the resume for "
                "implicit or transferable evidence before scoring MISSING."
            )
        elif w == "low_conf_high_score":
            issues.append(
                "Your confidence is low but the score is high — verify that evidence quotes "
                "are accurate before scoring PRESENT."
            )
        elif w == "reasoning_positive_score_low":
            issues.append(
                "Your reasoning uses positive language but the numeric score is low — "
                "make your score consistent with your stated reasoning."
            )
        elif w == "reasoning_negative_score_high":
            issues.append(
                "Your reasoning flags gaps but the numeric score is high — "
                "make your score consistent with your stated gaps."
            )
        elif w.startswith("claimed_present_no_evidence"):
            skills_str = w.split(": ", 1)[-1] if ": " in w else ""
            issues.append(
                f"Skills marked PRESENT have no resume evidence quote: {skills_str}. "
                "Find the exact resume quote or change is_present to 0."
            )
    return issues


def _build_retry_note(attempt: int, prev_confidence: int, threshold: int,
                       failure_reason: str = "", specific_issues: list | None = None) -> str:
    if attempt == 0:
        return ""
    parts = [
        f"RETRY ATTEMPT {attempt + 1}: Your previous response scored {prev_confidence}% confidence "
        f"(required: {threshold}%)."
    ]
    if failure_reason == "parse_error":
        parts.append(
            "⚠ Your previous response was not valid JSON and could not be parsed. "
            "Output ONLY a JSON object. No markdown fences, no explanation text before or after."
        )
    else:
        parts.append(
            "Improve your evidence quotes — use exact text copied from the resume. "
            "Be decisive: commit to PRESENT or MISSING and ensure your reasoning matches your score."
        )
    if specific_issues:
        parts.append("Specific issues to fix in this attempt:")
        for issue in specific_issues[:4]:
            parts.append(f"  → {issue}")
    return "\n".join(parts)


async def run_with_retry(
    prompt_fn,          # callable(*args, attempt, prev_confidence, confidence_threshold, failure_reason, specific_issues) → (sys, usr) or (sys, usr, extra)
    prompt_args: tuple,
    schema_class,
    llm_model: str,
    max_retries: int,
    confidence_threshold: int,
    extra_key: str = None,
    api_key: str = "",
    api_base_url: str = "",
    judge_name: str = "",
    resume_text: str = "",
    extra_system_context: str = "",
) -> dict:
    prev_confidence = 0
    prev_failure_reason = ""
    prev_specific_issues: list[str] = []
    t0 = time.monotonic()
    for attempt in range(max_retries):
        logger.info("attempt %d/%d", attempt + 1, max_retries)
        built = prompt_fn(*prompt_args, attempt=attempt,
                          prev_confidence=prev_confidence,
                          confidence_threshold=confidence_threshold,
                          failure_reason=prev_failure_reason,
                          specific_issues=prev_specific_issues)
        if len(built) == 3:
            system_prompt, user_prompt, extra = built
        else:
            system_prompt, user_prompt = built
            extra = None

        if extra_system_context:
            system_prompt = system_prompt.rstrip() + "\n\n" + extra_system_context

        # In-prompt self-validation when using an API backend (zero extra calls).
        # GrandCode-style: model checks its own evidence before finalising output.
        if api_key and "SELF-CHECK" not in system_prompt:
            system_prompt = system_prompt.rstrip() + (
                "\n\nSELF-CHECK (do this silently before outputting):\n"
                "For every skill/responsibility/signal you marked as present or scoring ≥0.75, "
                "verify the exact supporting phrase exists in the candidate-supplied content. "
                "If you cannot locate it, downgrade to partial (0.5) or absent (0). "
                "Ensure your numeric scores are consistent with your written reasoning. "
                "Then output your final JSON — no preamble, no markdown fences."
            )

        # Two-phase stub: hypothesis → challenge (API backend only, off by default).
        # Enable with ANALYSER_TWO_PHASE=true once latency budget allows.
        if cfg.two_phase_enabled and api_key:
            logger.info("two_phase_enabled: challenge phase stub — not yet implemented")

        try:
            result = await call_llm_async(system_prompt, user_prompt, llm_model,
                                          temperature=0.0, api_key=api_key, api_base_url=api_base_url)
            if not result["success"]:
                if attempt == max_retries - 1:
                    return {"success": False, "error": result["error"], "raw_response": "", "attempts": attempt + 1}
                prev_failure_reason = "llm_error"
                prev_specific_issues = [result.get("error", "")[:150]]
                continue

            llm_response = result["data"]
            try:
                parsed = parse_and_validate(llm_response, schema_class)
            except ValueError as e:
                logger.error("parse_and_validate failed: %s", e)
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e), "raw_response": llm_response, "attempts": attempt + 1}
                prev_failure_reason = "parse_error"
                prev_specific_issues = [str(e)[:200]]
                continue

            confidence = parsed.get("confidence", 0)
            if parsed.get("source") == "repaired":
                penalty = 20
                confidence = max(0, confidence - penalty)
                logger.warning("parse result was repaired — applying -%d confidence penalty", penalty)
                parsed["confidence"] = confidence
            if judge_name:
                parsed = post_validate(parsed, judge_name, resume_text=resume_text)
            prev_confidence = confidence

            # Build external signals from post_validate warnings for the next attempt
            warnings = parsed.get("_warnings", [])
            prev_specific_issues = _warnings_to_issues(warnings) if warnings else []
            prev_failure_reason = "low_confidence"

            logger.info("confidence=%d%% threshold=%d%% source=%s warnings=%s",
                        confidence, confidence_threshold, parsed.get("source", "?"), warnings or "none")

            if confidence >= confidence_threshold or attempt == max_retries - 1:
                out = {
                    "success": True, "data": parsed, "raw_response": llm_response,
                    "confidence_met": confidence >= confidence_threshold,
                    "attempts": attempt + 1,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                }
                if extra_key and extra is not None:
                    out[extra_key] = extra
                return out

            logger.warning("confidence below threshold — retrying with external signals: %s", prev_specific_issues or "none")

        except Exception as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e), "raw_response": "", "attempts": attempt + 1}
            prev_failure_reason = "exception"
            prev_specific_issues = [str(e)[:150]]

    # Primary model exhausted — try fallback once if configured
    fallback = cfg.fallback_model
    if fallback and fallback != llm_model:
        logger.info("Primary model %s failed all retries — trying fallback %s", llm_model, fallback)
        try:
            built = prompt_fn(*prompt_args, attempt=0, prev_confidence=0,
                              confidence_threshold=confidence_threshold)
            if len(built) == 3:
                fb_sys, fb_usr, fb_extra = built
            else:
                fb_sys, fb_usr = built
                fb_extra = None
            fb_result = await call_llm_async(fb_sys, fb_usr, fallback,
                                             temperature=0.0, api_key=api_key, api_base_url=api_base_url)
            if fb_result["success"]:
                parsed = parse_and_validate(fb_result["data"], schema_class)
                if judge_name:
                    parsed = post_validate(parsed, judge_name, resume_text=resume_text)
                out = {
                    "success": True, "data": parsed, "raw_response": fb_result["data"],
                    "confidence_met": parsed.get("confidence", 0) >= confidence_threshold,
                    "attempts": max_retries + 1,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    "used_fallback": True,
                    "fallback_model": fallback,
                }
                if extra_key and fb_extra is not None:
                    out[extra_key] = fb_extra
                return out
        except Exception as fb_e:
            logger.warning("Fallback model %s also failed: %s", fallback, fb_e)

    return {"success": False, "error": "Exhausted retries", "raw_response": "", "attempts": max_retries,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1)}
