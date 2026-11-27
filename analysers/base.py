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
from paths import STATE_DIR as _STATE_DIR
JUDGE_FEEDBACK_FILE = os.path.join(_STATE_DIR, "judge_feedback.jsonl")

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


# ── Text normalisation + injection handling ───────────────────────────────────
#
# Policy: untrusted text (resume, JD, scraped job pages) is NORMALISED first and only
# then inspected, because every cheap evasion (zero-width characters, fullwidth letters,
# Cyrillic look-alikes, bidi tricks) works by making the *string* differ from what a
# reader/LLM sees. Instruction-like spans are NEUTRALISED (replaced by a visible marker)
# and reported as flags rather than rejecting the whole submission: a security
# engineer's resume that quotes "ignore previous instructions" must not be a 422, and
# an attacker must not get a pass/fail oracle (error messages never echo patterns).

import unicodedata

_INVISIBLE_RE = re.compile(
    r'[\u00ad\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]'
)
_CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Length-preserving look-alike folding, used ONLY to build the string we scan, so
# match offsets map 1:1 onto the normalised text we then edit.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ո": "n", "ս": "u", "ɡ": "g", "ⅼ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X",
    "ο": "o", "α": "a", "ε": "e", "ι": "i", "ν": "v", "κ": "k", "ρ": "p", "τ": "t",
    "Ο": "O", "Α": "A", "Ε": "E", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Τ": "T",
    "0": "0", "ı": "i",
})

_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)', re.I),
    re.compile(r'disregard\s+(the\s+)?(above|previous|prior)', re.I),
    re.compile(r'(disregard|forget|ignore)\s+(everything|anything|what)\b.{0,40}\b(above|before|previous|earlier|written)', re.I | re.S),
    re.compile(r'you\s+are\s+now\s+(a|an|the)\b', re.I),
    re.compile(r'output\s+(only\s+)?this\s+(exact\s+)?json', re.I),
    re.compile(r'(?:assign|give|award|grant|return|output)\s+(?:me\s+|this\s+\w+\s+)?(?:maximum|perfect|highest|full)\s+(?:marks|scores?|ratings?)', re.I),
    re.compile(r'this\s+(candidate|applicant)\s+is\s+(a\s+|an\s+)?(perfect|ideal|great|excellent|top)\b', re.I),
    re.compile(r'(this|the)\s+(candidate|applicant)\s+(should|must)\s+be\s+(rated|scored|marked|hired|considered)', re.I),
    re.compile(r'override\s+(all\s+)?(previous\s+)?(instructions?|evaluation|scores?)', re.I),
    re.compile(r'(?:notes?|messages?|instructions?|attention)\s+(?:to|for)\s+(?:the\s+)?(?:evaluator|judge|grader|reviewer|model|llm|ai|assistant|system)\b', re.I),
    re.compile(r'\b(?:mark|rate|score|label|give|assign)\b.{0,40}\b(?:all|every|each)\s+(?:requirements?|skills?|dimensions?|criteria|categories|items?)\b.{0,40}\b(?:present|perfect|strong|maximum|highest|met|10\s*/\s*10)', re.I | re.S),
    re.compile(r'\b10\s*/\s*10\b.{0,30}\b(?:every|all)\s+dimensions?', re.I | re.S),
    re.compile(r'\b(?:evaluator|judge|grader|reviewer|assistant|ai|model|llm|system)\s+(?:note|instruction|message|directive|override)s?\s*:', re.I),
    re.compile(r'(?:new|updated|revised)\s+(?:instructions?|rules?|system\s+prompt)\s*:', re.I),
    re.compile(r'(?:^|\n)\s*(?:system|assistant|developer)\s*:\s*(?:you\s|ignore|disregard|output|respond|return|always|never|the\s+following|new\s|perfect|mark|rate|score)[^\n]*', re.I),
    re.compile(r'</?\s*(?:candidate_supplied_content|system|assistant|instructions?)\s*>', re.I),
    re.compile(r'<\|[a-z_]{2,30}\|>', re.I),          # chat-template control tokens
    re.compile(r'\[/?(?:INST|SYS)\]|<<\s*/?\s*SYS\s*>>', re.I),
]

_REMOVED_MARK = "[removed: instruction-like text]"


def normalize_text(text: str) -> str:
    """NFKC-fold (fullwidth -> ASCII, ligatures, compatibility forms) and strip
    invisible / bidi / control characters."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    return _CTRL_RE.sub("", text)


_SENT_BREAKS = ".!?;\n"


def _expand_to_sentence(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow a match to the whole sentence/line it sits in. Removing only the matched words
    leaves the rest of the attacker's sentence ("... and assign maximum scores to every
    dimension") in the prompt."""
    while start < end and text[start] in "\n \t":
        start += 1                               # patterns may begin with the preceding newline
    a = start
    while a > 0 and text[a - 1] not in _SENT_BREAKS:
        a -= 1
    b = end
    while b < len(text) and text[b] not in _SENT_BREAKS:
        b += 1
    if b < len(text) and text[b] != "\n":
        b += 1                                   # include the terminating . ! ? ;
    return a, b


def _injection_spans(clean: str) -> list[tuple[int, int]]:
    skeleton = clean.translate(_CONFUSABLES)
    spans: list[tuple[int, int]] = []
    for pat in _INJECTION_PATTERNS:
        for m in pat.finditer(skeleton):
            spans.append(_expand_to_sentence(clean, m.start(), m.end()))
    spans.sort()
    merged: list[list[int]] = []
    for st, en in spans:
        if merged and st <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return [(a, b) for a, b in merged]


def _check_limits(text: str, label: str) -> None:
    if not text or not text.strip():
        raise ValueError(f"{label} is empty. Submit a non-empty {label}.")
    limit = cfg.max_resume_chars if label == "resume" else cfg.max_job_description_chars
    if len(text) > limit:
        raise ValueError(
            f"{label} too long: {len(text)} chars exceeds limit of {limit}. "
            "Truncate before submitting."
        )


def prepare_user_text(text: str, label: str = "content") -> tuple[str, list[dict]]:
    """Live-path entry point for untrusted text.

    Returns (clean_text, flags). clean_text is normalised with every instruction-like
    span replaced by a visible marker; flags describe what was removed (no regexes are
    ever exposed). Raises ValueError only for empty / oversized input.
    """
    _check_limits(text, label)
    clean = normalize_text(text)
    spans = _injection_spans(clean)
    flags: list[dict] = []
    if spans:
        excerpts = [clean[a:b][:80] for a, b in spans[:5]]
        for a, b in reversed(spans):
            clean = clean[:a] + f" {_REMOVED_MARK} " + clean[b:]
        flags.append({"type": "prompt_injection", "field": label,
                      "spans_removed": len(spans), "excerpts": excerpts})
        logger.warning("prepare_user_text: neutralised %d instruction-like span(s) in %s",
                       len(spans), label)
    return clean, flags


def sanitize_user_content(text: str, label: str = "content") -> None:
    """Strict gate: raises ValueError on empty, oversized or injected input.
    Kept for callers/tests that want reject-on-detect. The error message deliberately
    does not reveal which rule fired."""
    _check_limits(text, label)
    if _injection_spans(normalize_text(text)):
        raise ValueError(f"Prompt injection detected in {label}. Submission rejected.")


def wrap_user_content(text: str) -> str:
    return f"<candidate_supplied_content>\n{text}\n</candidate_supplied_content>"


# ── Post-chunker injection isolation ─────────────────────────────────────────

# Region between a (forged) closing wrapper tag and the next (forged) opening tag —
# or the end of the chunk — is attacker-controlled "outside the data block" text.
_WRAPPER_BREAKOUT = re.compile(
    r'<\s*/\s*candidate_supplied_content\s*>.*?(?:<\s*candidate_supplied_content\s*>|\Z)',
    re.I | re.S,
)


def sanitize_chunk(text: str, label: str = "chunk") -> str:
    """Safety net applied to each formatted chunk before prompt assembly.

    Normalises, removes forged wrapper-tag break-outs (keeping legitimate content that
    follows a re-opened tag), and raises ValueError if an injection fragment survived
    upstream neutralisation. Never echoes the matching rule."""
    if not text:
        return text
    text = normalize_text(text)
    if _WRAPPER_BREAKOUT.search(text):
        logger.warning("sanitize_chunk: wrapper tag break-out in %s - region removed", label)
        text = _WRAPPER_BREAKOUT.sub(" ", text)
    if _injection_spans(text):
        raise ValueError(f"Injection fragment in {label}.")
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
    base_payload = {
        "model": model, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    # Exponential backoff on 429, respecting Retry-After header when present
    for attempt in range(4):
        # Try with json_object mode first; fall back to plain text if unsupported (400)
        payload = {**base_payload, "response_format": {"type": "json_object"}} if attempt == 0 else base_payload
        try:
            with httpx.Client(timeout=620.0) as client:
                resp = client.post(url, json=payload, headers=headers)
            if resp.status_code == 400 and attempt == 0:
                logger.info("response_format unsupported by %s, retrying without it", model)
                continue
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


# ── Native-async LLM transport ────────────────────────────────────────────────
#
# The old implementation ran the blocking client in asyncio.to_thread. wait_for() can
# cancel the awaiting coroutine but not the thread, so on a judge timeout the semaphore
# was released while the request kept running (4 in flight against a limit of 2 in a
# reproduction) and `time.sleep` back-off pinned executor threads for minutes. With real
# async I/O, cancellation closes the connection (Ollama stops generating) and the slot
# is released only when the work has actually stopped.

async def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> httpx.Response:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0), trust_env=False) as client:
        return await client.post(url, json=payload, headers=headers)


# Indirection point so tests can inject a fake transport without a network.
_http_post = _post_json


def _retry_after_seconds(resp: httpx.Response, attempt: int) -> float:
    raw = resp.headers.get("Retry-After") or resp.headers.get("X-RateLimit-Reset-Requests")
    try:
        wait = float(raw) if raw else 5.0 * (2 ** attempt)
    except ValueError:
        wait = 5.0 * (2 ** attempt)
    return max(1.0, min(wait, cfg.api_max_retry_wait))


async def _call_ollama_async(system_prompt: str, user_prompt: str, model: str,
                             json_schema: dict | None) -> dict:
    total_chars = len(system_prompt) + len(user_prompt)
    num_ctx = min(131072, max(8192, (total_chars // 4) + 4096))
    num_predict = int(os.getenv("ANALYSER_NUM_PREDICT", "3000"))
    url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    logger.info("Calling %s | ctx=%d | base=%s | schema=%s", model, num_ctx, cfg.ollama_base_url,
                bool(json_schema and cfg.structured_output))

    formats: list = []
    if json_schema and cfg.structured_output:
        formats.append(json_schema)
    formats.append("json")

    last_error = "no response"
    for fmt in formats:
        payload = {
            "model": model, "stream": False, "format": fmt,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": num_ctx, "num_predict": num_predict},
        }
        try:
            resp = await _http_post(url, payload, {}, cfg.llm_http_timeout)
        except httpx.HTTPError as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
        if resp.status_code == 400 and fmt != "json":
            last_error = f"schema format rejected: {resp.text[:120]}"
            continue                                   # older Ollama: retry with plain json mode
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        try:
            return {"success": True, "data": resp.json()["message"]["content"]}
        except (KeyError, ValueError) as e:
            return {"success": False, "error": f"malformed Ollama response: {e}"}
    return {"success": False, "error": last_error}


async def _call_api_async(system_prompt: str, user_prompt: str, model: str, api_key: str,
                          api_base_url: str, json_schema: dict | None) -> dict:
    url = (api_base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    formats: list = []
    if json_schema and cfg.structured_output:
        formats.append({"type": "json_schema",
                        "json_schema": {"name": "judge_output", "schema": json_schema}})
    formats += [{"type": "json_object"}, None]

    fmt_i, attempt, max_attempts = 0, 0, 4
    last_error = "exhausted retries"
    while attempt < max_attempts:
        payload = {"model": model, "temperature": 0.0,
                   "messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}]}
        if formats[fmt_i] is not None:
            payload["response_format"] = formats[fmt_i]
        try:
            resp = await _http_post(url, payload, headers, cfg.llm_http_timeout)
        except httpx.HTTPError as e:
            last_error = f"{type(e).__name__}: {e}"
            attempt += 1
            if attempt < max_attempts:
                await asyncio.sleep(min(2.0 * attempt, cfg.api_max_retry_wait))
            continue
        if resp.status_code == 400 and fmt_i < len(formats) - 1:
            logger.info("response_format unsupported by %s - degrading", model)
            fmt_i += 1                                  # capability probe, not a retry
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = f"HTTP {resp.status_code}"
            attempt += 1
            if attempt < max_attempts:
                wait = _retry_after_seconds(resp, attempt)
                logger.warning("%s on %s - retrying in %.0fs (%d/%d)", last_error, model, wait, attempt, max_attempts)
                await asyncio.sleep(wait)
            continue
        if resp.status_code != 200:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        try:
            return {"success": True, "data": resp.json()["choices"][0]["message"]["content"]}
        except (KeyError, IndexError, ValueError, TypeError) as e:
            return {"success": False, "error": f"malformed API response: {e}"}
    return {"success": False, "error": last_error}


async def call_llm_async(system_prompt: str, user_prompt: str, model: str = "deepseek-r1:14b",
                         temperature: float = 0.0, api_key: str = "", api_base_url: str = "",
                         json_schema: dict | None = None) -> dict:
    """Concurrency-limited, cancellation-safe LLM call. Temperature is fixed at 0 for
    judges (kept in the signature for backward compatibility)."""
    if api_key:
        async with _API_SEM:
            return await _call_api_async(system_prompt, user_prompt, model, api_key,
                                         api_base_url, json_schema)
    async with _OLLAMA_SEM:
        return await _call_ollama_async(system_prompt, user_prompt, model, json_schema)


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


def _norm_match(s: str) -> str:
    s = normalize_text(s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


class _Haystack:
    """Resume text pre-processed once per post_validate call (not once per item)."""
    __slots__ = ("padded", "bigrams")

    def __init__(self, resume_text: str):
        norm = _norm_match(resume_text)
        words = norm.split()
        self.padded = f" {norm} "
        self.bigrams = set(zip(words, words[1:]))


def _check_citation(evidence: str, resume_text) -> bool:
    """True if `evidence` really appears in the resume.

    1. whole-token substring match after normalisation (so "java" does not match
       "javascript"), else
    2. for evidence of >=4 words: >=70% of its ORDERED word pairs occur in the resume
       (tolerates trimmed/lightly-paraphrased quotes, unlike the previous order-blind
       65% bag-of-words test which any stuffing of common words could satisfy).
    Short evidence must match exactly. Pass a _Haystack to avoid recomputation."""
    hay = resume_text if isinstance(resume_text, _Haystack) else _Haystack(resume_text or "")
    ev = _norm_match(evidence or "")
    if not ev:
        return False
    if f" {ev} " in hay.padded:
        return True
    words = ev.split()
    if len(words) < 4:
        return False
    pairs = set(zip(words, words[1:]))
    return bool(pairs) and len(pairs & hay.bigrams) / len(pairs) >= 0.70


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
        # Precedence fix: previously `A or B and C` flagged EVERY absent skill with empty
        # evidence (they have nothing to cite) as "claimed present, no evidence", which
        # then fed misleading retry guidance. Only items claimed PRESENT can lack evidence.
        empty_evidence_skills = [
            item.get("skill", item.get("skill_requirement", "?"))
            for item in data.get("skill_analysis", [])
            if item.get("is_present", 0) == 1 and (
                not str(item.get("resume_evidence", "")).strip()
                or str(item.get("resume_evidence", "")).strip().lower() == "not found"
            )
        ]
        if empty_evidence_skills:
            warnings.append(f"claimed_present_no_evidence: {empty_evidence_skills[:3]}")

    # Citation validation — downgrade scores when evidence can't be found in resume text
    if resume_text:
        resume_text = _Haystack(resume_text)
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

    # A handful of outcomes is noise, and this text is injected into every prompt.
    if len(entries) < cfg.min_feedback_samples:
        return ""
    recent = entries[-10:]

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
    """Translate post_validate warning codes into retry guidance.

    Guidance is deliberately two-sided: every message tells the model to make its answer
    match the quoted evidence, never to move the score in a particular direction. (The
    old wording only ever pushed scores UP: "re-read for implicit evidence before
    scoring MISSING", "be more generous".)"""
    issues = []
    for w in warnings:
        if w == "high_conf_low_score":
            issues.append(
                "Confidence is high but the score is very low - re-verify each MISSING item "
                "against the resume text. Keep it MISSING if no supporting text exists; mark it "
                "PRESENT only if you can quote exact resume text."
            )
        elif w == "low_conf_high_score":
            issues.append(
                "Confidence is low but the score is high - verify that every evidence quote is "
                "exact before scoring PRESENT; downgrade anything you cannot quote."
            )
        elif w == "reasoning_positive_score_low":
            issues.append(
                "Your reasoning and your numeric score disagree. Keep whichever is supported by "
                "exact resume quotes and correct the other."
            )
        elif w == "reasoning_negative_score_high":
            issues.append(
                "Your reasoning and your numeric score disagree. Keep whichever is supported by "
                "exact resume quotes and correct the other."
            )
        elif w.startswith("claimed_present_no_evidence"):
            skills_str = w.split(": ", 1)[-1] if ": " in w else ""
            issues.append(
                f"Items marked PRESENT have no resume evidence quote: {skills_str}. "
                "Provide the exact quote or change them to absent."
            )
    return issues


def _build_retry_note(attempt: int, prev_confidence: int, threshold: int,
                       failure_reason: str = "", specific_issues: list | None = None) -> str:
    """Retry preamble. Does NOT disclose the confidence threshold or the previous
    confidence: telling a model "you scored 70%, 75% required" teaches it to inflate the
    self-reported number instead of improving the answer."""
    if attempt == 0:
        return ""
    parts = [f"RETRY ATTEMPT {attempt + 1}: your previous response could not be accepted."]
    if failure_reason == "parse_error":
        parts.append(
            "Your previous response was not valid JSON for the required schema. "
            "Output ONLY a JSON object. No markdown fences, no explanation text before or after."
        )
    else:
        parts.append(
            "Use exact text copied from the resume for every evidence quote. Make each judgement "
            "match its quote, and make your reasoning match your numeric fields. Do not raise or "
            "lower any judgement merely because this is a retry."
        )
    if specific_issues:
        parts.append("Specific issues to fix in this attempt:")
        for issue in specific_issues[:4]:
            parts.append(f"  -> {issue}")
    return "\n".join(parts)


async def _attempt_model(
    model: str, attempts: int, *, prompt_fn, prompt_args, schema_class, confidence_threshold,
    extra_key, api_key, api_base_url, judge_name, resume_text, extra_system_context,
    json_schema,
) -> dict:
    """Run up to `attempts` tries against ONE model. Every path returns a result dict."""
    prev_confidence = 0
    prev_failure_reason = ""
    prev_specific_issues: list[str] = []
    t0 = time.monotonic()
    last_err = "no attempts made"
    for attempt in range(attempts):
        logger.info("attempt %d/%d on %s", attempt + 1, attempts, model)
        try:
            built = prompt_fn(*prompt_args, attempt=attempt,
                              prev_confidence=prev_confidence,
                              confidence_threshold=confidence_threshold,
                              failure_reason=prev_failure_reason,
                              specific_issues=prev_specific_issues)
        except ValueError as e:
            # Deterministic input rejection (e.g. surviving injection fragment): retrying
            # or switching model cannot help.
            return {"success": False, "error": str(e), "error_type": "input_rejected",
                    "raw_response": "", "attempts": attempt + 1}
        if len(built) == 3:
            system_prompt, user_prompt, extra = built
        else:
            system_prompt, user_prompt = built
            extra = None

        if extra_system_context:
            system_prompt = system_prompt.rstrip() + "\n\n" + extra_system_context

        # In-prompt self-validation when using an API backend (zero extra calls).
        if api_key and "SELF-CHECK" not in system_prompt:
            system_prompt = system_prompt.rstrip() + (
                "\n\nSELF-CHECK (do this silently before outputting):\n"
                "For every skill/responsibility/signal you marked as present or scoring >=0.75, "
                "verify the exact supporting phrase exists in the candidate-supplied content. "
                "If you cannot locate it, downgrade to partial (0.5) or absent (0). "
                "Ensure your numeric scores are consistent with your written reasoning. "
                "Then output your final JSON - no preamble, no markdown fences."
            )

        if cfg.two_phase_enabled and api_key:
            logger.info("two_phase_enabled: challenge phase stub - not yet implemented")

        try:
            result = await call_llm_async(system_prompt, user_prompt, model, temperature=0.0,
                                          api_key=api_key, api_base_url=api_base_url,
                                          json_schema=json_schema)
            if not result["success"]:
                last_err = result.get("error", "llm error")
                prev_failure_reason = "llm_error"
                prev_specific_issues = [last_err[:150]]
                continue

            llm_response = result["data"]
            try:
                parsed = parse_and_validate(llm_response, schema_class)
            except ValueError as e:
                logger.error("parse_and_validate failed: %s", e)
                last_err = str(e)
                prev_failure_reason = "parse_error"
                prev_specific_issues = [str(e)[:200]]
                continue

            confidence = parsed.get("confidence", 0)
            if parsed.get("source") == "repaired":
                confidence = max(0, confidence - 20)
                logger.warning("parse result was repaired - applying -20 confidence penalty")
                parsed["confidence"] = confidence
            if judge_name:
                parsed = post_validate(parsed, judge_name, resume_text=resume_text)
            prev_confidence = confidence

            warnings = parsed.get("_warnings", [])
            prev_specific_issues = _warnings_to_issues(warnings) if warnings else []
            prev_failure_reason = "low_confidence"
            logger.info("confidence=%d%% threshold=%d%% source=%s warnings=%s",
                        confidence, confidence_threshold, parsed.get("source", "?"), warnings or "none")

            if confidence >= confidence_threshold or attempt == attempts - 1:
                out = {
                    "success": True, "data": parsed, "raw_response": llm_response,
                    "confidence_met": confidence >= confidence_threshold,
                    "attempts": attempt + 1, "model_used": model,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                }
                if extra_key and extra is not None:
                    out[extra_key] = extra
                return out
        except Exception as e:  # noqa: BLE001 - boundary: any failure becomes a retry/failed result
            last_err = f"{type(e).__name__}: {e}"
            prev_failure_reason = "exception"
            prev_specific_issues = [str(e)[:150]]

    return {"success": False, "error": last_err, "raw_response": "", "attempts": attempts,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1)}


async def run_with_retry(
    prompt_fn,          # callable(*args, attempt, prev_confidence, confidence_threshold, failure_reason, specific_issues) -> (sys, usr) or (sys, usr, extra)
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
    """Primary model with retries, then ONE attempt on the fallback model.

    (The previous version placed the fallback after a loop whose every iteration
    returned, so the fallback model could never run.)"""
    common = dict(
        prompt_fn=prompt_fn, prompt_args=prompt_args, schema_class=schema_class,
        confidence_threshold=confidence_threshold, extra_key=extra_key, api_key=api_key,
        api_base_url=api_base_url, judge_name=judge_name, resume_text=resume_text,
        extra_system_context=extra_system_context,
        json_schema=schema_class.model_json_schema() if schema_class is not None else None,
    )
    result = await _attempt_model(llm_model, max(1, max_retries), **common)
    if result.get("success") or result.get("error_type") == "input_rejected":
        return result

    fallback = cfg.fallback_model if not api_key else cfg.openrouter_fast_model
    if fallback and fallback != llm_model:
        logger.info("Primary model %s failed all attempts - trying fallback %s", llm_model, fallback)
        fb = await _attempt_model(fallback, 1, **common)
        if fb.get("success"):
            fb["used_fallback"] = True
            fb["fallback_model"] = fallback
            fb["attempts"] = result.get("attempts", max_retries) + fb.get("attempts", 1)
            return fb
        result["fallback_error"] = fb.get("error", "")
    return result
