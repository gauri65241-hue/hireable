"""
Pipeline profiler — cProfile + tracemalloc over a full multi-judge pass.

Usage:
    python scripts/profile_pipeline.py [--model deepseek-r1:14b] [--top N]

Output:
    - Top N functions by cumulative CPU time (ms)
    - Top N memory allocations by size (KB)
    - Per-stage breakdown: parse, prompt build, LLM call, parse+validate, DB write
    - Parse cache hit stats (chunker_compat.lru_cache)
    - pipeline_json column size before/after _slim_pipeline

Requires: a running Ollama instance OR set OPENROUTER_API_KEY + --model <api-model>
"""

import argparse
import asyncio
import cProfile
import io
import os
import pstats
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ──────────────────────────────────────────────────────────────────

RESUME = """Sarah Chen
sarah.chen@email.com | (415) 555-0192

SUMMARY
Staff Software Engineer with 9 years building distributed systems at scale.
Deep expertise in Python, Go, and cloud-native architectures on AWS.

SKILLS
Python, Go, AWS (ECS, Lambda, RDS, S3), Kubernetes, Docker, PostgreSQL,
Redis, Kafka, Terraform, GitHub Actions, Datadog, FastAPI, gRPC

EXPERIENCE
Staff Software Engineer – DataStream Inc (2020–2024)
- Designed event-driven microservices handling 2M events/day, reducing p99 latency from 800ms to 120ms
- Mentored team of 8 engineers; conducted 40+ technical interviews
- Led migration from monolith to 12 microservices; zero-downtime for 3M users
- Reduced infrastructure cost by $480k/year through auto-scaling

Senior Software Engineer – CloudBase Corp (2017–2020)
- Built real-time data pipeline processing 500k records/hour using Kafka and Flink
- Reduced query time by 65% on 200M-row PostgreSQL tables

EDUCATION
BS Computer Science – UC Berkeley (2015)
"""

JD = """Staff Software Engineer – Platform Team

Requirements
- 7+ years of software engineering experience
- 3+ years in a senior or staff-level IC role
- Strong expertise in Python or Go
- Deep experience with distributed systems (microservices, event-driven architecture)
- Hands-on AWS experience (ECS, Lambda, RDS, S3)
- Experience with Kubernetes and container orchestration
- Strong SQL and database performance optimization skills
- Demonstrated experience leading technical projects and mentoring engineers

Responsibilities
- Design and own platform services serving millions of users
- Set technical direction and engineering standards for the team
- Mentor junior and mid-level engineers
- Conduct technical interviews and contribute to hiring decisions
"""


# ── Stage timers ──────────────────────────────────────────────────────────────

class _Stage:
    def __init__(self, name: str):
        self.name = name
        self._start = None
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def _hr(label: str):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")


# ── Parse cache benchmark ─────────────────────────────────────────────────────

def benchmark_parse_cache():
    _hr("Parse cache: cold vs warm")
    from analysers.chunker_compat import parse_jd, parse_resume

    # Clear cache before measuring
    parse_jd.cache_clear()
    parse_resume.cache_clear()

    with _Stage("parse_jd cold") as s1:
        _ = parse_jd(JD)
    with _Stage("parse_jd warm (same text)") as s2:
        _ = parse_jd(JD)
    with _Stage("parse_resume cold") as s3:
        _ = parse_resume(RESUME)
    with _Stage("parse_resume warm") as s4:
        _ = parse_resume(RESUME)

    print(f"  parse_jd     cold: {s1.elapsed_ms:.3f}ms")
    print(f"  parse_jd     warm: {s2.elapsed_ms:.4f}ms  ({s1.elapsed_ms / max(s2.elapsed_ms, 0.001):.0f}x speedup)")
    print(f"  parse_resume cold: {s3.elapsed_ms:.3f}ms")
    print(f"  parse_resume warm: {s4.elapsed_ms:.4f}ms  ({s3.elapsed_ms / max(s4.elapsed_ms, 0.001):.0f}x speedup)")

    # Simulate 6-judge scenario: 1 cold + 5 warm per text
    total_cold = (s1.elapsed_ms + s3.elapsed_ms) * 6
    total_cached = s1.elapsed_ms + s3.elapsed_ms + (s2.elapsed_ms + s4.elapsed_ms) * 5
    print(f"\n  Without cache (6x each):  {total_cold:.2f}ms")
    print(f"  With cache (1 cold + 5x): {total_cached:.2f}ms")
    print(f"  Saved per analysis:       {total_cold - total_cached:.2f}ms")

    info = parse_jd.cache_info()
    print(f"\n  parse_jd   cache: hits={info.hits} misses={info.misses} size={info.currsize}/{info.maxsize}")
    info = parse_resume.cache_info()
    print(f"  parse_resume cache: hits={info.hits} misses={info.misses} size={info.currsize}/{info.maxsize}")


# ── Reasoning tag strip benchmark ─────────────────────────────────────────────

def benchmark_strip_reasoning():
    _hr("_strip_reasoning_tags: old (3x re.sub) vs new (fast path + 1x compiled)")
    from analysers.base import _strip_reasoning_tags, _REASONING_TAG_RE

    # Simulate a 30KB deepseek-r1 response
    think_block = "<think>" + ("x" * 28_000) + "</think>"
    json_part = '\n{"score": 0.85, "confidence": 90}'
    full_response = think_block + json_part
    no_think = json_part  # already stripped — fast path case

    N = 200

    with _Stage(f"new (_REASONING_TAG_RE, {N}x, with think)") as s1:
        for _ in range(N):
            _strip_reasoning_tags(full_response)

    with _Stage(f"new (fast path, {N}x, no think)") as s2:
        for _ in range(N):
            _strip_reasoning_tags(no_think)

    def old_strip(text):
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
        return text.strip()

    with _Stage(f"old (3x re.sub, {N}x, with think)") as s3:
        for _ in range(N):
            old_strip(full_response)

    print(f"  Old 3x re.sub (30KB, think):  {s3.elapsed_ms:.1f}ms  ({s3.elapsed_ms/N:.3f}ms each)")
    print(f"  New 1x compiled (30KB, think): {s1.elapsed_ms:.1f}ms  ({s1.elapsed_ms/N:.3f}ms each)")
    print(f"  New fast path (no think):       {s2.elapsed_ms:.1f}ms  ({s2.elapsed_ms/N:.4f}ms each)")
    print(f"  Speedup (think):   {s3.elapsed_ms / max(s1.elapsed_ms, 0.001):.1f}x")
    print(f"  Speedup (no-think): {s3.elapsed_ms / max(s2.elapsed_ms, 0.001):.0f}x")


# ── JSON extract benchmark ────────────────────────────────────────────────────

def benchmark_json_extract():
    _hr("_extract_outermost_json_object: fast-path check")
    from analysers.base import _extract_outermost_json_object

    prefix = "x" * 500
    json_body = '{"score": 0.85, "confidence": 90, "missing_skills": ["Python", "AWS"], "evidence": "' + "e" * 1000 + '"}'
    text_with_json = prefix + json_body
    text_no_json = "x" * 1500  # no { — hits find fast path

    N = 5000

    with _Stage(f"with JSON ({N}x)") as s1:
        for _ in range(N):
            _extract_outermost_json_object(text_with_json)
    with _Stage(f"no JSON ({N}x, find returns -1 fast)") as s2:
        for _ in range(N):
            _extract_outermost_json_object(text_no_json)

    print(f"  With JSON: {s1.elapsed_ms:.1f}ms  ({s1.elapsed_ms/N*1000:.2f}μs each)")
    print(f"  No JSON:   {s2.elapsed_ms:.1f}ms  ({s2.elapsed_ms/N*1000:.2f}μs each)")
    print("  NOTE: CPython slice iteration is faster than range(i) indexing —")
    print("        slice creates a compact char array; range + text[i] is slower.")


# ── DB JSON size benchmark ────────────────────────────────────────────────────

def benchmark_db_json_size():
    _hr("pipeline_json: raw vs slim + compact serialization")
    import json
    from db import _slim_pipeline

    # Simulate a pipeline result with 5 judge raw_response fields
    fake_pipeline = {
        "overall": {"overall_score": 0.78, "component_scores": {"skills": 0.8}},
    }
    for judge in ["skills", "experience", "seniority", "relevancy", "impact"]:
        fake_pipeline[judge] = {
            "analyser_result": {
                "success": True,
                "raw_response": "<think>" + "reasoning " * 3000 + "</think>\n{\"score\": 0.8}",
                "data": {"score": 0.8, "confidence": 85},
            },
            "scores": {"score": 0.8},
        }
    fake_pipeline["coach"] = {
        "success": True,
        "raw_response": "<think>" + "coach reasoning " * 2000 + "</think>\n{\"improvement_analysis\": []}",
        "data": {"improvement_analysis": []},
    }

    full_json_default    = json.dumps(fake_pipeline)
    full_json_compact    = json.dumps(fake_pipeline, separators=(',', ':'))
    slim_json_compact    = json.dumps(_slim_pipeline(fake_pipeline), separators=(',', ':'))

    print(f"  Full pipeline (default separators):   {len(full_json_default):>8,} bytes")
    print(f"  Full pipeline (compact separators):   {len(full_json_compact):>8,} bytes  ({(1 - len(full_json_compact)/len(full_json_default))*100:.1f}% smaller)")
    print(f"  Slim pipeline (no raw_response):      {len(slim_json_compact):>8,} bytes  ({(1 - len(slim_json_compact)/len(full_json_default))*100:.1f}% smaller)")


# ── Full cProfile pass (CPU) ──────────────────────────────────────────────────

def profile_cpu(top_n: int):
    _hr(f"cProfile: top {top_n} functions by cumulative time")
    from analysers.base import parse_and_validate, _strip_reasoning_tags, _extract_outermost_json_object
    from analysers.chunker_compat import parse_jd, parse_resume, format_for_skills_judge

    parse_jd.cache_clear()
    parse_resume.cache_clear()

    def workload():
        # Simulate the hot path: 6 judges each calling parse+format+strip+extract
        for _ in range(6):
            jd = parse_jd(JD)
            res = parse_resume(RESUME)
            format_for_skills_judge(jd, res)

        # Simulate LLM response processing
        fake_response = "<think>" + "reasoning " * 500 + "</think>\n" + \
            '{"score": 0.8, "confidence": 85, "missing_skills": ["Python"], "evidence": ["Led team"]}'
        for _ in range(10):
            stripped = _strip_reasoning_tags(fake_response)
            _extract_outermost_json_object(stripped)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        workload()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(top_n)
    output = s.getvalue()
    # Filter to just our code
    relevant = [l for l in output.splitlines()
                if any(kw in l for kw in ('chunker', 'base.py', 'prompts', 'analysers', 'ncalls', 'cumtime', '---'))]
    print('\n'.join(relevant[:top_n + 10]))


# ── tracemalloc (memory) ──────────────────────────────────────────────────────

def profile_memory(top_n: int):
    _hr(f"tracemalloc: top {top_n} allocation sites")
    from analysers.base import _strip_reasoning_tags, _extract_outermost_json_object
    from analysers.chunker_compat import parse_jd, parse_resume, format_for_skills_judge

    parse_jd.cache_clear()
    parse_resume.cache_clear()

    tracemalloc.start(10)  # 10 frame depth
    snap_before = tracemalloc.take_snapshot()

    # One simulated analysis pass
    for _ in range(6):
        jd = parse_jd(JD)
        res = parse_resume(RESUME)
        format_for_skills_judge(jd, res)

    fake_response = "<think>" + "reasoning " * 1000 + "</think>\n" + \
        '{"score": 0.8, "confidence": 85}'
    for _ in range(6):
        _strip_reasoning_tags(fake_response)
        _extract_outermost_json_object(fake_response)

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snap_after.compare_to(snap_before, 'lineno')
    print(f"  {'File:Line':<55} {'Size':>8}  {'Count':>7}")
    print(f"  {'-'*55} {'-'*8}  {'-'*7}")
    for stat in top_stats[:top_n]:
        loc = str(stat.traceback[0])
        size_kb = stat.size_diff / 1024
        print(f"  {loc:<55} {size_kb:>7.1f}KB  {stat.count_diff:>+7d}")

    total = sum(s.size_diff for s in top_stats)
    print(f"\n  Total heap delta: {total / 1024:.1f}KB")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20, help="Top N entries to show")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-mem", action="store_true")
    args = parser.parse_args()

    benchmark_parse_cache()
    benchmark_strip_reasoning()
    benchmark_json_extract()
    benchmark_db_json_size()

    if not args.skip_cpu:
        profile_cpu(args.top)

    if not args.skip_mem:
        profile_memory(args.top)

    print("\n")


if __name__ == "__main__":
    main()
