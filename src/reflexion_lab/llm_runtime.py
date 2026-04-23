from __future__ import annotations
import json
import os
import re
import time
from groq import Groq, RateLimitError
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        _client = Groq(api_key=api_key)
    return _client


_last_call: float = 0.0
_MIN_INTERVAL = 2.2  # stay under 30 RPM free tier


def _generate(system: str, user: str, max_tokens: int = 128) -> tuple[str, int]:
    """Returns (text, total_tokens). Throttles to stay under 30 RPM."""
    global _last_call
    client = _get_client()
    for attempt in range(8):
        # enforce minimum gap between requests
        elapsed = time.perf_counter() - _last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        try:
            _last_call = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = response.choices[0].message.content or ""
            total_tokens = response.usage.total_tokens if response.usage else 0
            return text.strip(), total_tokens
        except RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"[rate limit] waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Groq rate limit exceeded after retries")


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int]:
    context_text = "\n\n".join(f"[{c.title}]\n{c.text}" for c in example.context)
    memory_hint = ""
    if reflection_memory:
        hints = "\n".join(f"  - {s}" for s in reflection_memory)
        memory_hint = f"\n\nHints from previous failed attempts:\n{hints}"
    user_msg = (
        f"Context:\n{context_text}\n\n"
        f"Question: {example.question}{memory_hint}\n\n"
        "Answer (concise, a few words only):"
    )
    text, tokens = _generate(ACTOR_SYSTEM, user_msg, max_tokens=64)
    answer = text.split("\n")[0].strip()
    for prefix in ("Answer:", "answer:", "The answer is", "the answer is"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip().lstrip(":").strip()
    return answer or text[:80], tokens


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int]:
    user_msg = (
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}\n\n"
        "Return JSON only."
    )
    text, tokens = _generate(EVALUATOR_SYSTEM, user_msg, max_tokens=128)
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return JudgeResult(
                score=int(bool(data.get("score", 0))),
                reason=str(data.get("reason", "")),
                missing_evidence=list(data.get("missing_evidence", [])),
                spurious_claims=list(data.get("spurious_claims", [])),
            ), tokens
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Answer matches gold answer."), tokens
    return JudgeResult(
        score=0,
        reason=f"Predicted '{answer}' did not match gold '{example.gold_answer}'.",
        spurious_claims=[answer],
    ), tokens


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int]:
    user_msg = (
        f"Question: {example.question}\n"
        f"Failure reason: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}\n\n"
        "Return JSON only."
    )
    text, tokens = _generate(REFLECTOR_SYSTEM, user_msg, max_tokens=128)
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=str(data.get("failure_reason", judge.reason)),
                lesson=str(data.get("lesson", "")),
                next_strategy=str(data.get("next_strategy", "Re-read context carefully.")),
            ), tokens
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="Previous strategy was insufficient.",
        next_strategy=text[:200] if text else "Focus on completing all hops in the reasoning chain.",
    ), tokens
