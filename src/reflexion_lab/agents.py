from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Literal
from .llm_runtime import actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer


def _classify_failure(score: int, predicted: str, gold: str) -> str:
    if score == 1:
        return "none"
    pred = normalize_answer(predicted)
    gld = normalize_answer(gold)
    if pred and gld and (gld in pred or pred in gld):
        return "verbosity"
    pred_words = set(pred.split())
    gld_words = set(gld.split())
    if pred_words and gld_words and not pred_words & gld_words:
        return "entity_drift"
    return "wrong_final_answer"


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            answer, actor_tokens = actor_answer(
                example, attempt_id, self.agent_type, reflection_memory
            )
            judge, eval_tokens = evaluator(example, answer)
            attempt_tokens = actor_tokens + eval_tokens
            attempt_latency_ms = int((time.perf_counter() - t0) * 1000)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=attempt_tokens,
                latency_ms=attempt_latency_ms,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                t1 = time.perf_counter()
                reflection, refl_tokens = reflector(example, attempt_id, judge)
                trace.token_estimate += refl_tokens
                trace.latency_ms += int((time.perf_counter() - t1) * 1000)
                reflection_memory.append(reflection.next_strategy)
                reflections.append(reflection)
                trace.reflection = reflection

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=_classify_failure(final_score, final_answer, example.gold_answer),
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
