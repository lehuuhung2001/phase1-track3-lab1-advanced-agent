from __future__ import annotations
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {
    "hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes",
    "hp10": "Tanzania", "hp12": "Amsterdam", "hp14": "Welsh", "hp16": "Mount Everest",
    "hp18": "Franc", "hp20": "South America", "hp22": "Kenya", "hp24": "Australian",
    "hp26": "Thames", "hp28": "Slavic", "hp30": "Munich", "hp32": "Caspian Sea",
    "hp34": "Zimbabwe", "hp36": "Portuguese", "hp38": "Mediterranean Sea", "hp40": "Brazil",
    "hp42": "Indian Ocean", "hp44": "Mediterranean Sea", "hp46": "Florence",
    "hp48": "Alexandria", "hp50": "Mediterranean Sea",
    "hp52": "Nepal", "hp54": "Oslo", "hp56": "French", "hp58": "Hellenic",
    "hp60": "Stockholm", "hp62": "Adriatic Sea", "hp64": "Australia", "hp66": "London",
    "hp68": "Indian Ocean", "hp70": "Japanese", "hp72": "Beijing", "hp74": "Adriatic Sea",
    "hp76": "Benin", "hp78": "Ho Chi Minh City", "hp80": "North Sea", "hp82": "Romance",
    "hp84": "Paris", "hp86": "Pacific Ocean", "hp88": "Venice", "hp90": "Prague",
    "hp92": "Mediterranean Sea", "hp94": "Romance", "hp96": "Baghdad", "hp98": "Asia",
    "hp100": "Senegal", "hp102": "Saint Petersburg", "hp104": "Atlantic Ocean", "hp106": "Semitic",
    "hp108": "Florence", "hp110": "Atlantic Ocean", "hp112": "Guinea", "hp114": "Shanghai",
    "hp116": "Atlantic Ocean", "hp118": "Semitic", "hp120": "Adriatic Sea",
}
FAILURE_MODE_BY_QID = {
    "hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift",
    "hp10": "incomplete_multi_hop", "hp12": "wrong_final_answer", "hp14": "entity_drift", "hp16": "incomplete_multi_hop",
    "hp18": "wrong_final_answer", "hp20": "entity_drift", "hp22": "incomplete_multi_hop", "hp24": "wrong_final_answer",
    "hp26": "entity_drift", "hp28": "incomplete_multi_hop", "hp30": "wrong_final_answer", "hp32": "entity_drift",
    "hp34": "incomplete_multi_hop", "hp36": "wrong_final_answer", "hp38": "entity_drift", "hp40": "incomplete_multi_hop",
    "hp42": "wrong_final_answer", "hp44": "entity_drift", "hp46": "incomplete_multi_hop",
    "hp48": "wrong_final_answer", "hp50": "entity_drift",
    "hp52": "incomplete_multi_hop", "hp54": "wrong_final_answer", "hp56": "entity_drift", "hp58": "incomplete_multi_hop",
    "hp60": "wrong_final_answer", "hp62": "entity_drift", "hp64": "incomplete_multi_hop", "hp66": "wrong_final_answer",
    "hp68": "entity_drift", "hp70": "incomplete_multi_hop", "hp72": "wrong_final_answer", "hp74": "entity_drift",
    "hp76": "incomplete_multi_hop", "hp78": "wrong_final_answer", "hp80": "entity_drift", "hp82": "incomplete_multi_hop",
    "hp84": "wrong_final_answer", "hp86": "entity_drift", "hp88": "incomplete_multi_hop", "hp90": "wrong_final_answer",
    "hp92": "entity_drift", "hp94": "incomplete_multi_hop", "hp96": "wrong_final_answer", "hp98": "entity_drift",
    "hp100": "incomplete_multi_hop", "hp102": "wrong_final_answer", "hp104": "entity_drift", "hp106": "incomplete_multi_hop",
    "hp108": "wrong_final_answer", "hp110": "entity_drift", "hp112": "incomplete_multi_hop", "hp114": "wrong_final_answer",
    "hp116": "entity_drift", "hp118": "incomplete_multi_hop", "hp120": "entity_drift",
}

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
