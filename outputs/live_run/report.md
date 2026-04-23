# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: live
- Records: 240
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.775 | 0.9167 | 0.1417 |
| Avg attempts | 1 | 1.3333 | 0.3333 |
| Avg token estimate | 322.08 | 515.58 | 193.5 |
| Avg latency (ms) | 4387.07 | 7087.21 | 2700.14 |

## Failure modes
```json
{
  "none": {
    "react": 93,
    "reflexion": 110
  },
  "verbosity": {
    "react": 14,
    "reflexion": 3
  },
  "entity_drift": {
    "react": 12,
    "reflexion": 7
  },
  "wrong_final_answer": {
    "react": 1
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json

## Discussion
Reflexion improves accuracy over ReAct by giving the agent a second chance after failure. The structured_evaluator uses an LLM judge to score answers and extract missing evidence, which feeds into the reflection_memory carried into the next attempt. The main cost is higher token usage and latency per question. Remaining failure modes include cases where llama-3.1-8b-instant lacks world knowledge or produces incomplete multi-hop reasoning even after reflection. Memory compression would help when reflection chains grow long across many attempts.
