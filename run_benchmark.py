from __future__ import annotations
import json
from pathlib import Path
import typer
from dotenv import load_dotenv
load_dotenv()
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    max_samples: int = 0,
) -> None:
    examples = load_dataset(dataset)
    if max_samples and max_samples < len(examples):
        examples = examples[:max_samples]
        print(f"[yellow]Running on {max_samples}/{len(load_dataset(dataset))} samples[/yellow]")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    print(f"[cyan]ReAct: running {len(examples)} samples...[/cyan]")
    react_records = []
    for i, ex in enumerate(examples, 1):
        rec = react.run(ex)
        react_records.append(rec)
        print(f"  ReAct {i}/{len(examples)} | correct={rec.is_correct} | {ex.qid}")

    print(f"[cyan]Reflexion: running {len(examples)} samples...[/cyan]")
    reflexion_records = []
    for i, ex in enumerate(examples, 1):
        rec = reflexion.run(ex)
        reflexion_records.append(rec)
        print(f"  Reflexion {i}/{len(examples)} | correct={rec.is_correct} | attempts={rec.attempts} | {ex.qid}")
    all_records = react_records + reflexion_records

    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    report = build_report(all_records, dataset_name=Path(dataset).name, mode="live")
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
