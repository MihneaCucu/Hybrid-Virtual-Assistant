from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COMMANDS = [
    [sys.executable, "-m", "py_compile", "qa/qa_module.py", "scripts/evaluate_qa.py", "scripts/evaluate_qa_blind.py", "scripts/compare_qa_baselines.py", "scripts/compare_qa_ablation.py", "scripts/demo_qa.py", "scripts/generate_qa_story_assets.py", "scripts/hybrid_demo_cli.py"],
    [sys.executable, "-m", "pytest"],
    [sys.executable, "scripts/evaluate_qa.py"],
    [sys.executable, "scripts/evaluate_qa_blind.py"],
    [sys.executable, "scripts/compare_qa_baselines.py"],
    [sys.executable, "scripts/compare_qa_ablation.py"],
    [sys.executable, "scripts/generate_qa_story_assets.py"],
    [sys.executable, "scripts/demo_qa.py", "--queries", "data/demo_queries_story.jsonl", "--show-sources"],
    [sys.executable, "scripts/hybrid_demo_cli.py", "--story", "--show-meta"],
]


def _display_command(command: list[str]) -> str:
    return " ".join(command)


def main() -> None:
    for idx, command in enumerate(COMMANDS, start=1):
        print(f"\n[{idx}/{len(COMMANDS)}] {_display_command(command)}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    print("\nKnowledge/Q&A story check completed successfully.", flush=True)
    print("Key outputs:", flush=True)
    print("- results/qa_eval_summary.json", flush=True)
    print("- results/qa_blind_eval_summary.json", flush=True)
    print("- results/qa_baseline_comparison.json", flush=True)
    print("- results/qa_ablation_comparison.json", flush=True)
    print("- docs/qa_story_assets.md", flush=True)


if __name__ == "__main__":
    main()
