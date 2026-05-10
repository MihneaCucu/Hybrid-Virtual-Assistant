from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.qa_module import load_qa_system
from scripts.hybrid_demo_cli import route_request


DEFAULT_SCENARIOS = Path("data/integration_acceptance_scenarios.jsonl")


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("query"):
                raise ValueError(f"Invalid scenario at {path}:{line_no}")
            scenarios.append(row)
    return scenarios


def _contains_all(answer: object, expected_fragments: list[str]) -> bool:
    answer_text = str(answer or "").lower()
    return all(fragment.lower() in answer_text for fragment in expected_fragments)


def _check_scenario(row: dict[str, Any], result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_fields = {
        "expected_route": "route",
        "expected_status": "status",
        "expected_reason_code": "reason_code",
        "expected_source_doc": "source_doc",
        "expected_intent": "intent",
    }
    for expected_key, result_key in expected_fields.items():
        if expected_key not in row:
            continue
        expected = row.get(expected_key)
        got = result.get(result_key)
        if got != expected:
            failures.append(f"{result_key}: expected={expected!r} got={got!r}")

    expected_answer_fragments = row.get("answer_contains") or []
    if expected_answer_fragments and not _contains_all(result.get("answer"), expected_answer_fragments):
        failures.append(f"answer missing fragments: {expected_answer_fragments!r}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final hybrid assistant integration acceptance scenarios."
    )
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS),
        help=f"JSONL scenario path (default: {DEFAULT_SCENARIOS})",
    )
    parser.add_argument("--show-meta", action="store_true", help="Print route/status/source metadata.")
    args = parser.parse_args()

    scenario_path = Path(args.scenarios)
    scenarios = _load_scenarios(scenario_path)
    if not scenarios:
        raise SystemExit(f"No scenarios loaded from: {scenario_path}")

    load_qa_system()

    passed = 0
    print(f"Running {len(scenarios)} integration acceptance scenarios from {scenario_path}\n")
    for idx, row in enumerate(scenarios, start=1):
        query = str(row["query"])
        result = route_request(query)
        failures = _check_scenario(row, result)
        passed += int(not failures)

        print(f"[{idx:02d}] {row.get('id', f'int_{idx:03d}')}")
        print(f"User: {query}")
        print(f"Assistant: {result.get('answer')}")
        if args.show_meta:
            print(
                "[meta] "
                f"route={result.get('route')} "
                f"status={result.get('status')} "
                f"reason={result.get('reason_code') or result.get('intent') or '-'} "
                f"source={result.get('source_doc') or '-'} "
                f"confidence={result.get('confidence')}"
            )
        if row.get("story_point"):
            print(f"Story point: {row['story_point']}")
        if failures:
            print("Result: FAIL")
            for failure in failures:
                print(f"- {failure}")
        else:
            print("Result: PASS")
        print()

    print(f"Integration acceptance: {passed}/{len(scenarios)} = {passed / len(scenarios):.3f}")
    if passed != len(scenarios):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
