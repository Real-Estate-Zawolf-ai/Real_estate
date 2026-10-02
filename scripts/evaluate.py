"""Validate the evaluation dataset and report basic retrieval fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "data" / "evaluation"


def load_json(name: str) -> list[dict[str, Any]]:
    path = EVALUATION_DIR / name
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return value


def validate_dataset() -> dict[str, int]:
    questions = load_json("questions.json")
    golden = load_json("golden_dataset.json")
    expected = load_json("expected_recommendations.json")
    question_ids = {item.get("id") for item in questions}

    for collection_name, collection in (("golden", golden), ("expected", expected)):
        unknown = {item.get("question_id") for item in collection} - question_ids
        if unknown:
            raise ValueError(f"{collection_name} contains unknown question IDs: {sorted(unknown)}")

    return {"questions": len(questions), "golden": len(golden), "expected": len(expected)}


def main() -> int:
    counts = validate_dataset()
    print("Evaluation dataset valid: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
