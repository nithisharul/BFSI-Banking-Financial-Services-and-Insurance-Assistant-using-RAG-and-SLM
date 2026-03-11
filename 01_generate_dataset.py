"""Generate or validate instruction-following dataset.

This script loads `dataset/bfsi_alpaca.json` and validates the required fields.
"""

import json
from pathlib import Path


def validate_dataset(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Dataset must be a list of examples."

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not an object")
        for key in ("instruction", "input", "output"):
            if key not in item:
                raise ValueError(f"Item {idx} missing required key: {key}")
    print(f"Validated {len(data)} examples in {path}")
    return True


if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "dataset" / "bfsi_alpaca.json"
    validate_dataset(dataset_path)
