import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from datasets import Dataset, Features, Sequence, Value


# =========================
# CONFIG
# =========================
INPUT_PATH = Path("/workspaces/pixel-art-bench/data/pixel_art_raw_responses.json")

REPO_ID = "AINovice2005/pixel-art-bench-v1"
PRIVATE = False

SAVE_LOCAL_COPY = False
LOCAL_SAVE_PATH = "pixel_art_dataset"


# =========================
# SAFE CASTING UTILITIES
# =========================
def safe_list_of_str(x):
    if not isinstance(x, list):
        return None
    if not all(isinstance(i, str) for i in x):
        return None
    return x


def safe_bool(x):
    if isinstance(x, bool):
        return x
    return None


def safe_int(x):
    try:
        return int(x) if x is not None else None
    except:
        return None


def safe_float(x):
    try:
        return float(x) if x is not None else None
    except:
        return None


# =========================
# JSON EXTRACTION
# =========================
def extract_inner_json(content: str) -> Dict[str, Any]:
    if not content:
        return {}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


# =========================
# RECORD PROCESSING
# =========================
def process_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = record.get("raw_response", {})
    choices = raw.get("choices", [])

    if not choices:
        return None

    message = choices[0].get("message", {})
    content = message.get("content", "")

    parsed = extract_inner_json(content)

    palette = safe_list_of_str(parsed.get("palette"))
    grid = safe_list_of_str(parsed.get("grid"))

    # Hard filter: must be valid lists of strings
    if palette is None or grid is None:
        return None

    usage = raw.get("usage", {})

    return {
        # identity
        "id": safe_int(record.get("id")),
        "model_slug": str(record.get("model_slug") or ""),
        "model_name": str(record.get("model_name") or ""),
        "example_id": safe_int(record.get("example_id")),
        "example_name": str(record.get("example_name") or ""),

        # pixel data
        "palette": palette,
        "grid": grid,
        "is_appropriate": safe_bool(parsed.get("is_appropriate")),

        # derived
        "height": len(grid),
        "width": len(grid[0]) if grid else 0,
        "num_colors": len(palette),

        # metrics
        "input_tokens": safe_int(record.get("input_tokens")),
        "output_tokens": safe_int(record.get("output_tokens")),
        "total_tokens": safe_int(usage.get("total_tokens")),
        "cost": safe_float(record.get("cost")),
        "generation_time": safe_float(record.get("generation_duration_seconds")),
    }


# =========================
# FINAL VALIDATION
# =========================
def validate_row(row: Dict[str, Any]) -> bool:
    return (
        isinstance(row["palette"], list)
        and isinstance(row["grid"], list)
        and all(isinstance(x, str) for x in row["palette"])
        and all(isinstance(x, str) for x in row["grid"])
    )


# =========================
# MAIN PIPELINE
# =========================
def main():
    print(f"Loading raw data from: {INPUT_PATH}")

    with open(INPUT_PATH, "r") as f:
        raw_data = json.load(f)

    print(f"Total raw records: {len(raw_data)}")

    processed: List[Dict[str, Any]] = []
    skipped = 0

    for record in raw_data:
        item = process_record(record)
        if item is None:
            skipped += 1
            continue
        processed.append(item)

    print(f"After processing → {len(processed)} valid candidates")
    print(f"Skipped during parsing → {skipped}")

    # =========================
    # FINAL SANITY FILTER
    # =========================
    before = len(processed)
    processed = [r for r in processed if validate_row(r)]
    after = len(processed)

    print(f"After validation → {after} rows")
    print(f"Filtered invalid rows → {before - after}")

    if not processed:
        raise RuntimeError("No valid samples remain after validation.")

    # =========================
    # EXPLICIT SCHEMA (CRITICAL)
    # =========================
    features = Features({
        "id": Value("int64"),
        "model_slug": Value("string"),
        "model_name": Value("string"),
        "example_id": Value("int64"),
        "example_name": Value("string"),

        "palette": Sequence(Value("string")),
        "grid": Sequence(Value("string")),
        "is_appropriate": Value("bool"),

        "height": Value("int64"),
        "width": Value("int64"),
        "num_colors": Value("int64"),

        "input_tokens": Value("int64"),
        "output_tokens": Value("int64"),
        "total_tokens": Value("int64"),

        "cost": Value("float64"),
        "generation_time": Value("float64"),
    })

    # =========================
    # BUILD DATASET
    # =========================
    print("Building HF Dataset...")
    dataset = Dataset.from_list(processed, features=features)

    print(dataset)
    print("Sample row:")
    print(dataset[0])

    # =========================
    # OPTIONAL LOCAL SAVE
    # =========================
    if SAVE_LOCAL_COPY:
        dataset.save_to_disk(LOCAL_SAVE_PATH)
        print(f"Saved locally → {LOCAL_SAVE_PATH}")

    # =========================
    # PUSH TO HUB
    # =========================
    print(f"Pushing to HF Hub → {REPO_ID}")

    dataset.push_to_hub(
        REPO_ID,
        private=PRIVATE,
    )

    print("Upload complete.")


if __name__ == "__main__":
    main()