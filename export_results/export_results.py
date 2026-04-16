import json
import shutil
from pathlib import Path
import yaml

from huggingface_hub import HfApi

HF_REPO = "AINovice2005/pixel-art-bench"
REPO_TYPE = "dataset"

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
EVAL_DIR = ROOT_DIR / "eval_results"

CREATE_PR = False


def extract_metrics(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    raw_metrics = data.get("metrics", {})

    def get_value(metric):
        if isinstance(metric, dict):
            for v in metric.values():
                if isinstance(v, (int, float)):
                    return v
            return 0.0
        elif isinstance(metric, (int, float)):
            return metric
        return 0.0

    return {
        "json_validity": get_value(raw_metrics.get("json_validity", 0)),
        "render_success": get_value(raw_metrics.get("render_success", 0)),
        "pixel_art_quality": get_value(raw_metrics.get("pixel_art_quality", 0)),
    }


def build_eval_results():
    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"Missing {OUTPUTS_DIR}")

    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    leaderboard = []

    for model_dir in OUTPUTS_DIR.iterdir():
        yaml_file = model_dir / "pixel-art-bench-detailed.yaml"
        if not yaml_file.exists():
            continue

        model_name = model_dir.name

        target_dir = EVAL_DIR / model_name
        target_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(yaml_file, target_dir / "detailed.yaml")

        metrics = extract_metrics(yaml_file)

        entry = {"model": model_name, **metrics}
        values = list(metrics.values())
        entry["overall"] = sum(values) / len(values) if values else 0

        leaderboard.append(entry)

    leaderboard.sort(key=lambda x: x["overall"], reverse=True)

    with open(EVAL_DIR / "leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2)


def push_to_hub():
    api = HfApi()

    api.upload_folder(
        folder_path=str(EVAL_DIR),
        repo_id=HF_REPO,
        repo_type=REPO_TYPE,
        path_in_repo="eval_results",
        commit_message="Add evaluation results",
        create_pr=CREATE_PR,
    )


if __name__ == "__main__":
    build_eval_results()
    push_to_hub()