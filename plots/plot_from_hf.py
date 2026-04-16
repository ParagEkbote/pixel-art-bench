import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from huggingface_hub import hf_hub_download

REPO_ID = "AINovice2005/pixel-art-bench"
FILE_PATH = "eval_results/leaderboard.json"

OUT = Path("benchmark_plots")
OUT.mkdir(exist_ok=True)


def load_data():
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILE_PATH,
        repo_type="dataset"
    )
    with open(file_path) as f:
        return json.load(f)


def plot_all(data):
    data = sorted(data, key=lambda x: x["overall"], reverse=True)

    models = [d["model"] for d in data]

    # 1. Leaderboard
    plt.figure()
    plt.bar(models, [d["overall"] for d in data])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT / "leaderboard.png")
    plt.close()

    # 2. Breakdown
    metrics = ["json_validity", "render_success", "pixel_art_quality"]
    x = np.arange(len(data))
    width = 0.25

    plt.figure()
    for i, m in enumerate(metrics):
        plt.bar(x + i * width, [d[m] for d in data], width)

    plt.xticks(x + width, models, rotation=45)
    plt.tight_layout()
    plt.savefig(OUT / "breakdown.png")
    plt.close()

    # 3. Tradeoff
    plt.figure()
    plt.scatter(
        [d["json_validity"] for d in data],
        [d["pixel_art_quality"] for d in data]
    )

    for d in data:
        plt.annotate(d["model"], (d["json_validity"], d["pixel_art_quality"]))

    plt.tight_layout()
    plt.savefig(OUT / "tradeoff.png")
    plt.close()


if __name__ == "__main__":
    plot_all(load_data())