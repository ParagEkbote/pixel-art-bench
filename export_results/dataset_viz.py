import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, load_from_disk

# =========================
# CONFIG
# =========================
DATASET_NAME = "AINovice2005/pixel-art-bench-v1"   # OR None if local
LOCAL_PATH = None  # e.g. "pixel_art_dataset"

OUTPUT_DIR = Path("./eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style for high-quality plots
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


# =========================
# LOAD DATASET
# =========================
def load_data():
    if LOCAL_PATH:
        ds = load_from_disk(LOCAL_PATH)
    else:
        ds = load_dataset(DATASET_NAME, split="train")
    return ds


# =========================
# CORE METRICS
# =========================
def grid_entropy(grid):
    """Shannon entropy of color distribution in grid."""
    flat = "".join(grid)
    counts = Counter(flat)
    probs = np.array(list(counts.values())) / len(flat)
    return float(-(probs * np.log2(probs)).sum())


def color_diversity(grid, palette_size):
    """Fraction of palette colors actually used in grid."""
    flat = "".join(grid)
    unique_colors = len(set(flat))
    return unique_colors / max(1, palette_size)


def symmetry_score(grid):
    """Returns horizontal and vertical symmetry ratios (0-1)."""
    h = sum(row == row[::-1] for row in grid) / len(grid)

    v_matches = 0
    for i in range(len(grid) // 2):
        if grid[i] == grid[-i-1]:
            v_matches += 1
    v = v_matches / max(1, len(grid) // 2)

    return h, v


def sparsity_ratio(grid):
    """Fraction of cells that match dominant color."""
    flat = "".join(grid)
    counts = Counter(flat)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(flat)


def validate_grid(grid):
    if not grid or not isinstance(grid, list):
        return False
    w = len(grid[0])
    return all(isinstance(r, str) and len(r) == w for r in grid)


# =========================
# FINE-TUNING ELIGIBILITY ASSESSMENT
# =========================
class FineTuningEligibility:
    """Assess dataset suitability for fine-tuning vs. zero-shot inference."""

    def __init__(self):
        self.scores = {}

    def assess(self, stats):
        """Compute fine-tuning eligibility score (0-100)."""
        eligibility = {}

        # 1. Sequence length: shorter is better for fine-tuning
        seq_len_p90 = stats["seq_len_stats"]["p90"]
        seq_len_score = max(0, 100 - (seq_len_p90 / 10))  # Penalty for long sequences
        eligibility["sequence_length"] = {
            "value": seq_len_p90,
            "score": seq_len_score,
            "rationale": f"P90 sequence length: {seq_len_p90:.0f}. Fine-tuning prefers < 1000 tokens."
        }

        # 2. Dataset size: need sufficient samples
        num_valid = stats["num_samples"] - stats["invalid_samples"]
        size_score = min(100, (num_valid / 500) * 100)  # 500+ samples = 100
        eligibility["dataset_size"] = {
            "value": num_valid,
            "score": size_score,
            "rationale": f"Valid samples: {num_valid}. Fine-tuning needs ≥500 for meaningful signal."
        }

        # 3. Data quality: validity ratio
        valid_ratio = stats["valid_ratio"]
        quality_score = valid_ratio * 100
        eligibility["data_quality"] = {
            "value": valid_ratio,
            "score": quality_score,
            "rationale": f"Valid ratio: {valid_ratio:.2%}. Threshold: ≥95%."
        }

        # 4. Pattern complexity: entropy (moderate is best)
        entropy_mean = stats["entropy_stats"]["mean"]
        # Moderate entropy (2-3) is ideal; too low or too high is problematic
        if entropy_mean < 1.5:
            entropy_score = 50  # Too simple, may not benefit from fine-tuning
        elif entropy_mean < 3.5:
            entropy_score = 100  # Sweet spot
        else:
            entropy_score = 70  # Complex, but feasible
        eligibility["pattern_complexity"] = {
            "value": entropy_mean,
            "score": entropy_score,
            "rationale": f"Entropy: {entropy_mean:.2f}. Optimal range: 1.5–3.5 for fine-tuning."
        }

        # 5. Color diversity: higher is better (more generalization)
        color_div_mean = stats.get("color_diversity_mean", 0.5)
        diversity_score = color_div_mean * 100
        eligibility["color_diversity"] = {
            "value": color_div_mean,
            "score": diversity_score,
            "rationale": f"Avg color utilization: {color_div_mean:.2%}. Higher is better for generalization."
        }

        # 6. Sparsity: too sparse or too dense is harder
        sparsity_mean = stats.get("sparsity_mean", 0.5)
        if sparsity_mean < 0.3 or sparsity_mean > 0.8:
            sparsity_score = 60
        else:
            sparsity_score = 100
        eligibility["sparsity"] = {
            "value": sparsity_mean,
            "score": sparsity_score,
            "rationale": f"Sparsity: {sparsity_mean:.2%}. Moderate (0.3–0.8) is ideal."
        }

        # Overall score
        weights = {
            "sequence_length": 0.20,
            "dataset_size": 0.25,
            "data_quality": 0.20,
            "pattern_complexity": 0.15,
            "color_diversity": 0.10,
            "sparsity": 0.10,
        }
        overall = sum(eligibility[k]["score"] * weights[k] for k in weights)
        eligibility["overall_score"] = overall

        self.scores = eligibility
        return eligibility

    def recommend(self, eligibility):
        """Return recommendation based on eligibility score."""
        overall = eligibility["overall_score"]

        if overall >= 75:
            return {
                "tier": "ELIGIBLE FOR FINE-TUNING",
                "confidence": "High",
                "approach": "Direct fine-tuning (LoRA/QLoRA on LM models)",
                "models": ["Qwen2.5-1B", "SmolLM2-1.7B", "Llama-3.2-1B-Instruct"],
                "note": "Dataset is well-structured. Use supervised fine-tuning with standard LM objectives."
            }
        elif overall >= 55:
            return {
                "tier": "CONDITIONAL ELIGIBILITY",
                "confidence": "Medium",
                "approach": "Hybrid: fine-tune + in-context examples",
                "models": ["Qwen2.5-3B-Instruct", "Meta-Llama-3-8B-Instruct"],
                "note": "Dataset is usable but has quality/complexity issues. Augment with prompt engineering."
            }
        else:
            return {
                "tier": "USE ZERO-SHOT / PROMPTING",
                "confidence": "Low",
                "approach": "Prompting only (no fine-tuning)",
                "models": ["Qwen2.5-3B-Instruct", "GPT-4o-mini"],
                "note": "Dataset has significant quality or complexity issues. Fine-tuning unlikely to help."
            }


# =========================
# MAIN EDA
# =========================
def run_eda(ds):
    """Compute comprehensive statistics on pixel art dataset."""
    stats = defaultdict(list)
    invalid = 0

    for x in ds:
        grid = x["grid"]
        palette = x["palette"]

        if not validate_grid(grid):
            invalid += 1
            continue

        h = len(grid)
        w = len(grid[0])
        stats["height"].append(h)
        stats["width"].append(w)

        stats["palette_size"].append(len(palette))

        # Sequence length: grid cells + palette
        seq_len = sum(len(r) for r in grid) + len(palette)
        stats["seq_len"].append(seq_len)

        # Entropy
        stats["entropy"].append(grid_entropy(grid))

        # Symmetry
        h_sym, v_sym = symmetry_score(grid)
        stats["h_sym"].append(h_sym)
        stats["v_sym"].append(v_sym)

        # Color diversity
        stats["color_diversity"].append(color_diversity(grid, len(palette)))

        # Sparsity
        stats["sparsity"].append(sparsity_ratio(grid))

    results = {
        "num_samples": len(ds),
        "invalid_samples": invalid,
        "valid_ratio": 1 - (invalid / len(ds)) if len(ds) > 0 else 0,

        "height_dist": dict(Counter(stats["height"])),
        "width_dist": dict(Counter(stats["width"])),

        "palette_stats": summarize(stats["palette_size"]),
        "seq_len_stats": summarize(stats["seq_len"]),
        "entropy_stats": summarize(stats["entropy"]),
        "h_sym_stats": summarize(stats["h_sym"]),
        "v_sym_stats": summarize(stats["v_sym"]),

        "color_diversity_mean": float(np.mean(stats["color_diversity"])),
        "sparsity_mean": float(np.mean(stats["sparsity"])),

        # Raw arrays for plotting
        "_seq_len": stats["seq_len"],
        "_entropy": stats["entropy"],
        "_h_sym": stats["h_sym"],
        "_v_sym": stats["v_sym"],
        "_color_diversity": stats["color_diversity"],
        "_sparsity": stats["sparsity"],
    }

    return results


def summarize(arr):
    """Compute summary statistics."""
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


# =========================
# VISUALIZATION
# =========================
def plot_sequence_length_distribution(stats):
    """Distribution of sequence lengths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    seq_len = stats["_seq_len"]

    # Histogram
    axes[0].hist(seq_len, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(stats["seq_len_stats"]["mean"], color="red", linestyle="--", label=f"Mean: {stats['seq_len_stats']['mean']:.0f}")
    axes[0].axvline(stats["seq_len_stats"]["p90"], color="orange", linestyle="--", label=f"P90: {stats['seq_len_stats']['p90']:.0f}")
    axes[0].set_xlabel("Sequence Length (tokens)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sequence Length Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot(seq_len, vert=True, patch_artist=True)
    axes[1].set_ylabel("Sequence Length (tokens)")
    axes[1].set_title("Sequence Length Box Plot")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_sequence_length.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_entropy_distribution(stats):
    """Distribution of pattern entropy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    entropy = stats["_entropy"]

    # Histogram
    axes[0].hist(entropy, bins=30, color="seagreen", edgecolor="black", alpha=0.7)
    axes[0].axvline(stats["entropy_stats"]["mean"], color="red", linestyle="--", label=f"Mean: {stats['entropy_stats']['mean']:.2f}")
    axes[0].axvspan(1.5, 3.5, alpha=0.1, color="green", label="Fine-tuning Sweet Spot")
    axes[0].set_xlabel("Entropy (bits)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Pattern Entropy Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative
    sorted_ent = np.sort(entropy)
    axes[1].plot(sorted_ent, np.arange(1, len(sorted_ent) + 1) / len(sorted_ent) * 100, linewidth=2, color="seagreen")
    axes[1].set_xlabel("Entropy (bits)")
    axes[1].set_ylabel("Cumulative %")
    axes[1].set_title("Entropy Cumulative Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_entropy_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_symmetry_analysis(stats):
    """Horizontal and vertical symmetry analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    h_sym = stats["_h_sym"]
    v_sym = stats["_v_sym"]

    # Horizontal
    axes[0].hist(h_sym, bins=30, color="coral", edgecolor="black", alpha=0.7)
    axes[0].axvline(np.mean(h_sym), color="red", linestyle="--", label=f"Mean: {np.mean(h_sym):.2f}")
    axes[0].set_xlabel("Horizontal Symmetry Ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Horizontal Symmetry Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Vertical
    axes[1].hist(v_sym, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axes[1].axvline(np.mean(v_sym), color="red", linestyle="--", label=f"Mean: {np.mean(v_sym):.2f}")
    axes[1].set_xlabel("Vertical Symmetry Ratio")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Vertical Symmetry Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_symmetry_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_color_and_sparsity(stats):
    """Color diversity and sparsity analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    color_div = stats["_color_diversity"]
    sparsity = stats["_sparsity"]

    # Color diversity
    axes[0].hist(color_div, bins=30, color="mediumpurple", edgecolor="black", alpha=0.7)
    axes[0].axvline(np.mean(color_div), color="red", linestyle="--", label=f"Mean: {np.mean(color_div):.2f}")
    axes[0].set_xlabel("Color Utilization Ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Color Diversity Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sparsity
    axes[1].hist(sparsity, bins=30, color="lightsalmon", edgecolor="black", alpha=0.7)
    axes[1].axvline(np.mean(sparsity), color="red", linestyle="--", label=f"Mean: {np.mean(sparsity):.2f}")
    axes[1].axvspan(0.3, 0.8, alpha=0.1, color="green", label="Ideal Range")
    axes[1].set_xlabel("Sparsity Ratio")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sparsity Distribution (Dominant Color Freq)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_color_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_eligibility_scorecard(eligibility):
    """Radar/scorecard visualization of fine-tuning eligibility."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract scores
    categories = []
    scores = []
    for key in ["sequence_length", "dataset_size", "data_quality", "pattern_complexity", "color_diversity", "sparsity"]:
        if key in eligibility:
            categories.append(key.replace("_", "\n").title())
            scores.append(eligibility[key]["score"])

    # Horizontal bar chart
    colors = ["green" if s >= 75 else "orange" if s >= 55 else "red" for s in scores]
    y_pos = np.arange(len(categories))

    bars = ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 2, bar.get_y() + bar.get_height() / 2, f"{score:.0f}", va="center", fontweight="bold")

    # Threshold lines
    ax.axvline(75, color="green", linestyle="--", linewidth=2, alpha=0.5, label="High (75+)")
    ax.axvline(55, color="orange", linestyle="--", linewidth=2, alpha=0.5, label="Medium (55–74)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Eligibility Score")
    ax.set_title(f"Fine-Tuning Eligibility Scorecard\nOverall: {eligibility['overall_score']:.0f}/100", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_eligibility_scorecard.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_grid_size_distribution(stats):
    """Grid height and width distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    heights = list(stats["height_dist"].keys())
    h_counts = list(stats["height_dist"].values())

    widths = list(stats["width_dist"].keys())
    w_counts = list(stats["width_dist"].values())

    axes[0].bar(heights, h_counts, color="teal", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Height (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Grid Height Distribution")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(widths, w_counts, color="darkorange", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Width (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Grid Width Distribution")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_grid_size_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("PIXEL ART DATASET - FINE-TUNING ELIGIBILITY ANALYSIS")
    print("=" * 60)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    ds = load_data()
    print(f"  ✓ Loaded {len(ds)} samples")

    # Run EDA
    print("[2/4] Running EDA...")
    stats = run_eda(ds)
    valid_samples = stats["num_samples"] - stats["invalid_samples"]
    print(f"  ✓ Analyzed {valid_samples} valid samples (invalid: {stats['invalid_samples']})")

    # Assess eligibility
    print("[3/4] Assessing fine-tuning eligibility...")
    assessor = FineTuningEligibility()
    eligibility = assessor.assess(stats)
    recommendation = assessor.recommend(eligibility)
    print(f"  ✓ Overall Score: {eligibility['overall_score']:.1f}/100")
    print(f"  ✓ Tier: {recommendation['tier']}")

    # Generate visualizations
    print("[4/4] Generating visualizations...")
    plot_sequence_length_distribution(stats)
    print("  ✓ 01_sequence_length.png")
    plot_entropy_distribution(stats)
    print("  ✓ 02_entropy_distribution.png")
    plot_symmetry_analysis(stats)
    print("  ✓ 03_symmetry_analysis.png")
    plot_color_and_sparsity(stats)
    print("  ✓ 04_color_sparsity.png")
    plot_eligibility_scorecard(eligibility)
    print("  ✓ 05_eligibility_scorecard.png")
    plot_grid_size_distribution(stats)
    print("  ✓ 06_grid_size_distribution.png")

    # Save JSON reports
    eda_report = {k: v for k, v in stats.items() if not k.startswith("_")}
    with open(OUTPUT_DIR / "eda_metrics.json", "w") as f:
        json.dump(eda_report, f, indent=2)
    print(f"  ✓ eda_metrics.json")

    with open(OUTPUT_DIR / "eligibility_assessment.json", "w") as f:
        json.dump({
            "eligibility_scores": eligibility,
            "recommendation": recommendation
        }, f, indent=2)
    print(f"  ✓ eligibility_assessment.json")

    # Print detailed summary
    print("\n" + "=" * 60)
    print("ELIGIBILITY ASSESSMENT SUMMARY")
    print("=" * 60)
    for key, item in eligibility.items():
        if key == "overall_score":
            continue
        print(f"\n{key.upper().replace('_', ' ')}")
        print(f"  Score: {item['score']:.0f}/100")
        print(f"  Rationale: {item['rationale']}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    for key, value in recommendation.items():
        if key == "note":
            print(f"\n📌 {value}")
        else:
            print(f"\n{key.upper()}: {value}")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()