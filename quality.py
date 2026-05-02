#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")


# =========================
# CONFIG & STYLING
# =========================

PLOT_DIR = Path("benchmark_plots")
PLOT_DIR.mkdir(exist_ok=True)

# Professional publication style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Configure matplotlib for publication quality
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
})

COLORS = sns.color_palette("husl", 15)
ACCENT = "#E63946"
BG_COLOR = "#F8F9FA"


def save_plot(name):
    """Save plot with high quality and consistent styling."""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / name, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
    print(f"✓ Saved: {name}")
    plt.close()


# =========================
# LOAD DATASET
# =========================

DATASET_NAME = "AINovice2005/pixel-art-bench-v1"
ds = load_dataset(DATASET_NAME, split="train")


# =========================
# SAFE TOKEN PARSER
# =========================

def safe_int(c):
    try:
        return int(c)
    except:
        return None


# =========================
# SAFE GRID PARSER
# =========================

def safe_grid_to_numpy(grid, expected_h, expected_w):
    parsed = []

    for row in grid:
        if not isinstance(row, str) or len(row) != expected_w:
            raise ValueError

        parsed_row = []
        for c in row:
            val = safe_int(c)
            if val is None:
                raise ValueError
            parsed_row.append(val)

        parsed.append(parsed_row)

    if len(parsed) != expected_h:
        raise ValueError

    g = np.array(parsed)

    if g.ndim != 2 or g.shape != (expected_h, expected_w):
        raise ValueError

    return g


# =========================
# METRICS
# =========================

def grid_shape_valid(sample):
    try:
        safe_grid_to_numpy(sample["grid"], sample["height"], sample["width"])
        return 1
    except:
        return 0


def palette_validity(sample):
    grid = sample["grid"]
    max_idx = len(sample["palette"]) - 1

    valid, total = 0, 0
    for row in grid:
        for c in row:
            total += 1
            val = safe_int(c)
            if val is not None and val <= max_idx:
                valid += 1

    return valid / total if total > 0 else np.nan


def normalized_entropy(sample):
    try:
        g = safe_grid_to_numpy(sample["grid"], sample["height"], sample["width"])
    except:
        return np.nan

    flat = g.flatten()
    counts = Counter(flat)
    probs = np.array(list(counts.values())) / len(flat)

    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(len(sample["palette"]) + 1e-9)

    return entropy / max_entropy if max_entropy > 0 else np.nan


def edge_density(sample):
    try:
        g = safe_grid_to_numpy(sample["grid"], sample["height"], sample["width"])
    except:
        return np.nan

    h, w = g.shape
    if h < 2 or w < 2:
        return np.nan

    diffs, total = 0, 0
    for i in range(h - 1):
        for j in range(w - 1):
            total += 2
            diffs += (g[i, j] != g[i+1, j])
            diffs += (g[i, j] != g[i, j+1])

    return diffs / total if total > 0 else np.nan


def color_efficiency(sample):
    try:
        g = safe_grid_to_numpy(sample["grid"], sample["height"], sample["width"])
    except:
        return np.nan

    return len(set(g.flatten())) / len(sample["palette"])


def fill_balance(sample):
    try:
        g = safe_grid_to_numpy(sample["grid"], sample["height"], sample["width"])
    except:
        return np.nan

    ratio = np.mean(g != 0)
    return 1 - abs(ratio - 0.5)


# =========================
# COMPUTE METRICS
# =========================

rows = []

for sample in ds:
    rows.append({
        "model": sample["model_slug"],
        "grid_valid": grid_shape_valid(sample),
        "palette_valid": palette_validity(sample),
        "edge_density": edge_density(sample),
        "entropy": normalized_entropy(sample),
        "color_eff": color_efficiency(sample),
        "fill_balance": fill_balance(sample),
        "cost": sample["cost"],
        "time": sample["generation_time"]
    })

df = pd.DataFrame(rows)


# =========================
# CLEAN + FILTER
# =========================

df["degenerate"] = df[[
    "edge_density", "entropy", "color_eff", "fill_balance"
]].isna().any(axis=1).astype(int)

df_valid = df.dropna().copy()


# =========================
# QUALITY SCORE
# =========================

df_valid["quality_score"] = (
    2.0 * df_valid["grid_valid"] +
    2.0 * df_valid["palette_valid"] +
    1.5 * df_valid["edge_density"] +
    1.0 * df_valid["entropy"] +
    1.0 * df_valid["color_eff"] +
    1.0 * df_valid["fill_balance"]
)


# =========================
# EFFICIENCY
# =========================

df_valid["efficiency_score"] = (
    df_valid["quality_score"] - np.log(df_valid["cost"] + 1e-9)
)


# =========================
# PRINT SUMMARY
# =========================

print("\n" + "="*50)
print("PIXEL ART BENCH SUMMARY")
print("="*50)

print("\n=== TOP QUALITY MODELS ===")
top_quality = df_valid.groupby("model")["quality_score"].mean().nlargest(10)
for i, (model, score) in enumerate(top_quality.items(), 1):
    print(f"{i:2d}. {model:25s} {score:7.2f}")

print("\n=== ROBUSTNESS (Structural Validity) ===")
robustness = df.groupby("model")[["grid_valid", "degenerate"]].mean().sort_values("grid_valid", ascending=False).head(10)
print(robustness)

print("\n=== EFFICIENCY (Quality per Cost) ===")
top_efficiency = df_valid.groupby("model")["efficiency_score"].mean().nlargest(10)
for i, (model, score) in enumerate(top_efficiency.items(), 1):
    print(f"{i:2d}. {model:25s} {score:7.2f}")

print("\n" + "="*50)


# =========================
# VISUALIZATION 1: QUALITY DISTRIBUTION (VIOLIN + BOX)
# =========================

def plot_quality_distribution():
    top_models = df_valid.groupby("model")["quality_score"].mean().nlargest(8).index
    subset = df_valid[df_valid["model"].isin(top_models)]

    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.violinplot(data=subset, x="model", y="quality_score", palette="Set2", ax=ax)
    sns.stripplot(data=subset, x="model", y="quality_score", 
                  color="black", alpha=0.3, size=4, ax=ax)
    
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Quality Score", fontsize=12, fontweight="bold")
    ax.set_title("Quality Score Distribution (Top 8 Models)", 
                 fontsize=13, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    
    save_plot("01_quality_distribution.png")


# =========================
# VISUALIZATION 2: ROBUSTNESS HEATMAP + BAR
# =========================

def plot_robustness():
    agg = df.groupby("model")[["grid_valid", "palette_valid", "degenerate"]].mean().nlargest(12, "grid_valid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    sns.heatmap(agg.T, annot=True, fmt=".2f", cmap="RdYlGn", 
                cbar_kws={"label": "Score"}, ax=ax1, vmin=0, vmax=1)
    ax1.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Metric", fontsize=11, fontweight="bold")
    ax1.set_title("Robustness Heatmap", fontsize=12, fontweight="bold")
    
    # Bar chart
    grid_valid = agg["grid_valid"].sort_values(ascending=False)
    bars = ax2.barh(range(len(grid_valid)), grid_valid.values, color=COLORS[:len(grid_valid)])
    ax2.set_yticks(range(len(grid_valid)))
    ax2.set_yticklabels(grid_valid.index)
    ax2.set_xlabel("Structural Validity Rate", fontsize=11, fontweight="bold")
    ax2.set_title("Model Robustness Ranking", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 1)
    
    # Add value labels
    for i, (idx, v) in enumerate(grid_valid.items()):
        ax2.text(v + 0.02, i, f"{v:.1%}", va="center", fontsize=9)
    
    save_plot("02_robustness_analysis.png")


# =========================
# VISUALIZATION 3: EFFICIENCY FRONTIER (PARETO)
# =========================

def plot_frontier():
    agg = df_valid.groupby("model").agg({
        "quality_score": "mean",
        "cost": "mean",
        "efficiency_score": "mean"
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cost vs Quality
    scatter1 = ax1.scatter(agg["cost"], agg["quality_score"], 
                          s=agg["efficiency_score"]*30, alpha=0.6, 
                          c=agg["efficiency_score"], cmap="viridis")
    
    # Pareto front
    pareto_idx = []
    for i in range(len(agg)):
        if not any((agg.loc[j, "quality_score"] >= agg.loc[i, "quality_score"] and 
                    agg.loc[j, "cost"] <= agg.loc[i, "cost"] and
                    j != i) for j in range(len(agg))):
            pareto_idx.append(i)
    
    if pareto_idx:
        pareto = agg.iloc[pareto_idx].sort_values("cost")
        ax1.plot(pareto["cost"], pareto["quality_score"], 
                "r--", linewidth=2, alpha=0.7, label="Pareto Front")
    
    ax1.set_xlabel("Average Cost ($)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Quality Score", fontsize=11, fontweight="bold")
    ax1.set_title("Cost-Quality Tradeoff", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label("Efficiency", fontsize=10)
    
    # Efficiency ranking
    top_eff = agg.nlargest(12, "efficiency_score").sort_values("efficiency_score")
    colors_eff = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_eff)))
    ax2.barh(range(len(top_eff)), top_eff["efficiency_score"].values, color=colors_eff)
    ax2.set_yticks(range(len(top_eff)))
    ax2.set_yticklabels(top_eff["model"].values)
    ax2.set_xlabel("Efficiency Score", fontsize=11, fontweight="bold")
    ax2.set_title("Top Efficiency Models", fontsize=12, fontweight="bold")
    
    for i, v in enumerate(top_eff["efficiency_score"].values):
        ax2.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=9)
    
    save_plot("03_efficiency_frontier.png")


# =========================
# VISUALIZATION 4: FAILURE VS QUALITY SCATTER
# =========================

def plot_failure_vs_quality():
    q = df_valid.groupby("model")["quality_score"].mean()
    f = df.groupby("model")["degenerate"].mean()
    vol = df.groupby("model").size()

    common = q.index.intersection(f.index).intersection(vol.index)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(f.loc[common] * 100, q.loc[common], 
                        s=vol.loc[common]/2, alpha=0.6, 
                        c=q.loc[common], cmap="coolwarm", edgecolors="black", linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(f.loc[common] * 100, q.loc[common], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(f.loc[common].min()*100, f.loc[common].max()*100, 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.6, label="Trend (poly fit)")
    
    ax.set_xlabel("Failure Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Quality Score", fontsize=12, fontweight="bold")
    ax.set_title("Quality vs Reliability (bubble size = sample count)", 
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend()
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Quality Score", fontsize=10)
    
    save_plot("04_failure_vs_quality.png")


# =========================
# VISUALIZATION 5: METRIC CORRELATIONS
# =========================

def plot_metric_correlations():
    metrics = ["grid_valid", "palette_valid", "edge_density", "entropy", "color_eff", "fill_balance"]
    corr = df_valid[metrics].corr()
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"label": "Correlation"},
                vmin=-1, vmax=1, linewidths=1, linecolor="white")
    
    ax.set_title("Metric Correlation Matrix", fontsize=13, fontweight="bold", pad=20)
    labels = ["Grid Valid", "Palette Valid", "Edge Density", "Entropy", "Color Eff.", "Fill Balance"]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    
    save_plot("05_metric_correlations.png")


# =========================
# VISUALIZATION 6: MULTI-METRIC RADAR
# =========================

def plot_radar_comparison():
    # Top 5 models by quality
    top_models = df_valid.groupby("model")["quality_score"].mean().nlargest(5).index
    
    metrics_to_plot = ["grid_valid", "palette_valid", "edge_density", "entropy", "color_eff", "fill_balance"]
    model_scores = df_valid[df_valid["model"].isin(top_models)].groupby("model")[metrics_to_plot].mean()
    
    # Normalize to 0-1
    model_scores_norm = (model_scores - model_scores.min()) / (model_scores.max() - model_scores.min())
    
    angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    
    for idx, model in enumerate(top_models):
        values = model_scores_norm.loc[model].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=COLORS[idx])
        ax.fill(angles, values, alpha=0.15, color=COLORS[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Grid Valid", "Palette Valid", "Edge Density", 
                        "Entropy", "Color Eff.", "Fill Balance"], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.set_title("Top 5 Models: Multi-Metric Comparison", 
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    save_plot("06_radar_comparison.png")


# =========================
# VISUALIZATION 7: TIME VS COST VS QUALITY
# =========================

def plot_3d_analysis():
    agg = df_valid.groupby("model").agg({
        "cost": "mean",
        "time": "mean",
        "quality_score": "mean"
    }).reset_index()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    scatter = ax.scatter(agg["cost"], agg["time"], agg["quality_score"],
                        s=200, c=agg["quality_score"], cmap="viridis", alpha=0.7)
    
    ax.set_xlabel("Cost ($)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Generation Time (s)", fontsize=11, fontweight="bold")
    ax.set_zlabel("Quality Score", fontsize=11, fontweight="bold")
    ax.set_title("3D Performance Space: Cost × Time × Quality", 
                 fontsize=13, fontweight="bold", pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Quality", fontsize=10)
    
    save_plot("07_3d_analysis.png")


# =========================
# RUN ALL PLOTS
# =========================

print("\n📊 Generating publication-quality visualizations...\n")

plot_quality_distribution()
plot_robustness()
plot_frontier()
plot_failure_vs_quality()
plot_metric_correlations()
plot_radar_comparison()
plot_3d_analysis()

print("\n✅ All visualizations generated successfully!")
print(f"📁 Output directory: {PLOT_DIR.absolute()}\n")