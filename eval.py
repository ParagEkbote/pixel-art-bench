import json
import torch
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ==========================================
# 1. Setup
# ==========================================
MODEL_NAME = "Qwen/Qwen3-1.7B"
LORA_PATH = "./qwen-pixel-art-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_model = AutoModelForCausalLM.from_pretrained(
    LORA_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

dataset = load_dataset("AINovice2005/pixel-art-bench-v1")["train"]
dataset = dataset.select(range(200))  # eval subset

# ==========================================
# 2. Prompt
# ==========================================
def build_prompt(example):
    return f"""Draw pixel art on a 24x24 grid. Return JSON with:
- palette: list of hex colors (no #)
- grid: 24 strings of length 24 (digits 0-9)
Output ONLY JSON.

Draw: {example['example_name']}"""

# ==========================================
# 3. Generation
# ==========================================
def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=800,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==========================================
# 4. Metrics
# ==========================================
def json_validity(output):
    try:
        json.loads(output)
        return 1.0
    except:
        return 0.0

def render_success(output):
    try:
        data = json.loads(output)
        grid = data.get("grid", [])

        if not isinstance(grid, list) or len(grid) != 24:
            return 0.0

        for row in grid:
            if not isinstance(row, str) or len(row) != 24:
                return 0.0
            if not all(ch.isdigit() and 0 <= int(ch) <= 9 for ch in row):
                return 0.0

        return 1.0
    except:
        return 0.0

def pixel_art_quality(output):
    try:
        data = json.loads(output)
        grid = data.get("grid", [])

        if not grid:
            return 0.0

        pixels = [int(ch) for row in grid for ch in row]
        unique_colors = len(set(pixels))

        diversity = min(unique_colors / 10.0, 1.0)

        counts = np.bincount(pixels, minlength=10)
        dominant_ratio = counts.max() / len(pixels)

        balance = 1.0 - dominant_ratio

        return 0.7 * diversity + 0.3 * balance
    except:
        return 0.0

def row_consistency(output):
    try:
        data = json.loads(output)
        grid = data.get("grid", [])

        if not grid:
            return 0.0

        changes, total = 0, 0

        for row in grid:
            for i in range(len(row) - 1):
                total += 1
                if row[i] != row[i+1]:
                    changes += 1

        return 1.0 - (changes / total)
    except:
        return 0.0

# ==========================================
# 5. Evaluate (PAIRWISE)
# ==========================================
rows = []

for idx, sample in enumerate(tqdm(dataset)):
    prompt = build_prompt(sample)

    base_out = generate(base_model, prompt)
    lora_out = generate(lora_model, prompt)

    base_metrics = {
        "json": json_validity(base_out),
        "render": render_success(base_out),
        "quality": pixel_art_quality(base_out),
        "consistency": row_consistency(base_out),
    }

    lora_metrics = {
        "json": json_validity(lora_out),
        "render": render_success(lora_out),
        "quality": pixel_art_quality(lora_out),
        "consistency": row_consistency(lora_out),
    }

    row = {
        "id": idx,
        "example_name": sample["example_name"],

        # base
        "base_json": base_metrics["json"],
        "base_render": base_metrics["render"],
        "base_quality": base_metrics["quality"],
        "base_consistency": base_metrics["consistency"],

        # lora
        "lora_json": lora_metrics["json"],
        "lora_render": lora_metrics["render"],
        "lora_quality": lora_metrics["quality"],
        "lora_consistency": lora_metrics["consistency"],

        # deltas
        "delta_quality": lora_metrics["quality"] - base_metrics["quality"],
        "delta_consistency": lora_metrics["consistency"] - base_metrics["consistency"],

        # win signal
        "lora_win": int(lora_metrics["quality"] > base_metrics["quality"]),
    }

    rows.append(row)

# ==========================================
# 6. Save CSV
# ==========================================
df = pd.DataFrame(rows)
df.to_csv("lora_vs_base_eval.csv", index=False)

# ==========================================
# 7. Summary (console)
# ==========================================
summary = {
    "base_quality_mean": df["base_quality"].mean(),
    "lora_quality_mean": df["lora_quality"].mean(),
    "quality_gain": df["delta_quality"].mean(),
    "lora_win_rate": df["lora_win"].mean(),
    "render_success_gain": df["lora_render"].mean() - df["base_render"].mean(),
}

print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")