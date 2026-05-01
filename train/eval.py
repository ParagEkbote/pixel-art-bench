"""
eval_pixel_art_lora.py
Evaluates base Qwen3-1.7B vs LoRA-finetuned model on pixel-art generation.

Inference contract (must match training):
  INPUT : "Draw pixel art: {example_name} using colors [{palette_str}]"
  OUTPUT: {"grid": <24 strings of 24 palette-index chars>}
"""

import json
import re
import torch
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel          # FIX 2: proper LoRA loading
from tqdm import tqdm

# ==========================================
# Config
# ==========================================
MODEL_NAME  = "Qwen/Qwen3-1.7B"
LORA_PATH   = "./qwen-pixel-art-lora"
EVAL_N      = 10
GRID_SIZE   = 24

# ==========================================
# Load tokenizer + models
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model.eval()

# FIX 2: Load LoRA adapter on top of the base model — NOT a standalone model
lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    torch_dtype=torch.bfloat16,
)
lora_model.eval()

# FIX 3: torch.compile must be assigned back to the module-level name,
#         not a loop variable.  Also skip compile on LoRA (PEFT + compile
#         interaction is fragile; base gets the speed-up for fair comparison).
base_model = torch.compile(base_model, mode="reduce-overhead", fullgraph=False)

# ==========================================
# Dataset
# ==========================================
dataset = load_dataset("AINovice2005/pixel-art-bench-v1")["train"].select(range(EVAL_N))

# ==========================================
# Prompt builder — matches training contract exactly (Option B)
# ==========================================
def build_messages(example):
    # FIX 1: Prompt must match training format.
    # Training used: name + palette → grid only.
    # We therefore supply the ground-truth palette as a conditioning signal,
    # which is the intended inference contract for Option B.
    palette_str = ", ".join(example["palette"])
    return [
        {
            "role": "user",
            "content": (
                f"Draw pixel art: {example['example_name']} "
                f"using colors [{palette_str}]"
            ),
        }
    ]

# ==========================================
# Reasoning suppression (Qwen3-specific)
# ==========================================
# FIX 5: Pass enable_thinking=False through apply_chat_template instead of
#         relying on bad_words_ids (deprecated, fragile for multi-token strings).
THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)

def strip_reasoning(text: str) -> str:
    text = THINK_BLOCK.sub("", text)
    text = text.replace("```json", "").replace("```", "")
    return text.strip()

def extract_json(text: str) -> str:
    text = strip_reasoning(text)
    start = text.find("{")
    if start == -1:
        return ""
    text = text[start:]
    # Patch truncated JSON by balancing braces
    ob = text.count("{")
    cb = text.count("}")
    if cb < ob:
        text += "}" * (ob - cb)
    try:
        json.loads(text)
        return text
    except Exception:
        return ""

# ==========================================
# Generation
# ==========================================
def generate(model, example):
    messages = build_messages(example)

    # FIX 5: Disable thinking mode via chat template flag (Qwen3 supports this)
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,      # Qwen3: suppress <think> natively
        )
    except TypeError:
        # Fallback for tokenizers that don't support enable_thinking
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            # FIX 5: bad_words_ids removed; thinking suppressed via template flag
        )

    gen_ids = out[0][input_len:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    js  = extract_json(raw)
    return js, raw

# ==========================================
# Metrics
# FIX 4: All metrics updated to match new schema — output is {"grid": ...} only.
#         palette is no longer in the model output.
# ==========================================

def json_validity(s: str) -> float:
    """1.0 if s is valid JSON, else 0.0."""
    try:
        json.loads(s)
        return 1.0
    except Exception:
        return 0.0

def render_success(s: str, palette: list) -> float:
    """
    1.0 if the grid is a list of GRID_SIZE strings each of length GRID_SIZE,
    where every character is a valid palette index (0 … len(palette)-1).

    FIX 4: Checks grid-only output and validates against the actual palette
    length rather than hardcoded 0–9.
    """
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        if not isinstance(g, list) or len(g) != GRID_SIZE:
            return 0.0
        max_idx = str(len(palette) - 1)
        valid_chars = set(str(i) for i in range(len(palette)))
        for row in g:
            if not isinstance(row, str) or len(row) != GRID_SIZE:
                return 0.0
            if not all(ch in valid_chars for ch in row):
                return 0.0
        return 1.0
    except Exception:
        return 0.0

def pixel_art_quality(s: str, palette: list) -> float:
    """
    Composite score: color diversity + balance.

    FIX 6: Uses actual palette length instead of hardcoded 10 so the
    diversity ceiling is correct for palettes of size 4–8.
    """
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        if not g:
            return 0.0
        n_colors = len(palette)
        px = [int(ch) for row in g for ch in row if ch.isdigit()]
        if not px:
            return 0.0
        uniq = len(set(px))
        diversity = min(uniq / n_colors, 1.0)          # FIX 6: use palette length
        counts = np.bincount(px, minlength=n_colors)
        dom    = counts.max() / len(px)
        balance = 1.0 - dom
        return 0.7 * diversity + 0.3 * balance
    except Exception:
        return 0.0

def row_consistency(s: str) -> float:
    """Fraction of adjacent same-color pixel pairs (spatial smoothness proxy)."""
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        if not g:
            return 0.0
        changes, total = 0, 0
        for row in g:
            for i in range(len(row) - 1):
                total += 1
                if row[i] != row[i + 1]:
                    changes += 1
        return 1.0 - (changes / total) if total > 0 else 0.0
    except Exception:
        return 0.0

def grid_completeness(s: str) -> float:
    """Fraction of the 24×24 grid that was actually filled (handles truncation)."""
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        rows_ok   = min(len(g), GRID_SIZE)
        cells_ok  = 0
        for row in g[:GRID_SIZE]:
            if isinstance(row, str):
                cells_ok += min(len(row), GRID_SIZE)
        return cells_ok / (GRID_SIZE * GRID_SIZE)
    except Exception:
        return 0.0

# ==========================================
# Evaluate
# ==========================================
rows = []

for i, ex in enumerate(tqdm(dataset, desc="Evaluating")):
    palette = ex["palette"]

    base_js, base_raw = generate(base_model, ex)
    lora_js, lora_raw = generate(lora_model, ex)

    if i < 2:
        print("\n=== DEBUG ===")
        print("BASE RAW :\n", base_raw[:400])
        print("BASE JSON:\n", base_js[:200])
        print("LORA RAW :\n", lora_raw[:400])
        print("LORA JSON:\n", lora_js[:200])

    bm = {
        "json"        : json_validity(base_js),
        "render"      : render_success(base_js, palette),
        "quality"     : pixel_art_quality(base_js, palette),
        "consistency" : row_consistency(base_js),
        "completeness": grid_completeness(base_js),
    }
    lm = {
        "json"        : json_validity(lora_js),
        "render"      : render_success(lora_js, palette),
        "quality"     : pixel_art_quality(lora_js, palette),
        "consistency" : row_consistency(lora_js),
        "completeness": grid_completeness(lora_js),
    }

    rows.append({
        "id"          : i,
        "example_name": ex["example_name"],
        **{f"base_{k}": v for k, v in bm.items()},
        **{f"lora_{k}": v for k, v in lm.items()},
        "delta_quality"     : lm["quality"]      - bm["quality"],
        "delta_consistency" : lm["consistency"]  - bm["consistency"],
        "delta_completeness": lm["completeness"] - bm["completeness"],
        "lora_win"    : int(lm["quality"] > bm["quality"]),
    })

df = pd.DataFrame(rows)
df.to_csv("lora_vs_base_eval.csv", index=False)

summary = {
    "base_json_validity"   : df["base_json"].mean(),
    "lora_json_validity"   : df["lora_json"].mean(),
    "json_validity_gain"   : (df["lora_json"]  - df["base_json"]).mean(),

    "base_render_success"  : df["base_render"].mean(),
    "lora_render_success"  : df["lora_render"].mean(),
    "render_success_gain"  : (df["lora_render"] - df["base_render"]).mean(),

    "base_quality_mean"    : df["base_quality"].mean(),
    "lora_quality_mean"    : df["lora_quality"].mean(),
    "quality_gain"         : df["delta_quality"].mean(),

    "base_completeness"    : df["base_completeness"].mean(),
    "lora_completeness"    : df["lora_completeness"].mean(),
    "completeness_gain"    : df["delta_completeness"].mean(),

    "lora_win_rate"        : df["lora_win"].mean(),
}

print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k:<28}: {v:.4f}")