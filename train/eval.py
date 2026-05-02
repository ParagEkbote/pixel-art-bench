"""
eval_pixel_art_lora.py
Evaluates base Qwen3-1.7B vs LoRA-finetuned model on pixel-art generation.

Inference contract (must match training):
  INPUT : "Draw pixel art: {example_name} using colors [{palette_str}]"
  OUTPUT: {"grid": <24 strings of 24 palette-index chars>}

Key fixes vs previous version:
  - Raw ChatML prompt (not apply_chat_template) to match training format exactly
  - '{' injected as assistant prefix to bypass markdown/think preamble
  - add_special_tokens=False to avoid double-adding ChatML tokens
  - max_new_tokens=1500 (full grid needs ~700-900 tokens)
  - Preflight check before eval loop to catch format issues early
"""

import json
import re
import torch
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ==========================================
# Config
# ==========================================
MODEL_NAME = "Qwen/Qwen3-1.7B"
LORA_PATH  = "./qwen-pixel-art-lora"
EVAL_N     = 2
GRID_SIZE  = 24

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

# Load LoRA adapter on top of the base model — NOT a standalone model load
lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    torch_dtype=torch.bfloat16,
)
lora_model.eval()

# torch.compile assigned back to module-level name (loop variable fix).
# LoRA excluded: PEFT + fullgraph compile is fragile.
base_model = torch.compile(base_model, mode="reduce-overhead", fullgraph=False)

# ==========================================
# Dataset
# ==========================================
dataset = load_dataset("AINovice2005/pixel-art-bench-v1")["train"].select(range(EVAL_N))

# ==========================================
# Prompt builder
# ==========================================
def build_prompt(example: dict) -> str:
    """
    Reconstructs the EXACT string the LoRA saw during training:

        <|im_start|>user
        Draw pixel art: {name} using colors [{palette}]<|im_end|>
        <|im_start|>assistant
        {

    Three things matter here:
      1. Raw ChatML string — apply_chat_template adds a system prompt and
         reformats tokens; the LoRA never saw that, so it won't activate.
      2. add_special_tokens=False at tokenisation — the special tokens are
         already in the raw string; letting the tokenizer add them again
         corrupts the sequence.
      3. The trailing '{' forces the model to continue with JSON immediately,
         bypassing any markdown preamble or <think> block.
    """
    palette_str = ", ".join(example["palette"])
    user_content = (
        f"Draw pixel art: {example['example_name']} "
        f"using colors [{palette_str}]"
    )
    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{{"
    )

# ==========================================
# Reasoning suppression + JSON extraction
# ==========================================
THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)

def strip_reasoning(text: str) -> str:
    text = THINK_BLOCK.sub("", text)
    text = text.replace("```json", "").replace("```", "")
    return text.strip()

def extract_json(text: str) -> str:
    """
    text already starts with '{' (we prepend it after decoding).
    Strip any reasoning, find the JSON object, patch truncation, validate.
    """
    text = strip_reasoning(text)
    start = text.find("{")
    if start == -1:
        return ""
    text = text[start:]
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
def generate(model, example: dict):
    prompt = build_prompt(example)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,   # ChatML tokens already in raw string
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1500,    # full 24×24 grid needs ~700-900 tokens
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_len:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Re-attach the '{' we injected into the prompt
    js = extract_json("{" + raw)
    return js, raw

# ==========================================
# Preflight check — run BEFORE eval loop to verify format is landing correctly.
# If LoRA is activating you should see:  {"grid": ["0123...", ...
# If you still see markdown, training needs more epochs.
# ==========================================
def preflight_check(model, label: str = "model"):
    ex = dataset[0]
    prompt = build_prompt(ex)
    print(f"\n{'='*60}")
    print(f"PREFLIGHT: {label}")
    print(f"{'='*60}")
    print("PROMPT (repr, first 300 chars):")
    print(repr(prompt[:300]))

    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # skip_special_tokens=False so we can see exactly what was emitted
    raw_full = tokenizer.decode(out[0], skip_special_tokens=False)
    print("\nRAW OUTPUT (with special tokens, first 500 chars):")
    print(raw_full[:500])
    print(f"{'='*60}\n")

print("Running preflight checks...")
preflight_check(base_model, label="BASE")
preflight_check(lora_model, label="LORA")

# ==========================================
# Metrics
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
    1.0 if the grid is GRID_SIZE strings of length GRID_SIZE where every
    character is a valid palette index (0 … len(palette)-1).
    """
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        if not isinstance(g, list) or len(g) != GRID_SIZE:
            return 0.0
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
    Composite: 0.7 * color_diversity + 0.3 * color_balance.
    Uses actual palette length as diversity ceiling (not hardcoded 10).
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
        uniq     = len(set(px))
        diversity = min(uniq / n_colors, 1.0)
        counts   = np.bincount(px, minlength=n_colors)
        dom      = counts.max() / len(px)
        balance  = 1.0 - dom
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
    """Fraction of the 24×24 grid actually filled — diagnoses truncation."""
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        cells_ok = 0
        for row in g[:GRID_SIZE]:
            if isinstance(row, str):
                cells_ok += min(len(row), GRID_SIZE)
        return cells_ok / (GRID_SIZE * GRID_SIZE)
    except Exception:
        return 0.0

# ==========================================
# Eval loop
# ==========================================
rows = []

for i, ex in enumerate(tqdm(dataset, desc="Evaluating")):
    palette = ex["palette"]

    base_js, base_raw = generate(base_model, ex)
    lora_js, lora_raw = generate(lora_model, ex)

    if i < 2:
        print(f"\n=== DEBUG sample {i} ===")
        print("BASE RAW (first 400):\n", base_raw[:400])
        print("BASE JSON (first 200):\n", base_js[:200])
        print("LORA RAW (first 400):\n", lora_raw[:400])
        print("LORA JSON (first 200):\n", lora_js[:200])

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
        "id"                : i,
        "example_name"      : ex["example_name"],
        **{f"base_{k}": v for k, v in bm.items()},
        **{f"lora_{k}": v for k, v in lm.items()},
        "delta_quality"     : lm["quality"]      - bm["quality"],
        "delta_consistency" : lm["consistency"]  - bm["consistency"],
        "delta_completeness": lm["completeness"] - bm["completeness"],
        "lora_win"          : int(lm["quality"] > bm["quality"]),
    })

df = pd.DataFrame(rows)
df.to_csv("lora_vs_base_eval.csv", index=False)

summary = {
    "base_json_validity"  : df["base_json"].mean(),
    "lora_json_validity"  : df["lora_json"].mean(),
    "json_validity_gain"  : (df["lora_json"]    - df["base_json"]).mean(),
    "base_render_success" : df["base_render"].mean(),
    "lora_render_success" : df["lora_render"].mean(),
    "render_success_gain" : (df["lora_render"]  - df["base_render"]).mean(),
    "base_quality_mean"   : df["base_quality"].mean(),
    "lora_quality_mean"   : df["lora_quality"].mean(),
    "quality_gain"        : df["delta_quality"].mean(),
    "base_completeness"   : df["base_completeness"].mean(),
    "lora_completeness"   : df["lora_completeness"].mean(),
    "completeness_gain"   : df["delta_completeness"].mean(),
    "lora_win_rate"       : df["lora_win"].mean(),
}

print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k:<28}: {v:.4f}")