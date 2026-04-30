import json
import re
import torch
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen3-1.7B"
LORA_PATH = "./qwen-pixel-art-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
lora_model = AutoModelForCausalLM.from_pretrained(
    LORA_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)

# compile
for m in (base_model, lora_model):
    m.eval()
    m = torch.compile(m, mode="reduce-overhead", fullgraph=False)

dataset = load_dataset("AINovice2005/pixel-art-bench-v1")["train"].select(range(50))

# -------------------------------
# Prompt (chat-aligned)
# -------------------------------
def build_messages(example):
    return [
        {"role": "user",
         "content": (
            "Draw pixel art on a 24x24 grid. Return JSON with:\n"
            "- palette: list of hex colors (no #)\n"
            "- grid: 24 strings of length 24 (digits 0-9)\n"
            "Output ONLY JSON.\n\n"
            f"Draw: {example['example_name']}"
         )}
    ]

# -------------------------------
# Reasoning suppression
# -------------------------------
def get_bad_words_ids(tokenizer):
    # block tokens that start reasoning
    bad = ["<think>", "</think>"]
    ids = []
    for s in bad:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if len(toks) > 0:
            ids.append(toks)
    return ids

BAD_WORDS = get_bad_words_ids(tokenizer)

# -------------------------------
# Cleanup + extraction
# -------------------------------
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
    # balance braces (handles truncation)
    ob = text.count("{")
    cb = text.count("}")
    if cb < ob:
        text = text + "}" * (ob - cb)
    try:
        json.loads(text)
        return text
    except:
        return ""

# -------------------------------
# Generation
# -------------------------------
def generate(model, example):
    messages = build_messages(example)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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
            bad_words_ids=BAD_WORDS  # block <think>
        )

    gen = out[0][input_len:]
    raw = tokenizer.decode(gen, skip_special_tokens=True)
    js = extract_json(raw)
    return js, raw

# -------------------------------
# Metrics
# -------------------------------
def json_validity(s):
    try:
        json.loads(s); return 1.0
    except: return 0.0

def render_success(s):
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        if not isinstance(g, list) or len(g) != 24: return 0.0
        for r in g:
            if not isinstance(r, str) or len(r) != 24: return 0.0
            if not all(ch.isdigit() for ch in r): return 0.0
        return 1.0
    except:
        return 0.0

def pixel_art_quality(s):
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        if not g: return 0.0
        px = [int(ch) for row in g for ch in row]
        uniq = len(set(px))
        diversity = min(uniq / 10.0, 1.0)
        counts = np.bincount(px, minlength=10)
        dom = counts.max() / len(px)
        balance = 1.0 - dom
        return 0.7 * diversity + 0.3 * balance
    except:
        return 0.0

def row_consistency(s):
    try:
        d = json.loads(s)
        g = d.get("grid", [])
        if not g: return 0.0
        changes, total = 0, 0
        for r in g:
            for i in range(len(r)-1):
                total += 1
                if r[i] != r[i+1]: changes += 1
        return 1.0 - (changes / total)
    except:
        return 0.0

# -------------------------------
# Evaluate
# -------------------------------
rows = []
for i, ex in enumerate(tqdm(dataset)):
    base_js, base_raw = generate(base_model, ex)
    lora_js, lora_raw = generate(lora_model, ex)

    if i < 2:
        print("\n=== DEBUG ===")
        print("BASE RAW:\n", base_raw[:300])
        print("BASE JSON:\n", base_js)
        print("LORA RAW:\n", lora_raw[:300])
        print("LORA JSON:\n", lora_js)

    bm = {
        "json": json_validity(base_js),
        "render": render_success(base_js),
        "quality": pixel_art_quality(base_js),
        "consistency": row_consistency(base_js),
    }
    lm = {
        "json": json_validity(lora_js),
        "render": render_success(lora_js),
        "quality": pixel_art_quality(lora_js),
        "consistency": row_consistency(lora_js),
    }

    rows.append({
        "id": i,
        "example_name": ex["example_name"],
        **{f"base_{k}": v for k, v in bm.items()},
        **{f"lora_{k}": v for k, v in lm.items()},
        "delta_quality": lm["quality"] - bm["quality"],
        "delta_consistency": lm["consistency"] - bm["consistency"],
        "lora_win": int(lm["quality"] > bm["quality"]),
    })

df = pd.DataFrame(rows)
df.to_csv("lora_vs_base_eval.csv", index=False)

summary = {
    "base_quality_mean": df["base_quality"].mean(),
    "lora_quality_mean": df["lora_quality"].mean(),
    "quality_gain": df["delta_quality"].mean(),
    "lora_win_rate": df["lora_win"].mean(),
    "render_success_gain": df["lora_render"].mean() - df["base_render"].mean(),
    "json_validity_gain": df["lora_json"].mean() - df["base_json"].mean(),
}
print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")