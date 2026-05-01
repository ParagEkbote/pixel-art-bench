import json
import torch
import numpy as np
from collections import Counter

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ==========================================
# 1. Config
# ==========================================
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_NAME = "AINovice2005/pixel-art-bench-v1"
MAX_LENGTH = 1024

# ==========================================
# 2. Load tokenizer & model
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# FIX 1: `dtype` → `torch_dtype` for from_pretrained
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

# ==========================================
# 3. LoRA + Compile
# ==========================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # FIX 2: also target o_proj and gate_proj for better coverage on Qwen
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 4. Load dataset
# ==========================================
dataset = load_dataset(DATASET_NAME)

# Only keep the three fields that matter for the inference contract:
# example_name (semantic signal) + palette (conditioning) + grid (target output)
# height/width are always 24×24 (zero variance) and num_colors is inferrable
# from len(palette), so both are dropped.
keep_cols = ["example_name", "palette", "grid", "is_appropriate"]

train_ds = dataset["train"]
train_ds = train_ds.remove_columns(
    [col for col in train_ds.column_names if col not in keep_cols]
)

# Filter out flagged samples
train_ds = train_ds.filter(lambda x: x["is_appropriate"] is True)

# ==========================================
# 5. Prompt formatting
# ==========================================
def format_example(example):
    # Option B inference contract: name + palette → grid only.
    #
    # Rationale:
    #   - height/width are always 24×24 (zero variance) → dropped
    #   - num_colors is implicit from len(palette) → dropped
    #   - Conditioning on palette at train time means the same palette can be
    #     supplied at inference for controllable generation, and the model only
    #     needs to learn the easier sub-task of layout given a fixed color set.
    #   - Keeping palette out of the JSON target saves ~30–80 tokens per sample,
    #     easing pressure on MAX_LENGTH=1024 so the full grid fits.
    palette_str = ", ".join(example["palette"])
    prompt = (
        f"Draw pixel art: {example['example_name']} "
        f"using colors [{palette_str}]"
    )

    # Target is grid only — palette is already in the prompt
    target = json.dumps({"grid": example["grid"]})

    text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )

    return {"text": text}

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)

# ==========================================
# 6. Tokenization + loss masking
# ==========================================
# FIX 5: Find assistant header as a token sequence, not a single token.
# "<|im_start|>assistant" may encode to multiple tokens.
ASSISTANT_HEADER = "<|im_start|>assistant"
assistant_ids = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
header_len = len(assistant_ids)

def find_sublist(haystack, needle):
    """Return first index of needle in haystack, or -1."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1

def tokenize(example):
    enc = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors=None,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.copy()

    # Mask everything up to and including the assistant header
    start_idx = find_sublist(input_ids, assistant_ids)
    if start_idx == -1:
        # Header not found — mask all (skip sample from loss)
        mask_until = len(input_ids)
    else:
        mask_until = start_idx + header_len  # include the header itself in mask

    labels[:mask_until] = [-100] * mask_until

    # Also mask padding tokens explicitly
    for i, tok in enumerate(input_ids):
        if tok == tokenizer.pad_token_id:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_dataset = train_ds.map(
    tokenize,
    batched=False,
    remove_columns=train_ds.column_names,
)

# ==========================================
# 7. Token-level weight computation
# ==========================================
# FIX 6: Cap max weight to avoid the extreme frequency imbalance
#         that was causing the dominant-token-ratio collapse.
MAX_WEIGHT = 10.0

def compute_token_weights(tokenized_ds, tokenizer, model):
    counter = Counter()

    for ex in tokenized_ds:
        for label in ex["labels"]:
            if label != -100 and label != tokenizer.pad_token_id:
                counter[label] += 1

    total = sum(counter.values())
    vocab_size = model.config.vocab_size

    weights = torch.ones(vocab_size)

    for token, count in counter.items():
        freq = count / total
        raw_weight = 1.0 / (freq + 1e-8)
        weights[token] = min(raw_weight, MAX_WEIGHT)

    # Normalise to unit mean so LR stays meaningful
    weights = weights / weights.mean()
    return weights

token_weights = compute_token_weights(tokenized_dataset, tokenizer, model)

# ==========================================
# 8. Custom Trainer (token-weighted CE)
# ==========================================
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep on CPU; move per-step to avoid device mismatch
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            weight=self.class_weights.to(shift_logits.device),
            ignore_index=-100,
            reduction="mean",
        )

        # Diagnostic logging
        if self.state.global_step % 10 == 0:
            with torch.no_grad():
                valid = shift_labels != -100
                if valid.sum() > 0:
                    preds = shift_logits[valid].argmax(dim=-1)
                    mode_pred = preds.mode()[0]
                    dominant_ratio = (preds == mode_pred).float().mean()
                else:
                    dominant_ratio = torch.tensor(0.0)

            self.log({
                "train_loss": loss.item(),
                "dominant_token_ratio": dominant_ratio.item(),
            })

        return (loss, outputs) if return_outputs else loss

# ==========================================
# 9. Training args
# ==========================================
training_args = TrainingArguments(
    output_dir="./qwen-pixel-art",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    # FIX 7: Add gradient clipping — grad_norm was hitting ~50, unstable
    max_grad_norm=1.0,
    report_to="tensorboard",
    remove_unused_columns=False,
    logging_dir="./logs",
    # FIX 8: Disable label smoothing if using custom loss (avoids double-counting)
    label_smoothing_factor=0.0,
)

# ==========================================
# 10. Trainer
# ==========================================
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    class_weights=token_weights,
    # FIX 9: Use a proper data collator with pad_to_multiple_of for efficiency
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    ),
)

# ==========================================
# 11. Train
# ==========================================
trainer.train()

# ==========================================
# 12. Save
# ==========================================
model.save_pretrained("./qwen-pixel-art-lora")
tokenizer.save_pretrained("./qwen-pixel-art-lora")