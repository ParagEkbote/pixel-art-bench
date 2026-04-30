import json
import torch
import numpy as np
from collections import Counter

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ==========================================
# 3. LoRA
# ==========================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj","k_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model = torch.compile(
    model,
    mode="reduce-overhead",  # best for training
    fullgraph=False          # REQUIRED (dynamic shapes)
)

# ==========================================
# 4. Load dataset
# ==========================================
dataset = load_dataset(DATASET_NAME)

# ==========================================
# 5. Compute class weights (0–9 digits)
# ==========================================
def compute_class_weights(ds):
    counter = Counter()

    for ex in ds:
        for row in ex["grid"]:
            for ch in row:
                counter[int(ch)] += 1

    total = sum(counter.values())
    freqs = np.array([counter[i] for i in range(10)]) / total

    weights = 1.0 / (freqs + 1e-8)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)

class_weights = compute_class_weights(dataset["train"])

# ==========================================
# 6. Prompt formatting
# ==========================================
SYSTEM_PROMPT = """Draw pixel art on a 24x24 grid. Return JSON with:
- palette: list of hex colors (no #)
- grid: 24 strings of length 24 (digits 0-9)
Output ONLY JSON."""

def format_example(example):
    prompt = f"Draw: {example['example_name']}"

    target = json.dumps({
        "palette": example["palette"],
        "grid": example["grid"]
    })

    text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )

    return {"text": text}

dataset = dataset.map(format_example)

# ==========================================
# 7. Tokenization + masking
# ==========================================
assistant_token = tokenizer.encode(
    "<|im_start|>assistant", add_special_tokens=False
)[0]

def tokenize(example):
    enc = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    labels = enc["input_ids"].copy()

    # find assistant start
    try:
        start_idx = labels.index(assistant_token)
    except ValueError:
        start_idx = len(labels)

    # mask everything before assistant response
    labels[:start_idx] = [-100] * start_idx

    enc["labels"] = labels
    return enc

tokenized_dataset = dataset.map(tokenize, batched=False)

# ==========================================
# 8. Digit token mapping (robust)
# ==========================================
digit_token_ids = {
    tokenizer.encode(str(i), add_special_tokens=False)[0]: i
    for i in range(10)
}

digit_token_id_list = list(digit_token_ids.keys())

# ==========================================
# 9. Custom Trainer with weighted loss
# ==========================================
import torch.nn.functional as F

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        # shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # map tokens → digits
        mapped_labels = torch.full_like(shift_labels, -100)

        for token_id, digit in digit_token_ids.items():
            mapped_labels[shift_labels == token_id] = digit

        valid_mask = mapped_labels != -100

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True).to(logits.device)

        logits = shift_logits[valid_mask][:, digit_token_id_list]
        labels = mapped_labels[valid_mask]

        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights.to(logits.device)
        )

        # logging
        if self.state.global_step % 10 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                dominant_ratio = (preds == preds.mode()[0]).float().mean()

            self.log({
                "train_loss": loss.item(),
                "dominant_token_ratio": dominant_ratio.item()
            })

        return (loss, outputs) if return_outputs else loss

# ==========================================
# 10. Training args
# ==========================================
training_args = TrainingArguments(
    output_dir="./qwen-pixel-art",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="tensorboard",
    logging_dir="./logs"
)

# ==========================================
# 11. Trainer
# ==========================================
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    class_weights=class_weights,
)

# ==========================================
# 12. Train
# ==========================================
trainer.train()

# ==========================================
# 13. Save
# ==========================================
model.save_pretrained("./qwen-pixel-art-lora")
tokenizer.save_pretrained("./qwen-pixel-art-lora")