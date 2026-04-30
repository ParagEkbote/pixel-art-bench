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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)

# ==========================================
# 3. LoRA + Compile
# ==========================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


# ==========================================
# 4. Load dataset
# ==========================================
dataset = load_dataset(DATASET_NAME)

# ✅ NEW: Keep only relevant columns
keep_cols = ["example_name", "palette", "grid", "is_appropriate"]

dataset = dataset["train"].remove_columns(
    [col for col in dataset["train"].column_names if col not in keep_cols]
)

# ✅ OPTIONAL: filter bad samples
dataset = dataset.filter(lambda x: x["is_appropriate"] is True)

# ==========================================
# 5. Prompt formatting
# ==========================================
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
# 6. Tokenization + masking (CLEAN VERSION)
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

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.copy()

    try:
        start_idx = input_ids.index(assistant_token)
    except ValueError:
        start_idx = len(input_ids)

    labels[:start_idx] = [-100] * start_idx

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ✅ CRITICAL FIX: remove all old columns here
tokenized_dataset = dataset.map(
    tokenize,
    batched=False,
    remove_columns=dataset.column_names
)

# ==========================================
# 7. Token-level weight computation
# ==========================================
def compute_token_weights(tokenized_ds, tokenizer,model):
    counter = Counter()

    for ex in tokenized_ds:
        for token in ex["input_ids"]:
            if token != tokenizer.pad_token_id:
                counter[token] += 1

    total = sum(counter.values())
    vocab_size = model.config.vocab_size 

    weights = torch.ones(vocab_size)

    for token, count in counter.items():
        freq = count / total
        weights[token] = 1.0 / (freq + 1e-8)

    weights = weights / weights.mean()
    return weights

token_weights = compute_token_weights(
    tokenized_dataset,
    tokenizer,
    model
)

# ==========================================
# 8. Custom Trainer (token-weighted CE)
# ==========================================
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

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
            ignore_index=-100
        )

        # Logging (dominance check)
        if self.state.global_step % 10 == 0:
            with torch.no_grad():
                valid = shift_labels != -100
                preds = shift_logits.argmax(dim=-1)

                if valid.sum() > 0:
                    dominant_ratio = (
                        preds[valid] == preds[valid].mode()[0]
                    ).float().mean()
                else:
                    dominant_ratio = torch.tensor(0.0)

            self.log({
                "train_loss": loss.item(),
                "dominant_token_ratio": dominant_ratio.item()
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
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="tensorboard",
    remove_unused_columns=False,
    logging_dir="./logs"
)

# ==========================================
# 10. Trainer
# ==========================================
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    class_weights=token_weights,
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