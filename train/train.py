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
    EarlyStoppingCallback,  # FIX A: must be imported at top level, not inside class
)
from peft import LoraConfig, get_peft_model

import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ==========================================
# 1. Config
# ==========================================
MODEL_NAME   = "Qwen/Qwen3-1.7B"
DATASET_NAME = "AINovice2005/pixel-art-bench-v1"
MAX_LENGTH   = 1024

# ==========================================
# 2. Load tokenizer & model
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# FIX B: `dtype=` is not a valid kwarg — must be `torch_dtype=`
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ==========================================
# 3. LoRA
# ==========================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 4. Load & filter dataset
# ==========================================
dataset = load_dataset(DATASET_NAME)

keep_cols = ["example_name", "palette", "grid", "is_appropriate"]
train_ds  = dataset["train"]
train_ds  = train_ds.remove_columns(
    [col for col in train_ds.column_names if col not in keep_cols]
)
train_ds = train_ds.filter(lambda x: x["is_appropriate"] is True)

# ==========================================
# 5. Prompt formatting  (Option B: name + palette → grid only)
# ==========================================
def format_example(example):
    palette_str = ", ".join(example["palette"])
    prompt = (
        f"Draw pixel art: {example['example_name']} "
        f"using colors [{palette_str}]"
    )
    target = json.dumps({"grid": example["grid"]})
    text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )
    return {"text": text}

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)

# ==========================================
# 6. Tokenisation + loss masking
# ==========================================
ASSISTANT_HEADER = "<|im_start|>assistant"
assistant_ids    = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
header_len       = len(assistant_ids)

def find_sublist(haystack, needle):
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

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels         = input_ids.copy()

    start_idx  = find_sublist(input_ids, assistant_ids)
    mask_until = (start_idx + header_len) if start_idx != -1 else len(input_ids)
    labels[:mask_until] = [-100] * mask_until

    for i, tok in enumerate(input_ids):
        if tok == tokenizer.pad_token_id:
            labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

tokenized_dataset = train_ds.map(
    tokenize,
    batched=False,
    remove_columns=train_ds.column_names,
)

# ==========================================
# 7. Token-level weight computation  (capped at MAX_WEIGHT)
# ==========================================
MAX_WEIGHT = 10.0

def compute_token_weights(tokenized_ds, tokenizer, model):
    counter = Counter()
    for ex in tokenized_ds:
        for label in ex["labels"]:
            if label != -100 and label != tokenizer.pad_token_id:
                counter[label] += 1

    total      = sum(counter.values())
    vocab_size = model.config.vocab_size
    weights    = torch.ones(vocab_size)

    for token, count in counter.items():
        freq = count / total
        weights[token] = min(1.0 / (freq + 1e-8), MAX_WEIGHT)

    return weights / weights.mean()

token_weights = compute_token_weights(tokenized_dataset, tokenizer, model)

# ==========================================
# 8. Custom Trainer (token-weighted CE)
# ==========================================
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # FIX C: EarlyStoppingCallback belongs in the callbacks= arg of the
        #         Trainer constructor below, NOT as a stray line inside __init__

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size   = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            weight=self.class_weights.to(shift_logits.device),
            ignore_index=-100,
            reduction="mean",
        )

        if self.state.global_step % 10 == 0:
            with torch.no_grad():
                valid = shift_labels != -100
                if valid.sum() > 0:
                    preds          = shift_logits[valid].argmax(dim=-1)
                    dominant_ratio = (preds == preds.mode()[0]).float().mean()
                else:
                    dominant_ratio = torch.tensor(0.0)
            self.log({
                "train_loss"           : loss.item(),
                "dominant_token_ratio" : dominant_ratio.item(),
            })

        return (loss, outputs) if return_outputs else loss

# ==========================================
# 9. Training args
# ==========================================
training_args = TrainingArguments(
    output_dir="./qwen-pixel-art",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    # FIX D: warmup_steps must be an int (number of steps), not a float ratio.
    #         Use warmup_ratio= for a float.
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    report_to="tensorboard",
    remove_unused_columns=False,
    # FIX E: `TENSORBOARD_LOGGING_DIR` is not a valid arg — correct kwarg is `logging_dir`
    logging_dir="./logs",
    label_smoothing_factor=0.0,
    # Required by EarlyStoppingCallback
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    # FIX F: evaluation needs a strategy + eval dataset for early stopping to work.
    #         Using a small 5% split of train here since there is no val split.
    eval_strategy="steps",
    eval_steps=100,
)

# ==========================================
# 10. Train / eval split for early stopping
# ==========================================
# FIX F: EarlyStoppingCallback requires eval_dataset — carve out 5 % of train.
split       = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
train_split = split["train"]
eval_split  = split["test"]

# ==========================================
# 11. Trainer
# ==========================================
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=eval_split,          # FIX F: required for early stopping
    class_weights=token_weights,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    ),
    # FIX C: EarlyStoppingCallback placed here, not inside __init__
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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