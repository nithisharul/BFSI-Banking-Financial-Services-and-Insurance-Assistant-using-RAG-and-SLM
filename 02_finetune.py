import os
import json
import torch

# ── GPU Check ─────────────────────────────────────────────────────────────────
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL    = r"C:\Users\Asus\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"
DATA_PATH     = "dataset/bfsi_alpaca.json"
OUTPUT_DIR    = "models/phi2_bfsi_slm"
MAX_LENGTH    = 512
BATCH_SIZE    = 2
EPOCHS        = 3
LEARNING_RATE = 3e-4

# ── Load dataset ──────────────────────────────────────────────────────────────
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = {
    "instruction": [d["instruction"] for d in data],
    "input":       [d.get("input", "") for d in data],
    "output":      [d["output"] for d in data]
}

def combine_fields(examples):
    combined = []
    for ins, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        if inp.strip():
            text = f"Instruction: {ins}\nInput: {inp}\nAnswer: {out}"
        else:
            text = f"Instruction: {ins}\nAnswer: {out}"
        combined.append(text)
    return {"text": combined}

hf_dataset = Dataset.from_dict(dataset)
hf_dataset = hf_dataset.map(combine_fields, batched=True)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = hf_dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(
    ["instruction", "input", "output", "text"]
)

# ── 4-bit quantization config ─────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ── Load model ────────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda",        # ← force GPU
    trust_remote_code=True
)

# ── LoRA config ───────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training args ─────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_total_limit=2,
    save_strategy="epoch",
    fp16=True,                  # RTX 4050 supports fp16
    optim="adamw_torch",
    report_to="none"
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    args=training_args
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Fine-tuned model saved to", OUTPUT_DIR)
