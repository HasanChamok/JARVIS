"""
train.py
Fine-tunes JARVIS using Unsloth on RTX 4050.

KEY CHANGE from v1:
  Instead of always starting from base llama3.2,
  we now start from the PREVIOUS jarvis training if it exists.
  This means each run BUILDS ON TOP of the last — continual learning.

  Run 1: llama3.2 base → train all history  → jarvis-finetuned/
  Run 2: jarvis-finetuned/ → train new 20   → jarvis-finetuned/ (updated)
  Run 3: jarvis-finetuned/ → train new 20   → jarvis-finetuned/ (updated again)
"""

import json
import os
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_FILE = "dataset.json"
LORA_DIR     = "jarvis-finetuned"         # where LoRA adapters are saved
GGUF_DIR     = "jarvis-finetuned-gguf"    # where Ollama GGUF is saved
BASE_MODEL   = "unsloth/llama-3.2-3b-instruct-bnb-4bit"

MAX_SEQ_LEN  = 2048
EPOCHS       = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 2e-4
LORA_RANK    = 16

# ── Determine starting point ───────────────────────────────────────────────────
# This is the KEY change — if we have a previous jarvis model, use it.
# If not (first ever run), start from base llama.

if os.path.exists(LORA_DIR) and os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")):
    START_FROM = LORA_DIR
    is_first_run = False
    print("="*50)
    print("  JARVIS Incremental Fine-tuning")
    print(f"  Starting from: previous JARVIS model")
    print("="*50)
else:
    START_FROM = BASE_MODEL
    is_first_run = True
    print("="*50)
    print("  JARVIS Initial Fine-tuning")
    print(f"  Starting from: base llama3.2")
    print("="*50)

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading model from: {START_FROM}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = START_FROM,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)

# ── Apply LoRA ─────────────────────────────────────────────────────────────────
# On first run — add fresh LoRA adapters
# On subsequent runs — the adapters already exist in the loaded model
# Unsloth handles this automatically

print(f"\n[2/5] Applying LoRA adapters (rank={LORA_RANK})...")

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = LORA_RANK,    # keep alpha = rank for consistent scaling
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)

# ── Load dataset ───────────────────────────────────────────────────────────────
print(f"\n[3/5] Loading dataset from {DATASET_FILE}...")

if not Path(DATASET_FILE).exists():
    print(f"[ERROR] {DATASET_FILE} not found. Run clean_memory.py first.")
    exit(1)

raw        = json.loads(Path(DATASET_FILE).read_text(encoding="utf-8"))
print(f"[INFO] Training examples: {len(raw)}")

# Warn if dataset is too small
if len(raw) < 10:
    print("[WARN] Very small dataset — results may be poor.")
    print("[WARN] Talk to Jarvis more and retrain later.")

def format_messages(example):
    """Convert messages list → ChatML format string."""
    text = ""
    for msg in example["messages"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}

hf_dataset = Dataset.from_list(raw)
hf_dataset = hf_dataset.map(format_messages)

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Training...")
print(f"      Mode:       {'First run (full history)' if is_first_run else 'Incremental (new turns only)'}")
print(f"      Examples:   {len(raw)}")
print(f"      Epochs:     {EPOCHS}")
print(f"      GPU:        {torch.cuda.get_device_name(0)}")
print(f"      VRAM:       {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = hf_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    dataset_num_proc   = 2,
    args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps                = 5,
        num_train_epochs            = EPOCHS,
        learning_rate               = LR,
        fp16  = not torch.cuda.is_bf16_supported(),
        bf16  = torch.cuda.is_bf16_supported(),
        logging_steps               = 5,
        optim = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 42,
        output_dir                  = LORA_DIR + "-checkpoints",
        report_to                   = "none",
    ),
)

# Print VRAM before training
reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
total    = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
print(f"      VRAM used before training: {reserved}GB / {total}GB")

stats = trainer.train()

peak_vram = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n      Training complete!")
print(f"      Peak VRAM:      {peak_vram}GB / {total}GB")
print(f"      Training loss:  {stats.training_loss:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\n[5/5] Saving updated model...")

# Save LoRA adapters — this is what gets loaded NEXT time
# by replacing the old jarvis-finetuned/ folder
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"      LoRA adapters saved to: {LORA_DIR}/")

# Export GGUF for Ollama
print("      Exporting GGUF for Ollama...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method = "q4_k_m"
)
print(f"      GGUF saved to: {GGUF_DIR}/")

print("\n" + "="*50)
print("  Training complete!")
print(f"  Next: ollama create jarvis -f Modelfile")
print("="*50 + "\n")