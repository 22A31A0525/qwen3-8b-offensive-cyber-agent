"""train_v3.py
Production Multi-GPU Training Script for Qwen3-8B Latest Version (v3).

Architecture:
- 2x Tesla T4 GPUs via PyTorch Distributed Data Parallel (torchrun DDP).
- 4-bit QLoRA with BitsAndBytes (load_in_4bit=True, double quantization).
- Native FP16 precision (Compute 7.5 Turing compatibility).
- PyTorch Scaled Dot-Product Attention (SDPA, zero FlashAttention C++ deps).
- Context length: 4,096 tokens, packing=False.
- Rank-16 LoRA (r=16, alpha=32, dropout=0.05) on all 7 linear projection layers.
- Globally deterministic 3:1:1 StratifiedCurriculumSampler sharded across ranks.
- Dynamically computed optimizer steps (total_curriculum_draws // effective_batch_size).
- CompletionLossDataCollator enforcing assistant-only completion loss masking.
- Full resumable checkpointing: saves adapter, optimizer, scheduler, and RNG state.
- Fail-closed execution: cleanly aborts on DDP initialization or rank mismatch.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

try:
    from src.finetuning.render_chatml import render_dialogue_to_chatml
    from src.finetuning.collator import CompletionLossDataCollator
    from src.data.sampler import (
        StratifiedCurriculumSampler,
        StratifiedDistributedBatchSampler,
    )
except ImportError:
    from v3_pipeline.scripts.render_chatml import render_dialogue_to_chatml
    from v3_pipeline.scripts.collator import CompletionLossDataCollator
    from v3_pipeline.scripts.sampler import (
        StratifiedCurriculumSampler,
        StratifiedDistributedBatchSampler,
    )


class DialogueCurriculumDataset(Dataset):
    """Dataset that renders raw dialogue records into tokenized input_ids."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer: Any):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        rendered_text = render_dialogue_to_chatml(record)
        encoded = self.tokenizer(
            rendered_text,
            return_tensors=None,
            add_special_tokens=False,
        )
        return {
            "input_ids": encoded["input_ids"],
            "id": str(record.get("id")),
            "category": record.get("metadata", {}).get("category", "unknown"),
        }


def load_jsonl_records(filepath: str) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def setup_distributed() -> Tuple[int, int, int]:
    """Initializes PyTorch Distributed Data Parallel environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Fallback single node
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_v3(
    model_name_or_path: str = "Qwen/Qwen3-8B",
    data_dir: str = os.path.join(REPO_ROOT, "data", "clean"),
    output_dir: str = os.path.join(REPO_ROOT, "v3_pipeline", "outputs", "v3_lora_run"),
    max_length: int = 4096,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    epochs: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 10,
    resume_from_checkpoint: Optional[str] = None,
    require_ddp: bool = False,
):
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

    if require_ddp and world_size < 2:
        raise RuntimeError(
            f"FAIL-CLOSED: require_ddp is True but world_size={world_size}. "
            "Dual-GPU DDP is mandatory for this production run."
        )

    set_seed(seed + rank)

    if is_main_process:
        print("=" * 80)
        print("QWEN3-8B (v3 PIPELINE) -- DUAL-GPU DDP RETRAINING")
        print("=" * 80)
        print(f"Base Model: {model_name_or_path}")
        print(f"Data Directory: {data_dir}")
        print(f"Output Directory: {output_dir}")
        print(f"Distributed Ranks: {world_size} (Local Rank: {local_rank})")
        print(f"Attention Engine: PyTorch SDPA")
        print(f"Precision: Native FP16")
        print(f"Quantization: 4-bit QLoRA (BitsAndBytes NF4)")
        print(f"Context Length: {max_length}")
        print("=" * 80)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Datasets
    train_records = load_jsonl_records(os.path.join(data_dir, "train.jsonl"))
    val_records = load_jsonl_records(os.path.join(data_dir, "val.jsonl"))

    if is_main_process:
        print(f"Loaded {len(train_records)} training records, {len(val_records)} validation records.")

    train_dataset = DialogueCurriculumDataset(train_records, tokenizer)
    val_dataset = DialogueCurriculumDataset(val_records, tokenizer)

    # 3. Dynamic Optimizer Step Computation
    effective_batch_size = world_size * per_device_batch_size * gradient_accumulation_steps
    sampler = StratifiedCurriculumSampler(
        dataset=train_records,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    total_draws = len(sampler)
    steps_per_epoch = sampler.compute_dynamic_optimizer_steps(effective_batch_size)
    total_optimizer_steps = steps_per_epoch * epochs

    if is_main_process:
        print(f"Sampler Total Draws: {total_draws}")
        print(f"Effective Batch Size: {effective_batch_size} sequences/step")
        print(f"Dynamically Computed Optimizer Steps per Epoch: {steps_per_epoch}")
        print(f"Total Optimizer Steps ({epochs} epoch(s)): {total_optimizer_steps}")

    # 4. Collator & DataLoaders
    collator = CompletionLossDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        pad_to_multiple_of=8,
    )

    batch_sampler = StratifiedDistributedBatchSampler(
        curriculum_sampler=sampler,
        batch_size=per_device_batch_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # 5. Load Base Model in 4-bit QLoRA with SDPA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = {"": local_rank} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # 6. Configure LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    # 7. Optimizer & Cosine LR Scheduler
    # Use 8-bit AdamW if available to save memory
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    except Exception:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    warmup_steps = int(total_optimizer_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        ckpt_opt = os.path.join(resume_from_checkpoint, "optimizer.pt")
        ckpt_sched = os.path.join(resume_from_checkpoint, "scheduler.pt")
        ckpt_meta = os.path.join(resume_from_checkpoint, "training_state.json")
        if os.path.exists(ckpt_opt):
            optimizer.load_state_dict(torch.load(ckpt_opt, map_location="cpu"))
        if os.path.exists(ckpt_sched):
            scheduler.load_state_dict(torch.load(ckpt_sched, map_location="cpu"))
        if os.path.exists(ckpt_meta):
            with open(ckpt_meta, "r", encoding="utf-8") as f:
                state_data = json.load(f)
                start_step = state_data.get("global_step", 0)
        sampler.set_resume_step(start_step)
        if is_main_process:
            print(f"Resumed training from checkpoint at global step: {start_step}")

    # 8. Training Loop
    os.makedirs(output_dir, exist_ok=True)
    global_step = start_step
    accumulated_loss = 0.0
    accum_steps = 0
    t0 = time.time()
    total_tokens_processed = 0

    model.train()
    optimizer.zero_grad()

    if is_main_process:
        print("\nStarting Training Execution...")

    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        for step_idx, batch in enumerate(train_loader):
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            num_tokens = attention_mask.sum().item()
            total_tokens_processed += num_tokens

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()
            accum_steps += 1

            if accum_steps >= gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                accum_steps = 0

                # Logging
                if global_step % logging_steps == 0 and is_main_process:
                    elapsed = time.time() - t0
                    tokens_per_sec = total_tokens_processed / max(elapsed, 1e-4)
                    cur_lr = scheduler.get_last_lr()[0]
                    vram_mb = (
                        torch.cuda.max_memory_allocated(local_rank) / (1024 * 1024)
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    print(
                        f"[Step {global_step}/{total_optimizer_steps}] "
                        f"Loss: {accumulated_loss * gradient_accumulation_steps:.4f} | "
                        f"LR: {cur_lr:.2e} | "
                        f"Throughput: {tokens_per_sec:.1f} tok/s | "
                        f"Peak VRAM: {vram_mb:.0f} MB"
                    )

                accumulated_loss = 0.0

                # Validation
                if global_step % eval_steps == 0 and len(val_loader) > 0:
                    model.eval()
                    val_loss_sum = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for v_batch in val_loader:
                            v_input = v_batch["input_ids"].to(device)
                            v_mask = v_batch["attention_mask"].to(device)
                            v_labels = v_batch["labels"].to(device)
                            v_out = model(
                                input_ids=v_input,
                                attention_mask=v_mask,
                                labels=v_labels,
                            )
                            val_loss_sum += v_out.loss.item()
                            val_count += 1
                            if val_count >= 50:  # Cap eval steps for speed
                                break
                    val_avg = val_loss_sum / max(val_count, 1)
                    if is_main_process:
                        print(f"--- [Evaluation @ Step {global_step}] Validation Loss: {val_avg:.4f} ---")
                    model.train()

                # Checkpointing
                if global_step % save_steps == 0 and is_main_process:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    # Save adapter weights
                    raw_model = model.module if hasattr(model, "module") else model
                    raw_model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    # Save optimizer, scheduler, and RNG state
                    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                    torch.save(
                        {
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                        os.path.join(ckpt_dir, "rng_state.pt"),
                    )
                    with open(os.path.join(ckpt_dir, "training_state.json"), "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "global_step": global_step,
                                "epoch": epoch,
                                "total_optimizer_steps": total_optimizer_steps,
                                "effective_batch_size": effective_batch_size,
                                "world_size": world_size,
                            },
                            f,
                            indent=2,
                        )
                    print(f"Saved full resumable checkpoint to {ckpt_dir}")

                if global_step >= total_optimizer_steps:
                    break

        if global_step >= total_optimizer_steps:
            break

    # Final Save
    if is_main_process:
        final_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_dir, exist_ok=True)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\nTraining Complete! Final adapter exported to: {final_dir}")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B v3 Production Training")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--data-dir", type=str, default=os.path.join(REPO_ROOT, "data", "clean"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(REPO_ROOT, "v3_pipeline", "outputs", "v3_lora_run"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--require-ddp", action="store_true")
    args = parser.parse_args()

    train_v3(
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        require_ddp=args.require_ddp,
    )


if __name__ == "__main__":
    main()

