import argparse
import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import whisper
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from whisper import Whisper
from whisper.tokenizer import get_tokenizer

from dataloader import get_dataset, collate_fn
#from evaluate import ModelEvaluator

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    # Dataloader arguments
    parser.add_argument("--train-json", type=str, required=True, help="Path to training data json")
    parser.add_argument("--dev-json", type=str, required=True, help="Path to dev data json")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size (per GPU)")
    parser.add_argument("--dev-batch-size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--no-timestamps-training", action="store_true", help="Always use no-timestamps mode")
    parser.add_argument("--prompt-use-rate", type=float, default=0.5, help="Prompt usage probability")
    parser.add_argument("--no-timestamps-rate", type=float, default=0.5, help="No-timestamps probability")

    # Training arguments
    parser.add_argument("--save-dir", type=str, default="output", help="Model save directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--model", default="large", choices=whisper.available_models(), help="Whisper model name")
    parser.add_argument("--train-only-decoder", action="store_true", help="Train only decoder")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--accum-grad-steps", type=int, default=64, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps for scheduler")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--train-steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--save-all-checkpoints", action="store_true", help="Save all checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-adam-8bit", action="store_true", help="Use Adam 8bit optimizer")
    parser.add_argument("--use-amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--continue-from", type=str, default=None, help="Path to checkpoint to continue training")

    parser.add_argument("--eval-datasets", nargs="+", default=["common_voice", "fleurs", "controlled_no_diacritics",         "controlled_with_diacritics"], help="Datasets to evaluate on")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N epochs")
    
    # Distributed training arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU)")
    parser.add_argument("--dist-url", default="env://", help="URL used to set up distributed training")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")
    
    return parser

def train_step(
    model: Whisper,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    accum_grad_steps: int,
    train_only_decoder: bool,
    max_grad_norm: float,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0
    processed_steps = 0
    
    # Pre-fetch first batch
    next_batch = next(train_iter)
    
    for i in range(accum_grad_steps):
        # Start async transfer for current batch
        current_batch = next_batch
        if i < accum_grad_steps - 1:
            # Pre-fetch next batch in background
            with torch.no_grad():
                next_batch = next(train_iter)
        
        x, y_in, y_out = current_batch
        x, y_in, y_out = x.to(model.device, non_blocking=True), \
                          y_in.to(model.device, non_blocking=True), \
                          y_out.to(model.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if train_only_decoder:
                with torch.no_grad():
                    audio_features = model.embed_audio(x)
            else:
                audio_features = model.embed_audio(x)
                
            logits = model.logits(y_in, audio_features=audio_features)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out) / accum_grad_steps

        if use_amp:
            scaler.scale(loss).backward(retain_graph=False)
        else:
            loss.backward(retain_graph=False)
            
        total_loss += loss.item()
        processed_steps += 1

    if processed_steps > 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    
    return total_loss

@torch.no_grad()
def evaluate(model: Whisper, dev_loader: DataLoader, use_amp: bool) -> float:
    model.eval()
    total_loss = 0
    processed_batches = 0
    
    for x, y_in, y_out in tqdm(dev_loader):
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x, y_in)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)
            
        total_loss += loss.item()
        processed_batches += 1
        
    if processed_batches == 0:
        return float('inf')
    
    torch.cuda.empty_cache()
    
    return total_loss / processed_batches

def save_model(
    model: Whisper, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    save_path: str
) -> None:
    """Save model checkpoint with training states"""
    # Handle DDP model
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    
    # Save model in half precision
    model_copy = copy.deepcopy(model).half()
    
    checkpoint = {
        "model_state_dict": model_copy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "dims": asdict(model_copy.dims)
    }
    torch.save(checkpoint, save_path)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch

def main_loop(
    model: Whisper,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    start_step: int = 0,
    is_main_process: bool = True
) -> None:

    # Force 128 mel bins to prioritize v3 models
    model.dims.n_mels = 128
    model.encoder.conv1 = nn.Conv1d(128, model.dims.n_audio_state, kernel_size=3, padding=1)

    # Only main process handles evaluation
    if is_main_process:
        min_loss = evaluate(model, dev_loader, args.use_amp)
        print(f"Initial loss: {min_loss:.4f}")
        
        # Initialize evaluator for external datasets
        evaluator = ModelEvaluator(device=args.device, batch_size=args.dev_batch_size)
    else:
        min_loss = float('inf')
    
    # Broadcast initial loss to all processes
    if args.gpus > 1:
        min_loss_tensor = torch.tensor(min_loss).to(model.device)
        dist.broadcast(min_loss_tensor, 0)
        min_loss = min_loss_tensor.item()
    
    pbar = tqdm(range(start_step, args.train_steps + 1), disable=not is_main_process)
    train_iter = infinite_iter(train_loader)
    
    for step in pbar:
        if step % 100 == 0:
            torch.cuda.empty_cache()
        
        train_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            scaler,
            args.accum_grad_steps,
            args.train_only_decoder,
            args.max_grad_norm,
            args.use_amp,
        )
        
        if is_main_process:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{train_loss:.4f}", "lr": f"{current_lr:.2e}"})
        
        if step % args.eval_steps == 0 and is_main_process:
            # Calculate epoch number
            epoch = step // (len(train_loader) // args.accum_grad_steps)
            
            # Standard validation loss
            eval_loss = evaluate(model, dev_loader, args.use_amp)
            tqdm.write(f"Step {step}: validation loss={eval_loss:.4f}")
            
            # External dataset evaluation
            if epoch % args.eval_interval == 0:
                tqdm.write(f"Evaluating on external datasets at epoch {epoch}...")
                results = evaluator.evaluate_model(
                    model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model,
                    model_id=f"step_{step}",
                    datasets=args.eval_datasets
                )
                
                tqdm.write(f"\nEpoch {epoch} Metrics:")
                for dataset_id, metrics in results.items():
                    tqdm.write(f"  {metrics['name']}:")
                    tqdm.write(f"    WER: {metrics['WER']:.2%}")
                    tqdm.write(f"    CER: {metrics['CER']:.2%}")
                
                # Save metrics to file
                with open(f"{args.save_dir}/metrics_epoch_{epoch}.json", "w") as f:
                    json.dump(results, f, indent=2)
            
            # Model saving
            if eval_loss < min_loss:
                min_loss = eval_loss
                save_model(
                    model, optimizer, scheduler, step,
                    f"{args.save_dir}/best_model.pt"
                )

            if args.save_all_checkpoints:
                save_model(
                    model, optimizer, scheduler, step,
                    f"{args.save_dir}/step{step}.pt"
                )

            save_model(
                model, optimizer, scheduler, step,
                f"{args.save_dir}/last_model.pt"
            )
    
def main():
    args = get_parser().parse_args()
    
    # Initialize distributed training
    if args.gpus > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
        )
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_main_process = (local_rank == 0)
        world_size = torch.distributed.get_world_size()
    else:
        device = torch.device(args.device)
        local_rank = 0
        is_main_process = True
        world_size = 1
    
    # Only main process creates save dir
    if is_main_process:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        save_args(args, f"{args.save_dir}/args.json")
    
    set_seed(args.seed + local_rank)  # Different seed per process
    torch.backends.cudnn.benchmark = True

    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, device)
    
    # Wrap with DDP if multi-GPU
    if args.gpus > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    # Initialize optimizer and scheduler
    if args.use_adam_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=args.train_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Load checkpoint if continuing
    start_step = 0
    if args.continue_from:
        checkpoint = torch.load(args.continue_from, map_location=device)
        
        # Handle DDP model loading
        model_state = checkpoint["model_state_dict"]
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "step" in checkpoint:
            start_step = checkpoint["step"] + 1
            
        if is_main_process:
            print(f"Continued training from {args.continue_from} at step {start_step}")
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    
    # Get context size
    max_prompt_length = model.module.dims.n_text_ctx // 2 - 1 if args.gpus > 1 else model.dims.n_text_ctx // 2 - 1
    
    # Create datasets
    train_dataset = get_dataset(
        json=args.train_json,
        tokenizer=tokenizer,
        fp16=(device.type == "cuda"),
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=args.prompt_use_rate,
        no_timestamps_rate=args.no_timestamps_rate,
    )
    
    dev_dataset = get_dataset(
        json=args.dev_json,
        tokenizer=tokenizer,
        fp16=(device.type == "cuda"),
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1.0,
        no_timestamps_rate=0.0,
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.gpus > 1 else None
    dev_sampler = None  # Validation doesn't need distributed sampling
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.dev_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Only main process runs the training loop
    if is_main_process or args.gpus > 1:
        main_loop(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            start_step=start_step,
            is_main_process=is_main_process
        )
    elif args.gpus > 1:
        # Worker processes just run the training loop
        while True:
            time.sleep(10)  # Keep process alive

if __name__ == "__main__":
    main()