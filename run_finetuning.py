import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from whisper import Whisper
from whisper.tokenizer import get_tokenizer

from dataloader import get_dataloader

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    # Dataloader arguments
    parser.add_argument("--train-json", type=str, required=True, help="Path to training data json")
    parser.add_argument("--dev-json", type=str, required=True, help="Path to dev data json")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
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
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to reduce VRAM usage")
    parser.add_argument("--continue-from", type=str, default=None, help="Path to a checkpoint to continue training from (only model weights)")
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
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
    
    return total_loss

@torch.no_grad()
def evaluate(model: Whisper, dev_loader: DataLoader, use_amp: bool) -> float:
    model.eval()
    total_loss = 0
    processed_batches = 0
    
    for x, y_in, y_out in tqdm(dev_loader):

#        # Skip batches that exceed context length
#        if y_in.size(1) > model.dims.n_text_ctx:
#            print(f"Skipping validation batch with sequence length {y_in.size(1)}")
#            continue
            
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x, y_in)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)
            
        total_loss += loss.item()
        processed_batches += 1
        
    if processed_batches == 0:
        return float('inf')  # Return high loss if no batches processed
    
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
    # Save model in half precision
    model_copy = copy.deepcopy(model).half()
    
    # Create checkpoint dictionary
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
    start_step: int = 0  # Added start_step parameter
) -> None:
    min_loss = evaluate(model, dev_loader, args.use_amp)
    print(f"Initial loss: {min_loss:.4f}")
    
    # Start from start_step instead of 1
    pbar = tqdm(range(start_step, args.train_steps + 1))
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
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({"loss": f"{train_loss:.4f}", "lr": f"{current_lr:.2e}"})
        
        if step % args.eval_steps == 0:
            eval_loss = evaluate(model, dev_loader, args.use_amp)
            tqdm.write(f"Step {step}: validation loss={eval_loss:.4f}")
            
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
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, f"{args.save_dir}/args.json")

    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, args.device)
    
    # Initialize optimizer and scheduler first
    if args.use_adam_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Load checkpoint if continuing
    start_step = 0
    if args.continue_from:
        checkpoint = torch.load(args.continue_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer and scheduler states if available
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "step" in checkpoint:
            start_step = checkpoint["step"] + 1
            
        print(f"Continued training from {args.continue_from} at step {start_step}")
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    
    # Get context size dynamically
    max_prompt_length = model.dims.n_text_ctx // 2 - 1 if hasattr(model, 'dims') else 223

    fp16 = args.device == "cuda"
    train_loader = get_dataloader(
        json=args.train_json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=args.prompt_use_rate,
        no_timestamps_rate=args.no_timestamps_rate,
        shuffle=True,
    )
    dev_loader = get_dataloader(
        json=args.dev_json,
        tokenizer=tokenizer,
        batch_size=args.dev_batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1.0,  # Always use prompts for validation
        no_timestamps_rate=0.0,  # Always use timestamps for validation
        shuffle=False,
    )
    
    main_loop(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        start_step=start_step  # Pass start_step here
    )

if __name__ == "__main__":
    main()