"""Training script for TinyGPT."""

from typing import Optional, Dict, Any, Tuple
import argparse
from pathlib import Path
import json
import yaml
import math
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

from model.gpt import TinyGPT
from tok.bpe import BPETokenizer
from utils.metrics import compute_perplexity, ConsoleLogger, MovingAverage


def get_batch(
    data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get random batch of training data."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create learning rate scheduler with warmup and cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        elif step < max_steps:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:
            # Minimum learning rate
            return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
) -> float:
    """Single training step."""
    model.train()
    x, y = batch

    if scaler is not None:
        # Mixed precision training
        with autocast():
            logits, loss = model(x, targets=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Regular training
        logits, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    optimizer.zero_grad()
    return loss.item()


def evaluate(
    model: nn.Module,
    val_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    num_batches: int = 50,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(val_data, batch_size, seq_len, device)
            _, loss = model(x, targets=y)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = compute_perplexity(avg_loss)

    return {"loss": avg_loss, "perplexity": perplexity}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
    val_loss: float,
    config: Dict[str, Any],
    path: Path,
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def train(config: Dict[str, Any]) -> None:
    """Main training loop."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer_path = Path(config.get("tokenizer_path", "tok/bpe.json"))
    if tokenizer_path.exists():
        tokenizer.load(tokenizer_path)
    else:
        print(f"Training tokenizer with vocab size {config['vocab_size']}...")
        from tok.bpe import train_bpe

        train_bpe(
            Path(config["data_path"]), tokenizer_path, vocab_size=config["vocab_size"]
        )
        tokenizer.load(tokenizer_path)

    # Load and tokenize data
    print("Loading and tokenizing data...")
    with open(config["data_path"], "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)

    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Create model
    model = TinyGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
        tie_weights=config.get("tie_weights", False),
        use_rope=config.get("use_rope", False),
        use_flash=config.get("use_flash", True),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Create scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=config.get("warmup_steps", 100),
        max_steps=config["max_steps"],
        min_lr_ratio=config.get("min_lr_ratio", 0.1),
    )

    # Mixed precision training
    scaler = (
        GradScaler()
        if device.type == "cuda" and config.get("mixed_precision", True)
        else None
    )

    # Resume from checkpoint
    start_step = 0
    best_val_loss = float("inf")
    if config.get("resume_from"):
        checkpoint = load_checkpoint(
            Path(config["resume_from"]), model, optimizer, scheduler
        )
        start_step = checkpoint["step"]
        best_val_loss = checkpoint["val_loss"]
        print(f"Resumed from step {start_step}")

    # Training loop
    logger = ConsoleLogger(log_interval=config.get("log_interval", 100))
    loss_avg = MovingAverage(window_size=100)

    print("Starting training...")
    for step in range(start_step, config["max_steps"]):
        # Training step
        batch = get_batch(train_data, config["batch_size"], config["seq_len"], device)
        loss = train_step(model, batch, optimizer, scaler, config.get("grad_clip", 1.0))
        scheduler.step()

        loss_avg.update(loss)

        # Logging
        if step % config.get("log_interval", 100) == 0:
            lr = scheduler.get_last_lr()[0]
            ppl = compute_perplexity(loss_avg.get())
            logger.log(loss_avg.get(), ppl, lr)

        # Evaluation
        if step % config.get("eval_interval", 1000) == 0 and step > 0:
            val_metrics = evaluate(
                model, val_data, config["batch_size"], config["seq_len"], device
            )
            print(
                f"Validation - Loss: {val_metrics['loss']:.4f}, PPL: {val_metrics['perplexity']:.2f}"
            )

            # Save checkpoint if best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    best_val_loss,
                    config,
                    Path(config.get("checkpoint_path", "checkpoints/best.pt")),
                )
                print(f"Saved best checkpoint (loss: {best_val_loss:.4f})")

        # Regular checkpoint
        if step % config.get("checkpoint_interval", 5000) == 0 and step > 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                best_val_loss,
                config,
                Path(f"checkpoints/step_{step}.pt"),
            )

    print("Training complete!")

    # Final evaluation
    final_metrics = evaluate(
        model,
        val_data,
        config["batch_size"],
        config["seq_len"],
        device,
        num_batches=100,
    )
    print(
        f"Final - Loss: {final_metrics['loss']:.4f}, PPL: {final_metrics['perplexity']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train TinyGPT")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Train model
    train(config)


if __name__ == "__main__":
    main()
