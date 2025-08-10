"""Metrics and logging utilities."""

from typing import List, Optional
import torch
import numpy as np
import math
from datetime import datetime


class MovingAverage:
    """Track moving average of values."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values: List[float] = []

    def update(self, value: float) -> None:
        """Add value to moving average."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get(self) -> float:
        """Get current moving average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def reset(self) -> None:
        """Reset the moving average."""
        self.values = []


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))  # Cap to avoid overflow


class ConsoleLogger:
    """Simple console logger for training metrics."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step = 0
        self.start_time = datetime.now()

    def log(
        self,
        loss: float,
        perplexity: float,
        lr: float,
        tokens_per_sec: Optional[float] = None,
    ) -> None:
        """Log metrics to console."""
        self.step += 1

        if self.step % self.log_interval == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            log_str = f"Step {self.step:6d} | "
            log_str += f"Loss: {loss:.4f} | "
            log_str += f"PPL: {perplexity:.2f} | "
            log_str += f"LR: {lr:.2e} | "

            if tokens_per_sec is not None:
                log_str += f"Tok/s: {tokens_per_sec:.0f} | "

            log_str += f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}"

            print(log_str)
