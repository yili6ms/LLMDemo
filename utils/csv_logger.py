"""CSV logging utilities for training metrics."""

import csv
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CSVLogger:
    """CSV logger for training metrics."""

    def __init__(self, log_path: Path, fields: Optional[list] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Default fields
        if fields is None:
            self.fields = [
                "timestamp",
                "step",
                "epoch",
                "loss",
                "perplexity",
                "learning_rate",
                "tokens_per_sec",
                "gpu_memory_mb",
            ]
        else:
            self.fields = fields

        # Initialize CSV file with headers
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log(self, metrics: Dict[str, Any]):
        """Log metrics to CSV file."""
        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()

        # Filter metrics to only include defined fields
        filtered_metrics = {k: v for k, v in metrics.items() if k in self.fields}

        # Fill missing fields with None
        for field in self.fields:
            if field not in filtered_metrics:
                filtered_metrics[field] = None

        # Write to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(filtered_metrics)

    def read_logs(self) -> list:
        """Read all logs from CSV file."""
        if not self.log_path.exists():
            return []

        logs = []
        with open(self.log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for field in [
                    "step",
                    "epoch",
                    "loss",
                    "perplexity",
                    "learning_rate",
                    "tokens_per_sec",
                    "gpu_memory_mb",
                ]:
                    if field in row and row[field] is not None and row[field] != "":
                        try:
                            row[field] = float(row[field])
                            if field in ["step", "epoch"]:
                                row[field] = int(row[field])
                        except ValueError:
                            pass
                logs.append(row)

        return logs
