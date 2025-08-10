"""Tests for CSV logging utilities."""

import pytest
import csv
from pathlib import Path
import tempfile
from datetime import datetime

from utils.csv_logger import CSVLogger


class TestCSVLogger:
    """Test CSVLogger class."""

    def test_init_creates_file(self):
        """Test that initialization creates CSV file with headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = CSVLogger(log_path)

            assert log_path.exists()

            # Check headers
            with open(log_path, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)

            expected_fields = [
                "timestamp",
                "step",
                "epoch",
                "loss",
                "perplexity",
                "learning_rate",
                "tokens_per_sec",
                "gpu_memory_mb",
            ]
            assert headers == expected_fields

    def test_init_with_custom_fields(self):
        """Test initialization with custom field list."""
        custom_fields = ["timestamp", "step", "loss", "custom_metric"]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "custom.csv"
            logger = CSVLogger(log_path, fields=custom_fields)

            # Check headers
            with open(log_path, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)

            assert headers == custom_fields

    def test_log_basic_metrics(self):
        """Test logging basic metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = CSVLogger(log_path)

            metrics = {
                "step": 100,
                "loss": 2.5,
                "perplexity": 12.18,
                "learning_rate": 0.001,
            }

            logger.log(metrics)

            # Read back the logged data
            logs = logger.read_logs()
            assert len(logs) == 1

            logged_data = logs[0]
            assert logged_data["step"] == 100
            assert logged_data["loss"] == 2.5
            assert logged_data["perplexity"] == 12.18
            assert logged_data["learning_rate"] == 0.001
            assert "timestamp" in logged_data  # Should be auto-added

    def test_log_with_timestamp(self):
        """Test logging with explicit timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = CSVLogger(log_path)

            custom_timestamp = "2023-01-01T12:00:00"
            metrics = {"timestamp": custom_timestamp, "step": 50, "loss": 1.8}

            logger.log(metrics)

            logs = logger.read_logs()
            assert logs[0]["timestamp"] == custom_timestamp

    def test_log_missing_fields(self):
        """Test logging when some fields are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = CSVLogger(log_path)

            # Log with only some fields
            metrics = {"step": 25, "loss": 3.2}

            logger.log(metrics)

            logs = logger.read_logs()
            logged_data = logs[0]

            assert logged_data["step"] == 25
            assert logged_data["loss"] == 3.2
            # Missing fields should be None
            assert logged_data["epoch"] is None
            assert logged_data["tokens_per_sec"] is None

    def test_log_extra_fields_filtered(self):
        """Test that extra fields not in schema are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = CSVLogger(log_path)

            metrics = {
                "step": 10,
                "loss": 2.0,
                "extra_field": "should_be_ignored",
                "another_extra": 123,
            }

            logger.log(metrics)

            # Read raw CSV to check what was actually written
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                row = next(reader)

            assert "extra_field" not in row
            assert "another_extra" not in row
            assert row["step"] == "10"
            assert row["loss"] == "2.0"

    def test_multiple_logs(self):
        """Test logging multiple entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = CSVLogger(log_path)

            # Log multiple entries
            for step in range(5):
                metrics = {
                    "step": step * 100,
                    "loss": 3.0 - step * 0.5,
                    "perplexity": 20.0 - step * 2.0,
                }
                logger.log(metrics)

            logs = logger.read_logs()
            assert len(logs) == 5

            # Check that values are correctly stored and ordered
            for i, log_entry in enumerate(logs):
                assert log_entry["step"] == i * 100
                assert log_entry["loss"] == 3.0 - i * 0.5
                assert log_entry["perplexity"] == 20.0 - i * 2.0

    def test_read_logs_empty_file(self):
        """Test reading logs from file with only headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "empty.csv"
            logger = CSVLogger(log_path)

            logs = logger.read_logs()
            assert logs == []

    def test_read_logs_nonexistent_file(self):
        """Test reading logs from non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nonexistent.csv"
            logger = CSVLogger(log_path)
            # Don't create the file
            log_path.unlink()  # Remove the file that was created during init

            logs = logger.read_logs()
            assert logs == []

    def test_numeric_field_conversion(self):
        """Test that numeric fields are properly converted when reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"
            logger = CSVLogger(log_path)

            metrics = {
                "step": 123,
                "epoch": 5,
                "loss": 2.567,
                "perplexity": 13.89,
                "learning_rate": 0.0001,
                "tokens_per_sec": 1500.5,
                "gpu_memory_mb": 2048.0,
            }

            logger.log(metrics)
            logs = logger.read_logs()

            logged_data = logs[0]

            # Check types after reading
            assert isinstance(logged_data["step"], int)
            assert isinstance(logged_data["epoch"], int)
            assert isinstance(logged_data["loss"], float)
            assert isinstance(logged_data["perplexity"], float)
            assert isinstance(logged_data["learning_rate"], float)
            assert isinstance(logged_data["tokens_per_sec"], float)
            assert isinstance(logged_data["gpu_memory_mb"], float)

            # Check values
            assert logged_data["step"] == 123
            assert logged_data["epoch"] == 5
            assert abs(logged_data["loss"] - 2.567) < 1e-10
            assert abs(logged_data["perplexity"] - 13.89) < 1e-10

    def test_invalid_numeric_values(self):
        """Test handling of invalid numeric values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"

            # Manually create CSV with invalid numeric values
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "step", "loss", "perplexity"])
                writer.writerow(["2023-01-01T12:00:00", "invalid", "not_a_number", ""])

            logger = CSVLogger(log_path)
            logs = logger.read_logs()

            # Should handle invalid values gracefully
            assert len(logs) == 1
            logged_data = logs[0]
            assert logged_data["timestamp"] == "2023-01-01T12:00:00"
            # Invalid numeric fields should remain as strings
            assert logged_data["step"] == "invalid"
            assert logged_data["loss"] == "not_a_number"
            assert logged_data["perplexity"] == ""

    def test_append_to_existing_file(self):
        """Test appending to existing CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.csv"

            # Create logger and log some data
            logger1 = CSVLogger(log_path)
            logger1.log({"step": 100, "loss": 2.0})

            # Create new logger instance for same file and log more data
            logger2 = CSVLogger(log_path)
            logger2.log({"step": 200, "loss": 1.5})

            # Read all logs
            logs = logger2.read_logs()
            assert len(logs) == 2
            assert logs[0]["step"] == 100
            assert logs[1]["step"] == 200

    def test_directory_creation(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "test.csv"

            # Directory doesn't exist yet
            assert not nested_path.parent.exists()

            # Should create directories
            logger = CSVLogger(nested_path)

            assert nested_path.parent.exists()
            assert nested_path.exists()

    def test_realistic_training_scenario(self):
        """Test CSV logger with realistic training data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "training.csv"
            logger = CSVLogger(log_path)

            # Simulate training progress
            base_time = datetime(2023, 1, 1, 10, 0, 0)

            for step in range(1, 11):
                timestamp = base_time.replace(second=step * 10)
                metrics = {
                    "timestamp": timestamp.isoformat(),
                    "step": step * 100,
                    "epoch": step // 3,  # Rough epoch calculation
                    "loss": max(0.5, 3.0 - step * 0.25),  # Decreasing loss
                    "perplexity": max(1.6, 20.0 - step * 1.8),  # Decreasing perplexity
                    "learning_rate": max(
                        0.0001, 0.001 - step * 0.00005
                    ),  # Decreasing LR
                    "tokens_per_sec": 1500 + step * 10,  # Increasing throughput
                    "gpu_memory_mb": 2048 + step * 10,  # Slight memory increase
                }
                logger.log(metrics)

            # Verify logged data
            logs = logger.read_logs()
            assert len(logs) == 10

            # Check that loss and perplexity generally decrease
            losses = [log["loss"] for log in logs]
            perplexities = [log["perplexity"] for log in logs]

            assert losses[0] > losses[-1], "Loss should decrease over training"
            assert (
                perplexities[0] > perplexities[-1]
            ), "Perplexity should decrease over training"

            # Check that throughput increases
            throughputs = [log["tokens_per_sec"] for log in logs]
            assert throughputs[0] < throughputs[-1], "Throughput should increase"
