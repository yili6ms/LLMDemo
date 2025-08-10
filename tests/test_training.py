"""Tests for training utilities and functions."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from model.gpt import TinyGPT
from train import (
    evaluate,
    get_batch,
    get_lr_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


class TestTrainingUtils:
    """Test training utility functions."""

    def test_get_batch(self):
        """Test batch generation from data."""
        # Create test data
        data = torch.arange(100)  # Sequential data for predictable testing
        device = torch.device("cpu")

        batch_size = 4
        seq_len = 10

        x, y = get_batch(data, batch_size, seq_len, device)

        # Check shapes
        assert x.shape == (batch_size, seq_len)
        assert y.shape == (batch_size, seq_len)

        # Check that targets are shifted by 1
        for i in range(batch_size):
            start_idx = x[i, 0].item()
            # Verify the sequence is correct
            expected_x = torch.arange(start_idx, start_idx + seq_len)
            expected_y = torch.arange(start_idx + 1, start_idx + seq_len + 1)
            assert torch.equal(
                x[i], expected_x
            ), f"Row {i}: expected {expected_x}, got {x[i]}"
            assert torch.equal(
                y[i], expected_y
            ), f"Row {i}: expected {expected_y}, got {y[i]}"

    def test_get_batch_device(self):
        """Test that batch tensors are on correct device."""
        data = torch.arange(50)
        device = torch.device("cpu")

        x, y = get_batch(data, 2, 5, device)

        assert x.device == device
        assert y.device == device

    def test_lr_scheduler_warmup(self):
        """Test learning rate scheduler warmup phase."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        warmup_steps = 10
        max_steps = 100
        scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps)

        # Test warmup phase
        for step in range(warmup_steps):
            expected_lr = 0.001 * (step + 1) / warmup_steps
            actual_lr = scheduler.get_last_lr()[0]
            assert (
                abs(actual_lr - expected_lr) < 1e-6
            ), f"Step {step}: expected {expected_lr}, got {actual_lr}"
            scheduler.step()

    def test_lr_scheduler_cosine_decay(self):
        """Test learning rate scheduler cosine decay phase."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        warmup_steps = 10
        max_steps = 100
        scheduler = get_lr_scheduler(
            optimizer, warmup_steps, max_steps, min_lr_ratio=0.1
        )

        # Skip warmup
        for _ in range(warmup_steps):
            scheduler.step()

        # Test decay phase - should start at max LR and decrease
        first_decay_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # After several steps, LR should be lower
        for _ in range(20):
            scheduler.step()

        later_lr = scheduler.get_last_lr()[0]
        assert later_lr < first_decay_lr, "LR should decrease during cosine decay"

        # At the end, should approach min_lr
        for _ in range(max_steps):
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]
        expected_min_lr = 0.001 * 0.1  # min_lr_ratio = 0.1
        assert (
            abs(final_lr - expected_min_lr) < 1e-6
        ), f"Expected min LR {expected_min_lr}, got {final_lr}"

    def test_train_step(self):
        """Test single training step."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create batch
        batch_size, seq_len = 2, 8
        x = torch.randint(0, 100, (batch_size, seq_len))
        y = torch.randint(0, 100, (batch_size, seq_len))
        batch = (x, y)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Training step
        loss = train_step(model, batch, optimizer)

        # Check that loss is a float
        assert isinstance(loss, float)
        assert loss > 0

        # Check that parameters changed
        for initial_param, current_param in zip(initial_params, model.parameters(), strict=False):
            assert not torch.equal(
                initial_param, current_param
            ), "Parameters should change after training step"

    def test_train_step_with_scaler(self):
        """Test training step with gradient scaler."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()

        batch_size, seq_len = 2, 8
        x = torch.randint(0, 100, (batch_size, seq_len))
        y = torch.randint(0, 100, (batch_size, seq_len))
        batch = (x, y)

        loss = train_step(model, batch, optimizer, scaler=scaler)

        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate(self):
        """Test model evaluation."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2)

        # Create validation data
        val_data = torch.randint(0, 100, (500,))  # 500 tokens
        device = torch.device("cpu")

        metrics = evaluate(
            model, val_data, batch_size=4, seq_len=10, device=device, num_batches=5
        )

        assert "loss" in metrics
        assert "perplexity" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["perplexity"], float)
        assert metrics["loss"] > 0
        assert metrics["perplexity"] > 0

    def test_save_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = get_lr_scheduler(optimizer, 10, 100)

        config = {"vocab_size": 100, "d_model": 32, "n_layers": 1, "n_heads": 2}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

            # Save checkpoint
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step=123,
                val_loss=2.5,
                config=config,
                path=checkpoint_path,
            )

            assert checkpoint_path.exists()

            # Create new model and load checkpoint
            new_model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = get_lr_scheduler(new_optimizer, 10, 100)

            checkpoint_data = load_checkpoint(
                checkpoint_path, new_model, new_optimizer, new_scheduler
            )

            # Check loaded data
            assert checkpoint_data["step"] == 123
            assert checkpoint_data["val_loss"] == 2.5
            assert checkpoint_data["config"] == config

            # Check that model weights are the same
            for orig_param, new_param in zip(
                model.parameters(), new_model.parameters(), strict=False
            ):
                assert torch.equal(orig_param, new_param)


class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_training_config_loading(self):
        """Test loading training configuration."""
        config_data = {
            "vocab_size": 500,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 2,
            "learning_rate": 0.001,
            "batch_size": 4,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # Test YAML loading
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config == config_data

            # Test JSON config
            json_path = Path(tmpdir) / "test_config.json"
            with open(json_path, "w") as f:
                json.dump(config_data, f)

            with open(json_path) as f:
                loaded_json_config = json.load(f)

            assert loaded_json_config == config_data

    def test_model_creation_from_config(self):
        """Test creating model from configuration."""
        config = {
            "vocab_size": 500,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.1,
            "max_seq_len": 128,
            "tie_weights": True,
            "use_rope": False,
            "use_flash": True,
        }

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
        )

        # Basic checks
        assert model.vocab_size == config["vocab_size"]
        assert model.d_model == config["d_model"]
        assert model.n_layers == config["n_layers"]

        # Test forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

        logits, loss = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config["vocab_size"])
        assert loss is None  # No targets provided

    @pytest.mark.parametrize(
        "use_rope,use_flash",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_model_variants(self, use_rope, use_flash):
        """Test different model configuration variants."""
        config = {
            "vocab_size": 100,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "use_rope": use_rope,
            "use_flash": use_flash,
        }

        model = TinyGPT(**config)

        # Test forward pass
        x = torch.randint(0, 100, (2, 8))
        logits, _ = model(x)

        assert logits.shape == (2, 8, 100)

        # Test generation
        output = model.generate(x, max_new_tokens=5)
        assert output.shape == (2, 13)  # Original 8 + 5 new tokens
