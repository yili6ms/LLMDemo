"""Integration tests for end-to-end pipeline functionality."""

import pytest
import torch
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys

from tok.bpe import BPETokenizer, train_bpe
from model.gpt import TinyGPT
from train import get_batch, train_step, evaluate, save_checkpoint, load_checkpoint
from sample import sample
from utils.csv_logger import CSVLogger


class TestTokenizerIntegration:
    """Test tokenizer integration with other components."""

    def test_tokenizer_train_save_load_cycle(self):
        """Test complete tokenizer training, saving, and loading cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            data_path = Path(tmpdir) / "data.txt"
            tokenizer_path = Path(tmpdir) / "tokenizer.json"

            test_text = (
                """
            The quick brown fox jumps over the lazy dog.
            Pack my box with five dozen liquor jugs.
            How vexingly quick daft zebras jump!
            The five boxing wizards jump quickly.
            """
                * 10
            )  # Repeat for more training data

            with open(data_path, "w") as f:
                f.write(test_text)

            # Train tokenizer
            tokenizer = train_bpe(data_path, tokenizer_path, vocab_size=500)

            assert tokenizer_path.exists()
            assert len(tokenizer.merges) > 0
            assert tokenizer.vocab_size == 500

            # Load tokenizer
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(tokenizer_path)

            # Test round-trip encoding/decoding
            test_sentences = [
                "The quick brown fox",
                "Hello world!",
                "This is a test sentence.",
            ]

            for sentence in test_sentences:
                tokens = new_tokenizer.encode(sentence)
                decoded = new_tokenizer.decode(tokens)
                assert decoded == sentence, f"Round-trip failed for: {sentence}"

    def test_tokenizer_with_model_integration(self):
        """Test tokenizer integration with model training and inference."""
        tokenizer = BPETokenizer(vocab_size=200)
        test_text = "The quick brown fox jumps over the lazy dog. " * 20
        tokenizer.train(test_text)

        # Create model with matching vocab size
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            dropout=0.0,
        )

        # Test encoding and model forward pass
        text = "The quick brown"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])

        logits, _ = model(input_ids)
        assert logits.shape == (1, len(tokens), tokenizer.vocab_size)

        # Test generation
        generated = model.generate(input_ids, max_new_tokens=5)
        decoded = tokenizer.decode(generated[0].tolist())

        assert isinstance(decoded, str)
        assert text in decoded  # Original text should be preserved


class TestTrainingPipeline:
    """Test complete training pipeline integration."""

    def test_minimal_training_loop(self):
        """Test a minimal training loop with all components."""
        # Setup
        vocab_size = 100
        seq_len = 16
        batch_size = 2

        # Create synthetic data
        data = torch.randint(0, vocab_size, (500,))

        # Create model and optimizer
        model = TinyGPT(
            vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=2, dropout=0.1
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        # Training steps
        initial_loss = None
        for step in range(5):
            batch = get_batch(data, batch_size, seq_len, device)
            loss = train_step(model, batch, optimizer)

            if initial_loss is None:
                initial_loss = loss

            assert isinstance(loss, float)
            assert loss > 0

        # Loss should generally decrease (though not guaranteed in 5 steps)
        assert loss != initial_loss, "Loss should change during training"

    def test_training_with_evaluation(self):
        """Test training loop with periodic evaluation."""
        vocab_size = 50
        seq_len = 12
        batch_size = 2

        # Create train and validation data
        train_data = torch.randint(0, vocab_size, (300,))
        val_data = torch.randint(0, vocab_size, (100,))

        model = TinyGPT(
            vocab_size=vocab_size, d_model=32, n_layers=1, n_heads=2, dropout=0.0
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        # Training with evaluation
        for step in range(3):
            # Training step
            batch = get_batch(train_data, batch_size, seq_len, device)
            train_loss = train_step(model, batch, optimizer)

            # Evaluation
            val_metrics = evaluate(
                model, val_data, batch_size, seq_len, device, num_batches=5
            )

            assert "loss" in val_metrics
            assert "perplexity" in val_metrics
            assert val_metrics["loss"] > 0
            assert val_metrics["perplexity"] > 0

    def test_checkpoint_save_load_integration(self):
        """Test checkpoint saving and loading in training context."""
        model = TinyGPT(vocab_size=50, d_model=32, n_layers=1, n_heads=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Simulate LR scheduler (simplified)
        class MockScheduler:
            def __init__(self):
                self.state = {"step": 0}

            def state_dict(self):
                return self.state

            def load_state_dict(self, state):
                self.state = state

        scheduler = MockScheduler()

        config = {"vocab_size": 50, "d_model": 32, "n_layers": 1, "n_heads": 2}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Train for a few steps
            data = torch.randint(0, 50, (200,))
            device = torch.device("cpu")

            for step in range(3):
                batch = get_batch(data, 2, 8, device)
                loss = train_step(model, batch, optimizer)

            # Save checkpoint
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step=3,
                val_loss=loss,
                config=config,
                path=checkpoint_path,
            )

            # Create new model and load checkpoint
            new_model = TinyGPT(**config)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)
            new_scheduler = MockScheduler()

            checkpoint_data = load_checkpoint(
                checkpoint_path, new_model, new_optimizer, new_scheduler
            )

            # Verify checkpoint data
            assert checkpoint_data["step"] == 3
            assert checkpoint_data["val_loss"] == loss

            # Verify model weights are identical
            for orig_param, new_param in zip(
                model.parameters(), new_model.parameters()
            ):
                assert torch.allclose(orig_param, new_param, atol=1e-6)


class TestModelVariants:
    """Test different model configuration variants."""

    @pytest.mark.parametrize(
        "use_rope,use_flash",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_model_variants_training(self, use_rope, use_flash):
        """Test training with different model variants."""
        config = {
            "vocab_size": 100,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "use_rope": use_rope,
            "use_flash": use_flash,
            "dropout": 0.0,
        }

        model = TinyGPT(**config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Test training step
        data = torch.randint(0, 100, (150,))
        device = torch.device("cpu")

        batch = get_batch(data, 2, 8, device)
        loss = train_step(model, batch, optimizer)

        assert isinstance(loss, float)
        assert loss > 0

    def test_model_variants_generation(self):
        """Test text generation with different model variants."""
        tokenizer = BPETokenizer(vocab_size=150)
        tokenizer.train("The quick brown fox jumps over the lazy dog. " * 10)

        configs = [
            {"use_rope": False, "use_flash": False},
            {"use_rope": True, "use_flash": False},
            {"use_rope": False, "use_flash": True},
            {"use_rope": True, "use_flash": True},
        ]

        for config in configs:
            model = TinyGPT(
                vocab_size=150, d_model=64, n_layers=1, n_heads=4, dropout=0.0, **config
            )

            result = sample(
                model=model,
                tokenizer=tokenizer,
                prompt="The quick",
                max_new_tokens=5,
                seed=42,
                device=torch.device("cpu"),
            )

            assert isinstance(result, str)
            assert "The quick" in result


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    def test_complete_workflow(self):
        """Test complete workflow from data to generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create training data
            data_path = Path(tmpdir) / "data.txt"
            tokenizer_path = Path(tmpdir) / "tokenizer.json"
            model_path = Path(tmpdir) / "model.pt"

            training_text = (
                """
            Once upon a time, there was a brave knight.
            The knight fought dragons and saved kingdoms.
            Dragons breathed fire and guarded treasure.
            Kingdoms celebrated when evil was defeated.
            """
                * 5
            )

            with open(data_path, "w") as f:
                f.write(training_text)

            # 2. Train tokenizer
            tokenizer = train_bpe(data_path, tokenizer_path, vocab_size=300)

            # 3. Prepare data
            tokens = tokenizer.encode(training_text)
            data = torch.tensor(tokens)

            # 4. Create and train model
            model = TinyGPT(
                vocab_size=tokenizer.vocab_size,
                d_model=64,
                n_layers=2,
                n_heads=4,
                dropout=0.1,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            device = torch.device("cpu")

            # Quick training
            for step in range(10):
                batch = get_batch(data, batch_size=4, seq_len=16, device=device)
                loss = train_step(model, batch, optimizer)

            # 5. Save model
            config = {
                "vocab_size": tokenizer.vocab_size,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "dropout": 0.1,
            }

            class MockScheduler:
                def state_dict(self):
                    return {}

            save_checkpoint(
                model,
                optimizer,
                MockScheduler(),
                step=10,
                val_loss=loss,
                config=config,
                path=model_path,
            )

            # 6. Load model and generate text
            new_model = TinyGPT(**config)
            checkpoint_data = load_checkpoint(model_path, new_model)

            generated_text = sample(
                model=new_model,
                tokenizer=tokenizer,
                prompt="Once upon a time",
                max_new_tokens=10,
                temperature=1.0,
                device=device,
            )

            assert isinstance(generated_text, str)
            assert "Once upon a time" in generated_text
            assert len(generated_text) > len("Once upon a time")


class TestCSVLoggingIntegration:
    """Test CSV logging integration with training."""

    def test_csv_logging_during_training(self):
        """Test CSV logging integration with training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "training.csv"
            logger = CSVLogger(log_path)

            # Setup training
            model = TinyGPT(vocab_size=50, d_model=32, n_layers=1, n_heads=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            data = torch.randint(0, 50, (200,))
            device = torch.device("cpu")

            # Training with logging
            for step in range(5):
                batch = get_batch(data, 2, 8, device)
                loss = train_step(model, batch, optimizer)

                # Log metrics
                metrics = {
                    "step": step + 1,
                    "loss": loss,
                    "perplexity": torch.exp(torch.tensor(loss)).item(),
                    "learning_rate": 0.01,
                }
                logger.log(metrics)

            # Verify logged data
            logs = logger.read_logs()
            assert len(logs) == 5

            for i, log_entry in enumerate(logs):
                assert log_entry["step"] == i + 1
                assert isinstance(log_entry["loss"], float)
                assert log_entry["loss"] > 0


class TestConfigurationIntegration:
    """Test configuration file integration."""

    def test_yaml_config_integration(self):
        """Test training with YAML configuration."""
        config_data = {
            "vocab_size": 100,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.1,
            "max_seq_len": 64,
            "tie_weights": False,
            "use_rope": False,
            "use_flash": True,
            "batch_size": 4,
            "seq_len": 16,
            "learning_rate": 0.001,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # Load and use config
            with open(config_path, "r") as f:
                loaded_config = yaml.safe_load(f)

            # Create model from config
            model = TinyGPT(
                vocab_size=loaded_config["vocab_size"],
                d_model=loaded_config["d_model"],
                n_layers=loaded_config["n_layers"],
                n_heads=loaded_config["n_heads"],
                dropout=loaded_config["dropout"],
                max_seq_len=loaded_config["max_seq_len"],
                tie_weights=loaded_config.get("tie_weights", False),
                use_rope=loaded_config.get("use_rope", False),
                use_flash=loaded_config.get("use_flash", True),
            )

            # Test model works
            input_ids = torch.randint(0, loaded_config["vocab_size"], (2, 8))
            logits, _ = model(input_ids)

            assert logits.shape == (2, 8, loaded_config["vocab_size"])

    def test_json_config_integration(self):
        """Test training with JSON configuration."""
        config_data = {
            "vocab_size": 150,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.2,
            "learning_rate": 0.0005,
            "batch_size": 8,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Load and verify
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            assert loaded_config == config_data


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    def test_mixed_precision_integration(self):
        """Test mixed precision training integration."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")

        device = torch.device("cuda")
        model = TinyGPT(vocab_size=100, d_model=32, n_layers=1, n_heads=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scaler = torch.cuda.amp.GradScaler()

        data = torch.randint(0, 100, (200,)).to(device)

        # Training step with mixed precision
        batch = get_batch(data, 2, 8, device)
        loss = train_step(model, batch, optimizer, scaler=scaler)

        assert isinstance(loss, float)
        assert loss > 0

    def test_memory_efficiency_integration(self):
        """Test memory efficiency during training."""
        # Test with larger model to check memory usage
        model = TinyGPT(vocab_size=500, d_model=128, n_layers=2, n_heads=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        data = torch.randint(0, 500, (1000,))
        device = torch.device("cpu")

        # Multiple training steps to check for memory leaks
        for step in range(10):
            batch = get_batch(data, 4, 32, device)
            loss = train_step(model, batch, optimizer)

            # Should complete without memory issues
            assert isinstance(loss, float)

            # Clear any accumulated gradients
            optimizer.zero_grad()
