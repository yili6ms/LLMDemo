"""Tests for metrics and logging utilities."""

import pytest
import math
from io import StringIO
import sys
from unittest.mock import patch
from datetime import datetime

from utils.metrics import MovingAverage, compute_perplexity, ConsoleLogger


class TestMovingAverage:
    """Test MovingAverage class."""
    
    def test_init(self):
        """Test initialization."""
        ma = MovingAverage(window_size=5)
        assert ma.window_size == 5
        assert ma.values == []
        assert ma.get() == 0.0
    
    def test_single_value(self):
        """Test with single value."""
        ma = MovingAverage(window_size=3)
        ma.update(10.0)
        assert ma.get() == 10.0
    
    def test_multiple_values_within_window(self):
        """Test with multiple values within window size."""
        ma = MovingAverage(window_size=5)
        values = [1.0, 2.0, 3.0]
        
        for val in values:
            ma.update(val)
        
        expected = sum(values) / len(values)
        assert ma.get() == expected
    
    def test_window_overflow(self):
        """Test behavior when exceeding window size."""
        ma = MovingAverage(window_size=3)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for val in values:
            ma.update(val)
        
        # Should only keep last 3 values
        expected = (3.0 + 4.0 + 5.0) / 3
        assert ma.get() == expected
        assert len(ma.values) == 3
    
    def test_reset(self):
        """Test reset functionality."""
        ma = MovingAverage(window_size=3)
        ma.update(1.0)
        ma.update(2.0)
        
        assert ma.get() == 1.5
        
        ma.reset()
        assert ma.values == []
        assert ma.get() == 0.0
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Window size of 1
        ma = MovingAverage(window_size=1)
        ma.update(5.0)
        ma.update(10.0)
        assert ma.get() == 10.0  # Only keeps last value
        
        # Zero window size (should handle gracefully)
        ma_zero = MovingAverage(window_size=0)
        ma_zero.update(5.0)
        assert ma_zero.get() == 0.0  # Should return 0 for invalid window


class TestComputePerplexity:
    """Test perplexity computation."""
    
    def test_normal_values(self):
        """Test perplexity computation with normal values."""
        # Known relationship: perplexity = exp(loss)
        loss = 2.0
        ppl = compute_perplexity(loss)
        expected = math.exp(2.0)
        assert abs(ppl - expected) < 1e-6
    
    def test_zero_loss(self):
        """Test with zero loss."""
        ppl = compute_perplexity(0.0)
        assert ppl == 1.0  # exp(0) = 1
    
    def test_very_high_loss(self):
        """Test with very high loss (should cap to prevent overflow)."""
        loss = 200.0  # Would cause overflow if not capped
        ppl = compute_perplexity(loss)
        expected = math.exp(100)  # Should be capped at 100
        assert ppl == expected
    
    def test_negative_loss(self):
        """Test with negative loss."""
        loss = -1.0
        ppl = compute_perplexity(loss)
        expected = math.exp(-1.0)
        assert abs(ppl - expected) < 1e-6
    
    @pytest.mark.parametrize("loss,expected", [
        (0.0, 1.0),
        (1.0, math.e),
        (2.302585, 10.0),  # ln(10) ≈ 2.302585
        (4.605170, 100.0)   # ln(100) ≈ 4.605170
    ])
    def test_known_values(self, loss, expected):
        """Test with known loss-perplexity pairs."""
        ppl = compute_perplexity(loss)
        assert abs(ppl - expected) < 1e-5


class TestConsoleLogger:
    """Test ConsoleLogger class."""
    
    def test_init(self):
        """Test logger initialization."""
        logger = ConsoleLogger(log_interval=50)
        assert logger.log_interval == 50
        assert logger.step == 0
        assert isinstance(logger.start_time, datetime)
    
    def test_step_increment(self):
        """Test that step increments on each log call."""
        logger = ConsoleLogger(log_interval=1)  # Log every step
        
        with patch('builtins.print') as mock_print:
            logger.log(1.5, 4.5, 0.001)
            assert logger.step == 1
            
            logger.log(1.4, 4.1, 0.0009)
            assert logger.step == 2
    
    def test_log_interval(self):
        """Test that logging only happens at specified intervals."""
        logger = ConsoleLogger(log_interval=3)
        
        with patch('builtins.print') as mock_print:
            # First two calls should not print
            logger.log(1.5, 4.5, 0.001)
            logger.log(1.4, 4.1, 0.0009)
            assert mock_print.call_count == 0
            
            # Third call should print
            logger.log(1.3, 3.7, 0.0008)
            assert mock_print.call_count == 1
    
    def test_log_format(self):
        """Test log message format."""
        logger = ConsoleLogger(log_interval=1)
        
        # Capture print output
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            logger.log(loss=2.5, perplexity=12.18, lr=0.001, tokens_per_sec=1000)
        
        output = captured_output.getvalue().strip()
        
        # Check that all expected components are in the output
        assert "Step" in output
        assert "Loss: 2.5000" in output
        assert "PPL: 12.18" in output
        assert "LR: 1.00e-03" in output
        assert "Tok/s: 1000" in output
        assert "Time:" in output
    
    def test_log_format_without_tokens_per_sec(self):
        """Test log format when tokens_per_sec is not provided."""
        logger = ConsoleLogger(log_interval=1)
        
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            logger.log(loss=2.5, perplexity=12.18, lr=0.001)
        
        output = captured_output.getvalue().strip()
        
        # Should not contain tokens per second
        assert "Tok/s:" not in output
        assert "Loss: 2.5000" in output
        assert "PPL: 12.18" in output
    
    def test_time_formatting(self):
        """Test time formatting in logs."""
        logger = ConsoleLogger(log_interval=1)
        
        # Mock start time to control elapsed time calculation
        mock_start = datetime(2023, 1, 1, 10, 0, 0)
        mock_now = datetime(2023, 1, 1, 11, 23, 45)  # 1 hour 23 minutes 45 seconds later
        
        logger.start_time = mock_start
        
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            with patch('utils.metrics.datetime') as mock_datetime:
                mock_datetime.now.return_value = mock_now
                logger.log(loss=2.5, perplexity=12.18, lr=0.001)
        
        output = captured_output.getvalue().strip()
        assert "Time: 01:23:45" in output
    
    def test_scientific_notation_lr(self):
        """Test that learning rates are displayed in scientific notation."""
        logger = ConsoleLogger(log_interval=1)
        
        test_cases = [
            (0.001, "1.00e-03"),
            (0.0001, "1.00e-04"),
            (0.1, "1.00e-01"),
            (1.0, "1.00e+00")
        ]
        
        for lr, expected in test_cases:
            captured_output = StringIO()
            
            with patch('sys.stdout', captured_output):
                logger.log(loss=2.5, perplexity=12.18, lr=lr)
            
            output = captured_output.getvalue().strip()
            assert expected in output, f"Expected {expected} for LR {lr}, got {output}"


class TestMetricsIntegration:
    """Integration tests for metrics components."""
    
    def test_moving_average_with_realistic_losses(self):
        """Test moving average with realistic training loss sequence."""
        ma = MovingAverage(window_size=10)
        
        # Simulate decreasing loss over training
        losses = [3.5, 3.2, 2.9, 2.8, 2.6, 2.5, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8]
        
        for i, loss in enumerate(losses):
            ma.update(loss)
            avg = ma.get()
            
            # Moving average should be reasonable
            assert 0 < avg < 10, f"Step {i}: unrealistic average {avg}"
            
            # After window fills up, average should be close to recent values
            if i >= 9:  # Window is full
                recent_avg = sum(losses[i-9:i+1]) / 10
                assert abs(avg - recent_avg) < 1e-10, f"Moving average calculation error at step {i}"
    
    def test_perplexity_with_training_progression(self):
        """Test perplexity calculation with typical training loss progression."""
        # Simulate loss decreasing from ~4 to ~1 during training
        losses = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]
        perplexities = [compute_perplexity(loss) for loss in losses]
        
        # Perplexity should decrease as loss decreases
        for i in range(1, len(perplexities)):
            assert perplexities[i] < perplexities[i-1], \
                f"Perplexity should decrease: {perplexities[i-1]} -> {perplexities[i]}"
        
        # Check some expected values
        assert abs(perplexities[0] - math.exp(4.0)) < 1e-6  # ~54.6
        assert abs(perplexities[-1] - math.exp(1.0)) < 1e-6  # ~2.7
    
    def test_console_logger_realistic_training(self):
        """Test console logger with realistic training scenario."""
        logger = ConsoleLogger(log_interval=2)
        
        # Simulate 5 training steps
        training_data = [
            (3.5, 30.0, 0.001, 1500),
            (3.2, 24.5, 0.0009, 1520),
            (3.0, 20.1, 0.0008, 1480),
            (2.8, 16.4, 0.0007, 1505),
            (2.6, 13.5, 0.0006, 1490)
        ]
        
        print_count = 0
        with patch('builtins.print') as mock_print:
            for i, (loss, ppl, lr, tps) in enumerate(training_data):
                logger.log(loss, ppl, lr, tps)
                
                # Should print on steps 2 and 4 (every 2nd step)
                if (i + 1) % 2 == 0:
                    print_count += 1
        
        assert mock_print.call_count == print_count
        assert logger.step == 5