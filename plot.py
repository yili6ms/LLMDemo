"""Plot training metrics from CSV logs."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils.csv_logger import CSVLogger
from typing import List, Dict, Any


def plot_training_metrics(log_path: Path, output_path: Path = None):
    """Plot training metrics from CSV log file."""
    
    # Read logs
    logger = CSVLogger(log_path)
    logs = logger.read_logs()
    
    if not logs:
        print(f"No logs found in {log_path}")
        return
    
    # Extract data
    steps = [log['step'] for log in logs if log.get('step') is not None]
    losses = [log['loss'] for log in logs if log.get('loss') is not None]
    perplexities = [log['perplexity'] for log in logs if log.get('perplexity') is not None]
    learning_rates = [log['learning_rate'] for log in logs if log.get('learning_rate') is not None]
    tokens_per_sec = [log['tokens_per_sec'] for log in logs if log.get('tokens_per_sec') is not None]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot 1: Loss
    if losses and steps:
        axes[0, 0].plot(steps[:len(losses)], losses, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add smoothed trend line
        if len(losses) > 10:
            window_size = max(1, len(losses) // 20)
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = steps[:len(smoothed_losses)]
            axes[0, 0].plot(smoothed_steps, smoothed_losses, 'r-', linewidth=3, alpha=0.7, label='Smoothed')
            axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Training Loss (No Data)')
    
    # Plot 2: Perplexity
    if perplexities and steps:
        axes[0, 1].plot(steps[:len(perplexities)], perplexities, 'g-', linewidth=2, alpha=0.8)
        axes[0, 1].set_title('Perplexity')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add smoothed trend line
        if len(perplexities) > 10:
            window_size = max(1, len(perplexities) // 20)
            smoothed_ppl = np.convolve(perplexities, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = steps[:len(smoothed_ppl)]
            axes[0, 1].plot(smoothed_steps, smoothed_ppl, 'r-', linewidth=3, alpha=0.7, label='Smoothed')
            axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No perplexity data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Perplexity (No Data)')
    
    # Plot 3: Learning Rate
    if learning_rates and steps:
        axes[1, 0].plot(steps[:len(learning_rates)], learning_rates, 'orange', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Rate (No Data)')
    
    # Plot 4: Throughput
    if tokens_per_sec and steps:
        axes[1, 1].plot(steps[:len(tokens_per_sec)], tokens_per_sec, 'purple', linewidth=2, alpha=0.8)
        axes[1, 1].set_title('Training Throughput')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Tokens/sec')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add average line
        if tokens_per_sec:
            avg_throughput = np.mean(tokens_per_sec)
            axes[1, 1].axhline(y=avg_throughput, color='red', linestyle='--', alpha=0.7, 
                              label=f'Avg: {avg_throughput:.0f}')
            axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No throughput data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Throughput (No Data)')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    if losses:
        print(f"Final Loss: {losses[-1]:.4f}")
        print(f"Best Loss: {min(losses):.4f}")
        print(f"Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    if perplexities:
        print(f"Final Perplexity: {perplexities[-1]:.2f}")
        print(f"Best Perplexity: {min(perplexities):.2f}")
    
    if tokens_per_sec:
        print(f"Average Throughput: {np.mean(tokens_per_sec):.0f} tokens/sec")
        print(f"Peak Throughput: {max(tokens_per_sec):.0f} tokens/sec")
    
    if steps:
        print(f"Total Steps: {max(steps):,}")


def compare_experiments(log_paths: List[Path], labels: List[str] = None, output_path: Path = None):
    """Compare multiple training runs."""
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(log_paths))]
    
    plt.figure(figsize=(15, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot 1: Loss comparison
    plt.subplot(1, 3, 1)
    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        logger = CSVLogger(log_path)
        logs = logger.read_logs()
        
        if logs:
            steps = [log['step'] for log in logs if log.get('step') is not None]
            losses = [log['loss'] for log in logs if log.get('loss') is not None]
            
            if losses and steps:
                color = colors[i % len(colors)]
                plt.plot(steps[:len(losses)], losses, color=color, linewidth=2, alpha=0.8, label=label)
    
    plt.title('Loss Comparison')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Perplexity comparison
    plt.subplot(1, 3, 2)
    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        logger = CSVLogger(log_path)
        logs = logger.read_logs()
        
        if logs:
            steps = [log['step'] for log in logs if log.get('step') is not None]
            perplexities = [log['perplexity'] for log in logs if log.get('perplexity') is not None]
            
            if perplexities and steps:
                color = colors[i % len(colors)]
                plt.plot(steps[:len(perplexities)], perplexities, color=color, linewidth=2, alpha=0.8, label=label)
    
    plt.title('Perplexity Comparison')
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate comparison
    plt.subplot(1, 3, 3)
    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        logger = CSVLogger(log_path)
        logs = logger.read_logs()
        
        if logs:
            steps = [log['step'] for log in logs if log.get('step') is not None]
            lrs = [log['learning_rate'] for log in logs if log.get('learning_rate') is not None]
            
            if lrs and steps:
                color = colors[i % len(colors)]
                plt.plot(steps[:len(lrs)], lrs, color=color, linewidth=2, alpha=0.8, label=label)
    
    plt.title('Learning Rate Comparison')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--log", type=str, required=True, help="Path to CSV log file")
    parser.add_argument("--output", type=str, default="runs/plot.png", help="Output plot path")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare multiple log files")
    parser.add_argument("--labels", type=str, nargs="+", help="Labels for comparison plots")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple runs
        log_paths = [Path(p) for p in args.compare]
        compare_experiments(log_paths, args.labels, Path(args.output) if args.output else None)
    else:
        # Single run plot
        plot_training_metrics(Path(args.log), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()