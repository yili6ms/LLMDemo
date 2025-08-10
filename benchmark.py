"""Performance benchmarking script for TinyGPT."""

import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
import gc
import psutil
import os

from model.gpt import TinyGPT
from tok.bpe import BPETokenizer


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    result = {
        "ram_mb": memory_info.rss / 1024 / 1024,  # Resident set size
        "vram_mb": 0,
    }

    if torch.cuda.is_available():
        result["vram_mb"] = torch.cuda.memory_allocated() / 1024 / 1024

    return result


def benchmark_forward_pass(
    model, batch_size=8, seq_len=128, num_iterations=100, device="cuda"
):
    """Benchmark forward pass performance."""
    model.eval()

    # Generate random input
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)

    # Sync for accurate timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            logits, _ = model(input_ids)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = batch_size * seq_len * num_iterations
    tokens_per_second = total_tokens / total_time

    return {
        "tokens_per_second": tokens_per_second,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "avg_time_per_batch": total_time / num_iterations,
    }


def benchmark_training_step(
    model, batch_size=8, seq_len=128, num_iterations=50, device="cuda"
):
    """Benchmark training step performance."""
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Generate random input and targets
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        logits, loss = model(input_ids, targets=targets)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()

    for _ in range(num_iterations):
        optimizer.zero_grad()
        logits, loss = model(input_ids, targets=targets)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = batch_size * seq_len * num_iterations
    tokens_per_second = total_tokens / total_time

    return {
        "tokens_per_second": tokens_per_second,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "avg_time_per_step": total_time / num_iterations,
    }


def benchmark_generation(
    model, tokenizer, prompt="The quick brown fox", max_tokens=100, device="cuda"
):
    """Benchmark text generation performance."""
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    generation_time = end_time - start_time
    generated_tokens = output_ids.size(1) - input_ids.size(1)
    tokens_per_second = generated_tokens / generation_time

    # Decode output
    generated_text = tokenizer.decode(output_ids[0].tolist())

    return {
        "tokens_per_second": tokens_per_second,
        "generation_time": generation_time,
        "generated_tokens": generated_tokens,
        "prompt_tokens": input_ids.size(1),
        "total_tokens": output_ids.size(1),
        "generated_text": generated_text,
    }


def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark."""

    # Test configurations
    configs = [
        {"d_model": 64, "n_layers": 2, "n_heads": 2, "vocab_size": 500},
        {"d_model": 128, "n_layers": 4, "n_heads": 4, "vocab_size": 1000},
        {"d_model": 256, "n_layers": 6, "n_heads": 8, "vocab_size": 1000},
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on device: {device}")

    # Create tokenizer
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train("The quick brown fox jumps over the lazy dog. " * 100)

    results = {
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "configs": [],
    }

    for config in configs:
        print(f"\nTesting config: {config}")

        # Create model
        model = TinyGPT(**config).to(device)
        num_params = sum(p.numel() for p in model.parameters())

        config_results = {
            "config": config,
            "num_parameters": num_params,
            "memory_before": get_memory_usage(),
        }

        try:
            # Forward pass benchmark
            print("  Benchmarking forward pass...")
            forward_results = benchmark_forward_pass(model, device=device)
            config_results["forward_pass"] = forward_results

            # Training step benchmark
            print("  Benchmarking training step...")
            training_results = benchmark_training_step(model, device=device)
            config_results["training_step"] = training_results

            # Generation benchmark
            print("  Benchmarking text generation...")
            generation_results = benchmark_generation(model, tokenizer, device=device)
            config_results["generation"] = generation_results

            config_results["memory_after"] = get_memory_usage()

            print(f"    Forward: {forward_results['tokens_per_second']:.0f} tok/s")
            print(f"    Training: {training_results['tokens_per_second']:.0f} tok/s")
            print(
                f"    Generation: {generation_results['tokens_per_second']:.1f} tok/s"
            )
            print(f"    Parameters: {num_params:,}")

        except Exception as e:
            print(f"    Error: {e}")
            config_results["error"] = str(e)

        results["configs"].append(config_results)

        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def benchmark_flash_attention():
    """Benchmark FlashAttention vs manual attention."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nBenchmarking attention implementations on {device}")

    config = {"d_model": 128, "n_layers": 4, "n_heads": 4, "vocab_size": 1000}

    results = {}

    for use_flash in [False, True]:
        flash_str = "FlashAttention" if use_flash else "Manual"
        print(f"  Testing {flash_str}...")

        try:
            model = TinyGPT(use_flash=use_flash, **config).to(device)
            forward_results = benchmark_forward_pass(model, device=device)

            results[flash_str.lower()] = {
                "tokens_per_second": forward_results["tokens_per_second"],
                "avg_time_per_batch": forward_results["avg_time_per_batch"],
            }

            print(f"    {flash_str}: {forward_results['tokens_per_second']:.0f} tok/s")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    {flash_str} error: {e}")
            results[flash_str.lower()] = {"error": str(e)}

    if "manual" in results and "flashattention" in results:
        if (
            "tokens_per_second" in results["flashattention"]
            and "tokens_per_second" in results["manual"]
        ):
            speedup = (
                results["flashattention"]["tokens_per_second"]
                / results["manual"]["tokens_per_second"]
            )
            print(f"  FlashAttention speedup: {speedup:.2f}x")
            results["speedup"] = speedup

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TinyGPT performance")
    parser.add_argument("--output", type=str, default="BENCH.md", help="Output file")
    parser.add_argument(
        "--quick", action="store_true", help="Quick benchmark with fewer iterations"
    )

    args = parser.parse_args()

    print("🚀 TinyGPT Performance Benchmark")
    print("=" * 50)

    # System info
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    memory_info = get_memory_usage()
    print(f"RAM: {memory_info['ram_mb']:.0f} MB")

    # Run benchmarks
    benchmark_results = run_comprehensive_benchmark()
    flash_results = benchmark_flash_attention()

    # Combine results
    full_results = {
        "benchmark_results": benchmark_results,
        "flash_attention_comparison": flash_results,
    }

    # Save detailed results to JSON
    json_output = Path(args.output).with_suffix(".json")
    with open(json_output, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    # Generate markdown report
    with open(args.output, "w") as f:
        f.write("# TinyGPT Performance Benchmark\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        if torch.cuda.is_available():
            f.write(f"**GPU:** {torch.cuda.get_device_name()}\n")
        f.write(f"**PyTorch Version:** {torch.__version__}\n\n")

        f.write("## Model Performance\n\n")
        f.write(
            "| Config | Parameters | Forward (tok/s) | Training (tok/s) | Generation (tok/s) |\n"
        )
        f.write(
            "|--------|------------|-----------------|------------------|--------------------|\n"
        )

        for config_result in benchmark_results["configs"]:
            if "error" not in config_result:
                config = config_result["config"]
                params = config_result["num_parameters"]
                forward_tps = config_result["forward_pass"]["tokens_per_second"]
                train_tps = config_result["training_step"]["tokens_per_second"]
                gen_tps = config_result["generation"]["tokens_per_second"]

                f.write(
                    f"| d{config['d_model']}_l{config['n_layers']} | {params:,} | {forward_tps:.0f} | {train_tps:.0f} | {gen_tps:.1f} |\n"
                )

        f.write("\n## FlashAttention Comparison\n\n")
        if "speedup" in flash_results:
            f.write(f"**Speedup:** {flash_results['speedup']:.2f}x\n\n")
            f.write("| Implementation | Tokens/sec |\n")
            f.write("|----------------|------------|\n")
            if "manual" in flash_results:
                f.write(
                    f"| Manual Attention | {flash_results['manual']['tokens_per_second']:.0f} |\n"
                )
            if "flashattention" in flash_results:
                f.write(
                    f"| FlashAttention | {flash_results['flashattention']['tokens_per_second']:.0f} |\n"
                )
        else:
            f.write("FlashAttention comparison not available.\n")

        f.write("\n---\n")
        f.write(f"\n*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")

    print(f"\n✅ Benchmark complete!")
    print(f"Results saved to: {args.output}")
    print(f"Detailed data saved to: {json_output}")


if __name__ == "__main__":
    main()
