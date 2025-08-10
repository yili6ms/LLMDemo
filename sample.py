"""Text generation script."""

import argparse
from pathlib import Path

import torch

from model.gpt import TinyGPT
from tok.bpe import BPETokenizer


def sample(
    model: TinyGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
) -> str:
    """Generate text from prompt."""
    if device is None:
        device = torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    # Generate
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Decode output
    output_tokens = output_ids[0].tolist()
    generated_text = tokenizer.decode(output_tokens)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with TinyGPT")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--model", type=str, default="checkpoints/best.pt", help="Model checkpoint"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tok/bpe.json", help="Tokenizer path"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(Path(args.tokenizer))

    # Load checkpoint
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Create model
    model = TinyGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=0.0,  # No dropout during inference
        max_seq_len=config["max_seq_len"],
        tie_weights=config.get("tie_weights", False),
        use_rope=config.get("use_rope", False),
        use_flash=config.get("use_flash", True),
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded model from step {checkpoint['step']} "
        f"(val_loss: {checkpoint['val_loss']:.4f})"
    )
    print(f"Prompt: {args.prompt}")
    print("-" * 50)

    # Generate text
    generated = sample(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        device=device,
    )

    print(generated)


if __name__ == "__main__":
    main()
