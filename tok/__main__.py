"""BPE tokenizer CLI."""

import argparse
import sys
from pathlib import Path
from .bpe import BPETokenizer, train_bpe


def main():
    parser = argparse.ArgumentParser(description="BPE Tokenizer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train BPE tokenizer")
    train_parser.add_argument(
        "--input", type=str, default="data/tiny.txt", help="Input text file"
    )
    train_parser.add_argument(
        "--output", type=str, default="tok/bpe.json", help="Output tokenizer file"
    )
    train_parser.add_argument("--vocab", type=int, default=1000, help="Vocabulary size")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text")
    encode_parser.add_argument(
        "--tokenizer", type=str, default="tok/bpe.json", help="Tokenizer file"
    )
    encode_parser.add_argument("--text", type=str, help="Text to encode")
    encode_parser.add_argument("--file", type=str, help="File to encode")

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode tokens")
    decode_parser.add_argument(
        "--tokenizer", type=str, default="tok/bpe.json", help="Tokenizer file"
    )
    decode_parser.add_argument(
        "tokens", nargs="+", type=int, help="Token IDs to decode"
    )

    args = parser.parse_args()

    if args.command == "train":
        print(f"Training BPE tokenizer with vocab size {args.vocab}...")
        tokenizer = train_bpe(
            Path(args.input), Path(args.output), vocab_size=args.vocab
        )
        print(f"✓ Tokenizer saved to {args.output}")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        print(f"  Number of merges: {len(tokenizer.merges)}")

    elif args.command == "encode":
        tokenizer = BPETokenizer()
        tokenizer.load(Path(args.tokenizer))

        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print("Please provide --text or --file")
            sys.exit(1)

        tokens = tokenizer.encode(text)
        print("Tokens:", tokens)
        print(f"Length: {len(tokens)} tokens")

    elif args.command == "decode":
        tokenizer = BPETokenizer()
        tokenizer.load(Path(args.tokenizer))

        text = tokenizer.decode(args.tokens)
        print("Decoded:", text)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
