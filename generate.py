from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow import keras

from src.next_word.data import load_config, load_tokenizer
from src.next_word.generation import generate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained next-word model")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory containing saved model artifacts")
    parser.add_argument("--seed", required=True, help="Seed text prompt")
    parser.add_argument("--next-words", type=int, default=10, help="Number of tokens to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    model = keras.models.load_model(artifacts_dir / "next_word_model.keras")
    tokenizer = load_tokenizer(artifacts_dir / "tokenizer.json")
    config = load_config(artifacts_dir / "config.json")

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        seed_text=args.seed,
        max_sequence_len=int(config["max_sequence_len"]),
        next_words=args.next_words,
    )
    print(output)


if __name__ == "__main__":
    main()

