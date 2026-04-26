from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow import keras

from src.next_word.data import build_tokenizer, create_training_sequences, load_text, save_config, save_tokenizer
from src.next_word.modeling import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a next-word prediction model")
    parser.add_argument("--input", required=True, help="Path to a UTF-8 text corpus")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model artifacts")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--lstm-units", type=int, default=150, help="LSTM hidden units")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = load_text(args.input)
    tokenizer = build_tokenizer(text)
    predictors, labels, max_sequence_len = create_training_sequences(text, tokenizer)

    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(
        vocab_size=vocab_size,
        sequence_length=max_sequence_len,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
    )

    history = model.fit(predictors, labels, epochs=args.epochs, verbose=1)

    model.save(output_dir / "next_word_model.keras")
    save_tokenizer(tokenizer, output_dir / "tokenizer.json")
    save_config(
        output_dir / "config.json",
        {
            "input_path": args.input,
            "epochs": args.epochs,
            "embedding_dim": args.embedding_dim,
            "lstm_units": args.lstm_units,
            "max_sequence_len": max_sequence_len,
            "vocab_size": vocab_size,
            "final_loss": float(history.history["loss"][-1]),
            "final_accuracy": float(history.history["accuracy"][-1]),
        },
    )

    print(f"Saved model artifacts to {output_dir}")


if __name__ == "__main__":
    main()

