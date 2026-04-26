from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json


def load_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No text found in {path}")
    return text


def build_tokenizer(text: str) -> Tokenizer:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    return tokenizer


def create_training_sequences(text: str, tokenizer: Tokenizer) -> tuple[np.ndarray, np.ndarray, int]:
    input_sequences: list[list[int]] = []
    for line in text.splitlines():
        token_list = tokenizer.texts_to_sequences([line])[0]
        for index in range(1, len(token_list)):
            input_sequences.append(token_list[: index + 1])

    if not input_sequences:
        raise ValueError("Not enough tokenized text to create training sequences")

    max_sequence_len = max(len(seq) for seq in input_sequences)
    padded = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))
    predictors = padded[:, :-1]
    labels = padded[:, -1]
    return predictors, labels, max_sequence_len


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    Path(path).write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    return tokenizer_from_json(Path(path).read_text(encoding="utf-8"))


def save_config(path: str | Path, config: dict) -> None:
    Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

