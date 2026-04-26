from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM


def build_model(vocab_size: int, sequence_length: int, embedding_dim: int = 100, lstm_units: int = 150) -> Sequential:
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, input_length=sequence_length - 1),
            LSTM(lstm_units),
            Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

