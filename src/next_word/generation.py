from __future__ import annotations

from tensorflow.keras.preprocessing.sequence import pad_sequences


def generate_text(model, tokenizer, seed_text: str, max_sequence_len: int, next_words: int = 10) -> str:
    text = seed_text.strip()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
        predicted_index = int(model.predict(token_list, verbose=0).argmax(axis=-1)[0])

        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break

        if not next_word:
            break
        text = f"{text} {next_word}"
    return text

