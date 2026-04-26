from __future__ import annotations

import unittest

from src.next_word.data import build_tokenizer, create_training_sequences


class DataPipelineTests(unittest.TestCase):
    def test_sequence_generation_returns_predictors_and_labels(self) -> None:
        text = "hello world\nhello there world"
        tokenizer = build_tokenizer(text)
        predictors, labels, max_sequence_len = create_training_sequences(text, tokenizer)

        self.assertGreater(len(predictors), 0)
        self.assertEqual(len(predictors), len(labels))
        self.assertGreaterEqual(max_sequence_len, 2)


if __name__ == "__main__":
    unittest.main()

