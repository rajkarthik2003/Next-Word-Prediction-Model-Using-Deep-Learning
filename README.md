# Next Word Prediction Using Deep Learning

Sequence modeling project that started as a notebook workflow and is now packaged as a small, runnable NLP project with training and generation scripts.

## Why This Repo Matters

This repository shows more than a one-off notebook:

- text preprocessing for language modeling
- reusable training and inference code
- model artifact saving
- command-line workflows for retraining and text generation
- lightweight test coverage for the data pipeline

It is still a foundational NLP project, but it now reads more like engineering work than coursework.

## Repository Structure

```text
Deep Learning Project.pptx
Deep Learning Report.pdf
Project.ipynb.txt
README.md
generate.py
requirements.txt
train.py
src/next_word/
tests/
```

## Model Design

The training flow is based on the original notebook approach:

- Keras tokenizer for vocabulary building
- n-gram style sequence generation
- pre-padding of token sequences
- embedding layer
- single LSTM layer
- softmax output over the vocabulary

## Run It

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train on a UTF-8 text file:

```bash
python train.py --input path/to/corpus.txt --output-dir artifacts --epochs 100
```

3. Generate text from a prompt:

```bash
python generate.py --artifacts-dir artifacts --seed "joe biden and" --next-words 8
```

## Artifacts Saved After Training

- `artifacts/next_word_model.keras`
- `artifacts/tokenizer.json`
- `artifacts/config.json`

## Notes On The Original Work

The repository still includes the original presentation, report, and exported notebook so the project history stays visible. The new scripts make the work easier to run, review, and extend.

## Extension Ideas

- compare LSTM output against transformer baselines
- add validation split and perplexity tracking
- support top-k or temperature-based sampling
- add experiment logging for multiple corpora

## Suggested Recruiter Framing

This project is best read as an early but legitimate NLP systems project that demonstrates:

- hands-on sequence modeling
- practical preprocessing knowledge
- ability to turn notebook logic into a reusable codebase

For my more production-oriented work, see:

- [fraud-mlops-system](https://github.com/rajkarthik2003/fraud-mlops-system)
- [grounded-llm-system](https://github.com/rajkarthik2003/grounded-llm-system)
- [EduvisionMVC](https://github.com/rajkarthik2003/EduvisionMVC)
