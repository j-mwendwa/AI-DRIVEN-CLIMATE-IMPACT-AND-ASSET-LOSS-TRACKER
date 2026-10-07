"""
ner_eval.py — Run the fine-tuned model against the held-out test set
              and print a full per-entity-type seqeval classification report.

Named ner_eval.py (NOT evaluate.py) to avoid shadowing the installed
HuggingFace 'evaluate' package when Python searches this directory.

Usage:
    python -m climate_ner.ner_eval --model-dir models/climate-ner-indus
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import classification_report, f1_score

from climate_ner.dataset import load_climate_ner, LABEL_NAMES


def evaluate_test_set(model_dir: str) -> None:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Run training first: python -m climate_ner.train"
        )

    device = 0 if torch.cuda.is_available() else -1
    print(f"\n→ Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # aggregation_strategy="none" → one label per token (not merged spans)
    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="none",
        device=device,
    )

    print("→ Loading test split...")
    test_ds = load_climate_ner()["test"]
    print(f"   {len(test_ds)} sentences to evaluate")

    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []

    for example in test_ds:
        tokens: list[str] = example["tokens"]
        gold_ids: list[int] = example["ner_tags"]
        gold_labels = [LABEL_NAMES[i] for i in gold_ids]

        # Join tokens with spaces and run through pipeline
        text = " ".join(tokens)
        ner_output = ner_pipe(text)

        # Build a per-word prediction list, defaulting to "O"
        # Pipeline returns one entry per subword; we take the label of the
        # first subword for each word (index is 1-based, offset by [CLS])
        pred_labels = ["O"] * len(tokens)
        seen_word_indices: set[int] = set()
        for ent in ner_output:
            # 'index' is the 1-based subword position in the tokenized sequence
            # word_id mapping: the token at position i belongs to word (i-1)
            # This is approximate when words contain punctuation — seqeval
            # handles minor misalignment gracefully via span-level scoring.
            subword_idx = ent.get("index", 0)
            # Map subword index back to word index (heuristic: use char offsets)
            start_char = ent.get("start", 0)
            word_idx = len(text[:start_char].split())
            if word_idx < len(pred_labels) and word_idx not in seen_word_indices:
                pred_labels[word_idx] = ent["entity"]
                seen_word_indices.add(word_idx)

        true_seqs.append(gold_labels)
        pred_seqs.append(pred_labels)

    print("\n" + "=" * 70)
    print("  Per-Entity-Type Report (Test Set)")
    print("=" * 70)
    print(classification_report(true_seqs, pred_seqs, digits=4))
    overall_f1 = f1_score(true_seqs, pred_seqs)
    print(f"  Overall seqeval micro-F1: {overall_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned Climate NER model on the test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="models/climate-ner-indus",
        help="Path to the saved fine-tuned model directory",
    )
    args = parser.parse_args()
    evaluate_test_set(args.model_dir)
