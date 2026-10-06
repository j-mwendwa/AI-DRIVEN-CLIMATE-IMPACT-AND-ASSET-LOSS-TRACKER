"""
train.py — Fine-tune nasa-impact/nasa-smd-ibm-v0.1 (INDUS / RoBERTa-base)
           for token classification on ibm-research/Climate-Change-NER.

Run locally (CPU smoke-test):
    python -m climate_ner.train --epochs 1 --batch-size 4

Run in Google Colab (T4 GPU, recommended):
    python -m climate_ner.train \\
        --output-dir /content/climate-ner-indus \\
        --epochs 5 \\
        --batch-size 16 \\
        --lr 2e-5

Key design decisions:
  - RoBERTa BPE tokenizer splits words into subwords. Labels are aligned
    by assigning the original label to the FIRST subword of each word, and
    -100 (ignored by PyTorch loss) to all subsequent subword pieces and to
    special tokens ([CLS], [SEP]).  This is the standard HF NER approach.

  - Class-weighted cross-entropy (Q5 decision) is applied via a custom
    WeightedLossTrainer subclass, using weights from dataset.compute_class_weights().

  - fp16 mixed-precision is auto-enabled when CUDA is available (Colab T4/A100).

  - Evaluation metric is seqeval span-level F1, NOT per-token accuracy
    (accuracy is misleading when O dominates ~83% of tokens).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import evaluate as hf_evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from climate_ner.dataset import (
    LABEL_NAMES,
    LABEL2ID,
    ID2LABEL,
    load_climate_ner,
    compute_class_weights,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "nasa-impact/nasa-smd-ibm-v0.1"  # RoBERTa-base, 768-hidden, 12-layer
MAX_LENGTH = 512   # RoBERTa hard limit; climate abstracts fit comfortably


# ---------------------------------------------------------------------------
# Step 1 — Tokenize words into subwords and align NER labels
# ---------------------------------------------------------------------------

def tokenize_and_align_labels(
    examples: dict,
    tokenizer: AutoTokenizer,
) -> dict:
    """Tokenize a batch of word-list examples and align IOB labels to subwords.

    RoBERTa's BPE tokenizer splits some words into multiple subword pieces.
    Strategy:
      - first subword of a word  → inherits the word's integer label
      - subsequent subwords      → -100  (ignored by CrossEntropyLoss)
      - special tokens (CLS/SEP) → -100

    Args:
        examples: Batch dict with keys 'tokens' (list[list[str]])
                  and 'ner_tags' (list[list[int]]).
        tokenizer: Loaded RobertaTokenizer (add_prefix_space=True).

    Returns:
        Tokenizer output dict with an additional 'labels' key.
    """
    # is_split_into_words=True tells the tokenizer we're passing pre-split tokens
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,   # DataCollatorForTokenClassification pads dynamically
    )

    all_labels: list[list[int]] = []

    for batch_idx, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_idx)
        aligned: list[int] = []
        prev_word_id: int | None = None

        for word_id in word_ids:
            if word_id is None:
                # Special token ([CLS], [SEP], padding) → ignore
                aligned.append(-100)
            elif word_id != prev_word_id:
                # First subword of a new word → assign real label
                aligned.append(word_labels[word_id])
            else:
                # Continuation subword → ignore
                aligned.append(-100)
            prev_word_id = word_id

        all_labels.append(aligned)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# ---------------------------------------------------------------------------
# Step 2 — Custom Trainer with class-weighted loss (Q5 decision)
# ---------------------------------------------------------------------------

class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies per-class weight to CrossEntropyLoss.

    Addresses the severe O-class dominance (~83% of tokens) identified in
    the dataset audit (Q5).  Weights are pre-computed via
    dataset.compute_class_weights() and passed at construction time.
    """

    def __init__(
        self,
        *args,
        class_weights: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._class_weights = torch.tensor(
                class_weights, dtype=torch.float32, device=device
            )
        else:
            self._class_weights = None

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch, seq_len, num_labels)

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self._class_weights,
            ignore_index=-100,    # skip subword continuations and specials
        )
        # Flatten batch×seq_len for CrossEntropyLoss
        loss = loss_fn(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Step 3 — seqeval metrics (span-level, not token accuracy)
# ---------------------------------------------------------------------------

def build_compute_metrics(label_names: list[str]):
    """Return a compute_metrics function wired to seqeval for NER spans."""
    seqeval = hf_evaluate.load("seqeval")

    def compute_metrics(eval_pred) -> dict:
        raw_preds, raw_labels = eval_pred
        # Argmax over logits → integer label IDs
        pred_ids = np.argmax(raw_preds, axis=2)

        true_label_seqs: list[list[str]] = []
        pred_label_seqs: list[list[str]] = []

        for pred_seq, label_seq in zip(pred_ids, raw_labels):
            true_sent, pred_sent = [], []
            for p_id, l_id in zip(pred_seq, label_seq):
                if l_id == -100:
                    # Skip ignored subword positions
                    continue
                true_sent.append(label_names[l_id])
                pred_sent.append(label_names[p_id])
            true_label_seqs.append(true_sent)
            pred_label_seqs.append(pred_sent)

        results = seqeval.compute(
            predictions=pred_label_seqs,
            references=true_label_seqs,
        )
        return {
            "precision": results["overall_precision"],
            "recall":    results["overall_recall"],
            "f1":        results["overall_f1"],
            "accuracy":  results["overall_accuracy"],
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Step 4 — Main training entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    use_gpu = torch.cuda.is_available()
    device_label = "GPU (CUDA)" if use_gpu else "CPU"
    print(f"\n{'='*60}")
    print(f"  Climate NER Fine-Tuning")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Device     : {device_label}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Epochs     : {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Load & parse dataset ──────────────────────────────────────────────
    print("→ [1/5] Loading and parsing dataset...")
    ds = load_climate_ner()
    print(
        f"   train={len(ds['train'])} sents | "
        f"val={len(ds['validation'])} sents | "
        f"test={len(ds['test'])} sents"
    )

    # ── Load tokenizer ────────────────────────────────────────────────────
    print("→ [2/5] Loading tokenizer...")
    # add_prefix_space=True is required for RoBERTa when is_split_into_words=True
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)

    # ── Tokenize all splits ───────────────────────────────────────────────
    print("→ [3/5] Tokenizing dataset (subword alignment)...")
    tokenized_ds = ds.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer),
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenizing",
    )

    # ── Load model ────────────────────────────────────────────────────────
    print("→ [4/5] Loading model and attaching classification head...")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # ignore_mismatched_sizes replaces the MLM head with a fresh Linear(768→27)
        ignore_mismatched_sizes=True,
    )

    # ── Class weights ─────────────────────────────────────────────────────
    print("→ [5/5] Computing class weights for weighted loss...")
    class_weights = compute_class_weights(ds["train"], cap=10.0)
    print(f"   O weight={class_weights[0]:.4f}  |  "
          f"max weight={max(class_weights):.4f}  |  "
          f"n_classes={len(class_weights)}")

    # ── TrainingArguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Evaluate and checkpoint once per epoch
        eval_strategy="epoch",
        save_strategy="epoch",

        # Optimiser
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,       # 10% of steps as linear warmup

        # Batch sizes
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,

        # Schedule
        num_train_epochs=args.epochs,

        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Mixed-precision: auto-enable on GPU, no-op on CPU
        fp16=use_gpu,

        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=25,

        # Disable push-to-hub (user will do this manually if desired)
        push_to_hub=False,

        # Disable third-party trackers (clean Colab output)
        report_to="none",
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = build_compute_metrics(LABEL_NAMES)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n─── Starting Training ───────────────────────────────────────────")
    train_result = trainer.train()
    print("\n─── Training Complete ───────────────────────────────────────────")
    print(f"   Runtime : {train_result.metrics.get('train_runtime', 0):.1f}s")
    print(f"   Train Loss : {train_result.metrics.get('train_loss', 0):.4f}")

    # ── Save best model ───────────────────────────────────────────────────
    print(f"\n→ Saving best model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── Final validation metrics ──────────────────────────────────────────
    print("\n→ Final validation metrics:")
    val_metrics = trainer.evaluate(tokenized_ds["validation"])
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")

    print(f"\n✓ Done. Model saved to: {args.output_dir}")
    print("  Next step: python -m climate_ner.evaluate --model-dir", args.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune INDUS (nasa-smd-ibm-v0.1) for Climate NER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="models/climate-ner-indus",
        help="Directory to save checkpoints and final model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device train batch size (use 4 for CPU smoke-test)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    main(parser.parse_args())
