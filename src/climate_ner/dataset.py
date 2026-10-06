"""
dataset.py — Parse ibm-research/Climate-Change-NER raw IOB lines
             into token-classification-ready HuggingFace Dataset rows.

Audit decisions baked in:
  Q1: Raw text lines, not pre-parsed → custom _parse_iob_lines() required.
  Q4: Pre-split at document level by IBM (unique hashes in -DOCSTART- lines)
      → use train/validation/test splits as-is, no re-splitting needed.
  Q5: O-label dominates (~83% of tokens) → compute_class_weights() provides
      inverse-frequency weights capped at 10x for weighted cross-entropy loss.
"""
from __future__ import annotations

import collections
from datasets import load_dataset, Dataset, DatasetDict

# ---------------------------------------------------------------------------
# Label registry — 27 IOB tags for 13 climate entity types + O
# Order matters: index 0 = "O", then B-/I- pairs alphabetically by type.
# ---------------------------------------------------------------------------
LABEL_NAMES: list[str] = [
    "O",
    "B-climate-assets",           "I-climate-assets",
    "B-climate-datasets",         "I-climate-datasets",
    "B-climate-greenhouse-gases", "I-climate-greenhouse-gases",
    "B-climate-hazards",          "I-climate-hazards",
    "B-climate-impacts",          "I-climate-impacts",
    "B-climate-mitigations",      "I-climate-mitigations",
    "B-climate-models",           "I-climate-models",
    "B-climate-nature",           "I-climate-nature",
    "B-climate-observations",     "I-climate-observations",
    "B-climate-organisms",        "I-climate-organisms",
    "B-climate-organizations",    "I-climate-organizations",
    "B-climate-problem-origins",  "I-climate-problem-origins",
    "B-climate-properties",       "I-climate-properties",
]

LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABEL_NAMES)}

# Dataset ID confirmed on HuggingFace Hub
HF_DATASET_ID = "ibm-research/Climate-Change-NER"


# ---------------------------------------------------------------------------
# IOB parser  (answers Q1)
# ---------------------------------------------------------------------------

def _parse_iob_lines(raw_rows: list[dict]) -> list[dict]:
    """Convert raw HF rows (each a single IOB text line) into sentence dicts.

    The HF dataset exposes every line of the CoNLL-style .txt file as a row
    with a single 'text' field.  This function:
      - Skips -DOCSTART- metadata lines
      - Groups token-label pairs into sentences at blank-line boundaries
      - Returns a list of {'tokens': list[str], 'ner_tags': list[int]}

    Args:
        raw_rows: List of dicts from a HF Dataset split, each {'text': str}.

    Returns:
        List of sentence dicts with integer-encoded ner_tags.
    """
    sentences: list[dict] = []
    current_tokens: list[str] = []
    current_labels: list[int] = []

    for row in raw_rows:
        line: str = row["text"].strip()

        # Skip document boundary markers (Q1 — these are metadata, not tokens)
        if line.startswith("-DOCSTART-"):
            continue

        # Blank line → sentence boundary; flush current buffer
        if not line:
            if current_tokens:
                sentences.append(
                    {"tokens": current_tokens, "ner_tags": current_labels}
                )
                current_tokens = []
                current_labels = []
            continue

        # Token–label pair: format is "token label" (label is last field)
        parts = line.split()
        if len(parts) < 2:
            # Malformed line — skip silently (none expected in this dataset)
            continue

        token = parts[0]
        label = parts[-1]

        # Guard: any unseen label falls back to O (shouldn't occur here)
        if label not in LABEL2ID:
            label = "O"

        current_tokens.append(token)
        current_labels.append(LABEL2ID[label])

    # Flush trailing sentence if file doesn't end with a blank line
    if current_tokens:
        sentences.append({"tokens": current_tokens, "ner_tags": current_labels})

    return sentences


# ---------------------------------------------------------------------------
# Public loader  (answers Q4 — use pre-existing splits)
# ---------------------------------------------------------------------------

def load_climate_ner(hf_dataset_id: str = HF_DATASET_ID) -> DatasetDict:
    """Download and parse ibm-research/Climate-Change-NER from the HF Hub.

    Returns a DatasetDict with train / validation / test splits.
    Each example has:
        tokens   : list[str]  — original whitespace-split tokens
        ner_tags : list[int]  — integer label IDs matching LABEL_NAMES

    The pre-existing document-level splits are preserved as-is (Q4 decision).
    """
    raw = load_dataset(hf_dataset_id)
    parsed: dict[str, Dataset] = {}
    for split in ("train", "validation", "test"):
        rows = list(raw[split])
        sentences = _parse_iob_lines(rows)
        parsed[split] = Dataset.from_list(sentences)
    return DatasetDict(parsed)


# ---------------------------------------------------------------------------
# Class-weight computation  (answers Q5)
# ---------------------------------------------------------------------------

def compute_class_weights(
    train_dataset: Dataset,
    cap: float = 10.0,
) -> list[float]:
    """Inverse-frequency class weights for weighted cross-entropy (Q5).

    O dominates ~83 % of tokens; rare tags like I-climate-greenhouse-gases
    appear only 10 times in training.  Weights are capped at `cap` × the
    minimum weight to prevent extreme gradient magnitudes for very rare tags.

    Args:
        train_dataset: The parsed training split.
        cap:           Maximum multiple of the smallest weight (default 10×).

    Returns:
        List of float weights, one per label in LABEL_NAMES order.
    """
    counts: collections.Counter = collections.Counter()
    for example in train_dataset:
        for tag_id in example["ner_tags"]:
            counts[tag_id] += 1

    total = sum(counts.values())
    n_classes = len(LABEL_NAMES)

    raw_weights: list[float] = []
    for i in range(n_classes):
        freq = counts.get(i, 1) / total          # floor at 1 to avoid div/0
        raw_weights.append(1.0 / (freq * n_classes))

    min_w = min(raw_weights)
    capped = [min(w, min_w * cap) for w in raw_weights]
    return capped
