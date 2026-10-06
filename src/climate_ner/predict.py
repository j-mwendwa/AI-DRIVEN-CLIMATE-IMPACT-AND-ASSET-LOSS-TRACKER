"""
predict.py — Standalone inference: extract climate entity spans from raw text.

Usage (command-line demo):
    python -m climate_ner.predict

Usage (as a library):
    from climate_ner.predict import ClimateNER
    ner = ClimateNER("models/climate-ner-indus")
    entities = ner.extract("Flash floods damaged maize crops and local bridges.")
    # Returns list of dicts: {entity_group, word, score, start, end}
"""
from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)


class ClimateNER:
    """Inference wrapper around the fine-tuned Climate NER model.

    Loads the saved model once, then exposes an ``extract`` method that
    accepts a raw text string and returns a list of structured entity
    spans — ready for downstream Neo4j ingestion or LLM risk analysis.

    Args:
        model_dir: Path to the directory produced by train.py (or a HF Hub
                   model ID if the model was pushed there).

    Example:
        >>> ner = ClimateNER("models/climate-ner-indus")
        >>> ner.extract(
        ...     "Intense El Niño rainfall triggered flash floods in lower regions, "
        ...     "damaging maize crops and washing out local bridges."
        ... )
        [
            {'entity_group': 'climate-hazards', 'word': 'El Niño rainfall ...', ...},
            {'entity_group': 'climate-assets',  'word': 'maize crops', ...},
            {'entity_group': 'climate-assets',  'word': 'bridges', ...},
        ]
    """

    def __init__(self, model_dir: str = "models/climate-ner-indus") -> None:
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)

        # aggregation_strategy="simple" automatically merges consecutive
        # B-/I- tokens of the same type into a single span — no post-processing needed.
        self._pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )

    def extract(self, text: str) -> list[dict]:
        """Extract climate entity spans from raw text.

        Args:
            text: A raw input sentence or paragraph.

        Returns:
            List of entity dicts, each containing:
              - entity_group : str   — climate category (e.g. 'climate-hazards')
              - word         : str   — the extracted surface form
              - score        : float — model confidence (0–1)
              - start        : int   — character start offset in `text`
              - end          : int   — character end offset in `text`
        """
        raw = self._pipe(text)
        # Clean up the output: strip B-/I- prefixes if present (aggregation
        # should handle this, but guard for edge cases)
        cleaned = []
        for ent in raw:
            group = ent["entity_group"]
            if group.startswith(("B-", "I-")):
                group = group[2:]
            cleaned.append({
                "entity_group": group,
                "word":  ent["word"],
                "score": round(float(ent["score"]), 4),
                "start": ent["start"],
                "end":   ent["end"],
            })
        return cleaned


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Climate NER on a demo sentence")
    parser.add_argument(
        "--model-dir",
        default="models/climate-ner-indus",
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--text",
        default=(
            "Intense El Niño rainfall triggered flash floods in lower regions, "
            "damaging maize crops and washing out local bridges."
        ),
        help="Input sentence to tag",
    )
    args = parser.parse_args()

    print(f"\nInput: {args.text}\n")
    ner = ClimateNER(args.model_dir)
    entities = ner.extract(args.text)

    if not entities:
        print("(No entities detected)")
    else:
        print(f"{'Entity Type':<30} {'Text':<35} {'Score':>6}")
        print("-" * 73)
        for ent in entities:
            print(
                f"{ent['entity_group']:<30} "
                f"'{ent['word']}'".ljust(35) +
                f" {ent['score']:>6.3f}"
            )
