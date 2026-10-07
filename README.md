# Climate NER — INDUS Fine-Tuning

Fine-tunes `nasa-impact/nasa-smd-ibm-v0.1` (INDUS, a domain-adapted RoBERTa-base)
on `ibm-research/Climate-Change-NER` for **27-class IOB token classification**
(13 climate entity types + O).

## Project Structure

```
CLIMATE-TECH/
├── src/climate_ner/
│   ├── dataset.py      # IOB parser + class weight computation
│   ├── train.py        # fine-tuning entry point (WeightedLossTrainer)
│   ├── ner_eval.py     # per-entity seqeval report on test set
│   └── predict.py      # ClimateNER inference wrapper
├── notebooks/
│   └── colab_train.ipynb   # Ready-to-run Colab notebook
├── models/             # fine-tuned model saved here (git-ignored)
├── requirements.txt
└── pyproject.toml
```

## Quick Start (Google Colab)

Open `notebooks/colab_train.ipynb` — it installs deps, clones the repo,
and runs full training end-to-end on a Colab T4 GPU.

## Local Smoke-Test (CPU — verify parser, no GPU needed)

```bash
pip install -e .
python -c "
from climate_ner.dataset import load_climate_ner, LABEL_NAMES
ds = load_climate_ner()
print('Train:', len(ds['train']), 'sentences')
ex = ds['train'][10]
print('Tokens:', ex['tokens'][:6])
print('Labels:', [LABEL_NAMES[i] for i in ex['ner_tags'][:6]])
"
```

## Training (Colab / GPU)

```bash
python -m climate_ner.train \
  --output-dir models/climate-ner-indus \
  --epochs 15 \
  --batch-size 16 \
  --lr 2e-5
```

## Evaluation

```bash
python -m climate_ner.ner_eval --model-dir models/climate-ner-indus
```

## Inference

```python
from climate_ner.predict import ClimateNER
ner = ClimateNER("models/climate-ner-indus")
print(ner.extract("Flash floods damaged maize crops and local bridges."))
```
# AI-DRIVEN-CLIMATE-IMPACT-AND-ASSET-LOSS-TRACKER
