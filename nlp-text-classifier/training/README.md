# Training Layer

This directory contains the complete training pipeline, faithfully extracted from the original Jupyter Notebook.

## Contents

| File | Purpose |
|---|---|
| `original_notebook.ipynb` | The original notebook — preserved as-is |
| `train.py` | Full training script (exact reproduction of notebook) |
| `evaluate.py` | Evaluation script — computes loss, accuracy, precision, recall, F1 |
| `inference_test.py` | Smoke tests verifying model predictions |
| `training_history.json` | Training curves (auto-generated after training) |
| `evaluation_report.json` | Evaluation metrics (auto-generated after evaluate.py) |

---

## ACTUAL IMPLEMENTATION DISCOVERED

```
DOCUMENTATION (notebook markdown):
  Classifies news articles into four categories (AG News)

ACTUAL CODE:
  Dataset:   IMDb Reviews (binary sentiment)
  Task:      Positive / Negative sentiment classification
  Classes:   2 (0 = negative, 1 = positive)
  Embedding: https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1 (20-dim)
  Model:     Hub Layer → Dense(16, relu) → Dense(1) [logit]
  Loss:      BinaryCrossentropy(from_logits=True)
  Optimizer: Adam
  Batch:     100
  Epochs:    25
  Test Acc:  94.636%   Test Loss: 0.2265
```

⚠️ The notebook's conclusion incorrectly describes the project as AG News / 4-class. The executable code uses IMDb / binary sentiment. The code is the source of truth.

---

## Dataset Split (as used in the original notebook)

```python
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=("test[:60%]", "test[60%:]", "test"),
    as_supervised=True,
)
```

- **Train:** 15,000 examples (60% of the IMDb test split)
- **Validation:** 10,000 examples (40% of the IMDb test split)
- **Test:** 25,000 examples (full IMDb test split)

Note: This is an unusual split — it uses only the `test` partition of IMDb, not the standard `train` split.

---

## Training Results (from original notebook)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|---------|--------|
| 1 | 0.6241 | 62.69% | 0.5308 | 69.15% |
| 5 | 0.2460 | 89.70% | 0.3048 | 86.58% |
| 10 | 0.1094 | 96.13% | 0.2979 | 87.74% |
| 15 | 0.0456 | 99.05% | 0.3593 | 87.23% |
| 20 | 0.0167 | 99.88% | 0.4632 | 87.06% |
| 25 | 0.0060 | 99.99% | 0.5591 | 86.60% |

**Observation:** Training accuracy reaches ~100% while validation accuracy plateaus at ~87–88% after epoch 10. This is a clear sign of **overfitting**. The model memorises the training set after about epoch 8–10. The test accuracy of 94.64% is measured on the same distribution as the training split (all from `test` partition).

---

## How to Train

```bash
cd training

# Install dependencies
pip install tensorflow tensorflow-hub tensorflow-datasets

# Train (downloads IMDb + Swivel embeddings on first run)
python train.py
```

This saves the model to:
```
backend/models/sentiment_model.keras
```

## How to Evaluate

```bash
python evaluate.py
```

## How to Smoke Test Inference

```bash
python inference_test.py
```
