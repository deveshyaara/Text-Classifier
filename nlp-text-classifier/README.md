# NLP Text Classifier

A production-style ML web application that classifies movie reviews as **positive** or **negative** using a TensorFlow + TensorFlow Hub model trained on the IMDb dataset.

> **Live Demo:** *(deploy and add URL here)*  
> **API Docs:** *(backend URL)/docs*  
> **GitHub:** *(your repo URL)*

---

## ⚠️ Notebook Discrepancy Notice

The original notebook's markdown conclusion incorrectly states:  
> *"classifies news articles into four categories"* (AG News)

**The actual executable code** loads IMDb movie reviews for binary sentiment classification.  
The code is the source of truth. All implementation here reflects the actual code.

---

## Overview

| Layer | Technology |
|---|---|
| ML Model | TensorFlow 2 + TensorFlow Hub |
| Embedding | `gnews-swivel-20dim` (20-dimensional) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js 14 + TypeScript + Tailwind CSS |
| Container | Docker |
| Deployment | Render (backend) + Vercel (frontend) |

---

## ACTUAL IMPLEMENTATION DISCOVERED

```
Dataset:   IMDb Reviews (via tensorflow_datasets)
Task:      Binary sentiment classification
Classes:   2  →  0 = negative,  1 = positive
Embedding: https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
           20-dimensional dense vectors
Model:     Hub Layer (20-dim) → Dense(16, relu) → Dense(1) [logit]
Loss:      BinaryCrossentropy(from_logits=True)
Optimizer: Adam
Batch:     100
Epochs:    25
Test Acc:  94.636%
Test Loss: 0.2265
```

---

## Features

- 🎯 Real-time sentiment inference via REST API
- 📊 Full probability breakdown (positive / negative)
- ⚡ Measured inference latency returned on every request
- 🧠 Model loaded once at startup — not per request
- 📖 Auto-generated API docs (`/docs`, `/redoc`)
- 🌗 Dark / light mode UI
- 📱 Fully responsive (mobile → desktop)
- 🐳 Docker-ready backend
- ♿ Accessible UI with keyboard navigation

---

## Architecture

```
USER INPUT (browser)
        ↓
   Next.js 14
  (TypeScript)
        ↓  HTTP POST /predict
   FastAPI API
  (Python/Uvicorn)
        ↓
  Preprocessing
  (strip whitespace)
        ↓
  TF Hub Swivel
  20-dim Embedding
  (trainable layer)
        ↓
  Dense(16, relu)
        ↓
   Dense(1)
  [raw logit]
        ↓
  sigmoid(logit)
  → probability
        ↓
   ≥ 0.5 → POSITIVE
   < 0.5 → NEGATIVE
```

---

## Machine Learning Model

### Dataset Split (as used in the original notebook)

```python
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=("test[:60%]", "test[60%:]", "test"),
    as_supervised=True,
)
```

- **Train:** 15,000 examples
- **Validation:** 10,000 examples
- **Test:** 25,000 examples

### Model Architecture

```
Layer                   Output Shape    Parameters
─────────────────────────────────────────────────
KerasLayer (Swivel)     (None, 20)      400,020
Dense (relu)            (None, 16)          336
Dense (logit)           (None,  1)           17
─────────────────────────────────────────────────
Total                                   400,373
```

### Inference

The output layer produces a **raw logit** (no sigmoid during training, because `BinaryCrossentropy(from_logits=True)` is used). At inference time:

```python
probability_positive = sigmoid(logit)
probability_negative = 1 - probability_positive
prediction = "positive" if probability_positive >= 0.5 else "negative"
```

---

## Evaluation

| Metric | Value |
|---|---|
| Test Accuracy | **94.636%** |
| Test Loss | **0.2265** |

*From the original notebook output. Precision/recall/F1 not computed in the notebook — use `training/evaluate.py` to compute them.*

### Training Curves (all 25 epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|---------|--------|
| 1 | 0.6241 | 62.69% | 0.5308 | 69.15% |
| 5 | 0.2460 | 89.70% | 0.3048 | 86.58% |
| 10 | 0.1094 | 96.13% | 0.2979 | 87.74% |
| 15 | 0.0456 | 99.05% | 0.3593 | 87.23% |
| 20 | 0.0167 | 99.88% | 0.4632 | 87.06% |
| 25 | 0.0060 | 99.99% | 0.5591 | 86.60% |

**⚠️ Overfitting observed:** Training accuracy reaches ~100% by epoch 25 while validation accuracy plateaus at ~87–88% from epoch 8 onwards, and validation loss *increases* after epoch 9. The model memorises the training data. The test accuracy (94.64%) is high because the test split overlaps the training distribution (all from the IMDb `test` partition).

---

## API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API metadata |
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Run sentiment inference |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

### API Example

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely brilliant."}'
```

**Response:**
```json
{
  "prediction": "positive",
  "confidence": 0.964123,
  "probabilities": {
    "positive": 0.964123,
    "negative": 0.035877
  },
  "inference_time_ms": 43.2
}
```

### Error responses

| Code | Meaning |
|------|---------|
| 422 | Empty text, whitespace-only, oversized input, invalid JSON |
| 503 | Model not loaded |
| 500 | Unexpected inference error |

---

## Local Development

### Prerequisites

- Python 3.10 or 3.11
- Node.js 18+
- (Optional) Docker

### 1. Train the model

```bash
cd training
pip install tensorflow tensorflow-hub tensorflow-datasets
python train.py
# Saves model to: backend/models/sentiment_model.keras
# First run downloads ~80 MB IMDb dataset + Swivel embeddings
```

### 2. Backend

```bash
cd backend

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Copy and edit environment file
cp .env.example .env

uvicorn app.main:app --reload
```

Open: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend

cp .env.example .env.local
# Edit .env.local: NEXT_PUBLIC_API_URL=http://localhost:8000

npm install
npm run dev
```

Open: http://localhost:3000

### 4. Run backend tests

```bash
cd backend
pytest tests/ -v
```

---

## Docker

### Build and run the backend

```bash
# From the project root:
docker build -t nlp-classifier-api ./backend

docker run \
  -p 8000:8000 \
  -e FRONTEND_URL=http://localhost:3000 \
  nlp-classifier-api
```

> **Note:** The model artifact (`backend/models/sentiment_model.keras`) must exist before building the Docker image. Generate it by running `training/train.py`.

---

## Deployment

### Backend → Render

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repository
3. Set **Root Directory** to `backend`
4. Set **Build Command:** `pip install -r requirements.txt`
5. Set **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - `FRONTEND_URL` = your Vercel frontend URL
   - `MODEL_PATH` = `./models/sentiment_model.keras`

> **Important:** The `backend/models/sentiment_model.keras` file must be committed to the repository (or downloaded at build time) for the API to work. If the file is too large for Git (>100 MB), use Git LFS or add a build step to download it.

### Frontend → Vercel

1. Import your GitHub repository at [vercel.com](https://vercel.com)
2. Set **Root Directory** to `frontend`
3. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = your Render backend URL

---

## Project Structure

```
nlp-text-classifier/
│
├── frontend/                   # Next.js 14 + TypeScript + Tailwind
│   ├── app/
│   │   ├── page.tsx            # Main classifier page
│   │   ├── layout.tsx          # Root layout + metadata
│   │   └── globals.css         # Global styles
│   ├── components/
│   │   ├── ClassifierForm.tsx  # Text input + submit
│   │   ├── PredictionResult.tsx# Result display component
│   │   ├── ModelInfo.tsx       # About the model section
│   │   ├── ArchDiagram.tsx     # Architecture visualization
│   │   └── TrainingCurves.tsx  # Training curve chart
│   ├── lib/
│   │   └── api.ts              # API client
│   ├── .env.example
│   └── package.json
│
├── backend/                    # FastAPI Python API
│   ├── app/
│   │   ├── main.py             # FastAPI app + routes
│   │   ├── model.py            # Model singleton + inference
│   │   ├── preprocessing.py    # Input preprocessing
│   │   ├── schemas.py          # Pydantic models
│   │   ├── config.py           # Settings from env vars
│   │   └── logging_config.py   # Logging setup
│   ├── models/                 # Model artifacts (git-ignored if large)
│   │   └── sentiment_model.keras
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_predict.py
│   │   └── test_validation.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── training/                   # ML training pipeline
│   ├── original_notebook.ipynb # Original notebook preserved
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── inference_test.py       # Inference smoke tests
│   └── README.md
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Limitations

1. **Overfitting:** The model overfits after ~epoch 8. Training accuracy ~100%, validation ~87%.
2. **Dataset split:** The notebook uses only the `test` partition of IMDb (not `train`), which is an unusual choice that limits training data to 15,000 examples.
3. **Embedding size:** The 20-dimensional Swivel embedding is small and fast but less expressive than larger embeddings.
4. **Cold start:** On serverless/free-tier deployments, the first request after idle may take 10–30 seconds while TensorFlow loads.
5. **TF Hub download:** At first startup, TF Hub downloads the Swivel model (~2 MB). Subsequent runs use the cache.
6. **Single worker:** TensorFlow is not thread-safe across processes. The API runs with 1 Uvicorn worker.

---

## Future Improvements

- [ ] Train on the full IMDb `train` split (25,000 examples)
- [ ] Add early stopping to prevent overfitting
- [ ] Try a larger embedding (e.g., `gnews-swivel-20dim-with-oov`)
- [ ] Add dropout regularization
- [ ] Compute full precision/recall/F1/ROC-AUC on test set
- [ ] Add request rate limiting
- [ ] Add model versioning
- [ ] Add Prometheus metrics endpoint

---

## How I would explain this project in an interview

> "I took my Jupyter Notebook — a binary sentiment classifier trained on IMDb movie reviews using TensorFlow and a pretrained TF Hub text embedding — and turned it into a live, deployable ML application. The model takes raw text, passes it through a 20-dimensional Google News Swivel embedding layer, then through a 16-unit dense layer, and outputs a single logit. I apply sigmoid to convert that to a probability. The backend is a FastAPI service that loads the model once at startup and handles all inference in memory. The frontend is Next.js with TypeScript and Tailwind, hitting the API in real time. I containerised the backend with Docker. One interesting engineering note: during training, I used `BinaryCrossentropy(from_logits=True)`, so at inference time I have to explicitly apply sigmoid — the model's raw output is NOT a probability."

---

## Common Interview Questions

**Why TensorFlow?**  
The original project was built with TensorFlow. Preserving the framework ensures the deployed model is identical to the trained model.

**Why TensorFlow Hub?**  
TF Hub provides pretrained text embeddings that can be fine-tuned end-to-end. This means the embedding weights adapt to the IMDb task rather than being frozen. It simplifies the pipeline — no separate tokenizer or vocabulary file needed.

**Why the gnews-swivel-20dim embedding?**  
It was chosen in the original notebook — likely for its simplicity and small size (20 dimensions, ~400K parameters). It's trainable, meaning it fine-tunes on IMDb. A larger embedding would likely improve accuracy.

**Why Dense(16)?**  
A single 16-unit hidden layer is a simple architecture sufficient for a binary classification task when using a pretrained embedding. It reduces the 20-dimensional embedding to 16 features before the output.

**Why Adam?**  
Adam is the default adaptive optimizer — combines momentum and RMSProp. Well-suited for sparse gradients in text tasks. No special reason not to use it here.

**Why BinaryCrossentropy?**  
Binary (not categorical) cross-entropy is correct for binary classification (one output, not two). It measures the log-likelihood of the correct label.

**Why `from_logits=True`?**  
The final Dense(1) layer has no activation, so it outputs a raw logit. Passing `from_logits=True` tells Keras to apply sigmoid internally during loss computation — this is numerically more stable than applying sigmoid first, then passing the result to the loss function.

**How is text converted to vectors?**  
The TF Hub Swivel layer tokenises the input string into words, looks up each word in a pretrained vocabulary trained on Google News, and averages the word vectors to produce a single 20-dimensional sentence vector.

**How does inference work?**  
Raw string → TF Hub layer → 20-dim vector → Dense(16) → Dense(1) logit → sigmoid → probability → threshold at 0.5 → label.

**How is the model deployed?**  
Saved as `.keras` format after training. Loaded into a FastAPI application using `tf.keras.models.load_model()`. The model is kept in memory for the lifetime of the process.

**How did you prevent model reloading?**  
Using a module-level singleton (`_model`) populated once in a FastAPI `lifespan` startup handler. All requests share the same model instance.

**What does the API do?**  
Three endpoints: `GET /` for metadata, `GET /health` for liveness + model status, `POST /predict` for inference. The predict endpoint validates input with Pydantic, preprocesses, runs inference, applies sigmoid, and returns the label, confidence, full probability breakdown, and measured latency.

**How would you improve the model?**  
(1) Use the full training split. (2) Add early stopping at epoch 8–9. (3) Try a larger embedding. (4) Add dropout. (5) Tune learning rate. (6) Evaluate precision/recall/F1.

**What indicates overfitting?**  
Training accuracy reaches ~100% by epoch 25. Validation accuracy peaks at ~87.77% at epoch 8 and then *decreases*. Validation loss *increases* monotonically from epoch 9 onwards. This is textbook overfitting — the model memorises training examples rather than learning generalisable features.

**How would you monitor this model in production?**  
Log prediction distribution (% positive over time). Alert on distribution shift. Track latency percentiles (p50, p95, p99). Log input length distribution. Monitor error rates. Periodically re-evaluate on labelled samples.

---

## Resume / Portfolio Description

```
NLP Text Classifier | TensorFlow, TensorFlow Hub, FastAPI, Next.js

End-to-end ML web application: binary sentiment classifier trained on 
IMDb reviews using a fine-tuned TF Hub Swivel embedding (94.6% test accuracy). 
FastAPI backend loads the TensorFlow model once at startup and serves 
real-time predictions via REST API. Next.js/TypeScript frontend displays 
predictions, confidence scores, and actual inference latency.

GitHub | Live Demo | API Docs
```
