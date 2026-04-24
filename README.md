# Relevant Priors — HTTP API

FastAPI service that, given a current radiology exam + a list of priors for the same patient, returns a boolean `predicted_is_relevant` for every prior.

## Stack

- Python 3.12 + FastAPI
- Gradient-boosted classifier (LightGBM) over ~100 engineered features per (current, prior) pair: TF-IDF char + word cosine similarity, modality / anatomy / laterality / contrast match, time delta, recency rank, and a handful of length / position features.
- Vectorizers + booster are serialized into `models/bundle.joblib`. Inference loads the bundle once at startup and scores each case in one batched call (no per-prior API calls).

## Layout

```
app/                FastAPI app + shared feature extractor + predictor
train/              Dataset loader, EDA, training pipeline
models/bundle.joblib  trained artifact (TF-IDF vectorizers + LightGBM + threshold)
data/               public JSON (not shipped in the Docker image)
tests/eval_public.py  end-to-end accuracy probe against the running API
```

## Run locally

```bash
pip install -r requirements.txt
python train/train.py             # retrain if you want; bundle is checked in
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Quick probe:
```bash
curl -s http://127.0.0.1:8000/healthz
python tests/eval_public.py --url http://127.0.0.1:8000/predict
```

## Deploy

### Render (primary)
1. Push this folder to a GitHub repo.
2. In Render → **New → Blueprint**, select the repo. `render.yaml` wires up a Docker web service with `/healthz` health checks.
3. Render returns a URL like `https://relevant-priors.onrender.com` — that is the endpoint for submission.

### Fly.io (backup)
```bash
fly launch --no-deploy --copy-config --name relevant-priors
fly deploy
fly status                 # URL shown at the bottom
```

## Contract

`POST /predict` — request/response schemas exactly match the challenge brief; see [app/schemas.py](app/schemas.py). For every prior in the request, the response carries a `{case_id, study_id, predicted_is_relevant}` tuple in the same order.

Every request is logged as `rid=<id> cases=N priors=M elapsed_ms=…` for evaluator-side debugging.

## Results (public split, 5-fold grouped CV by `case_id`)

| model | accuracy | AUC |
|-------|---------:|----:|
| rules-only baseline (exact-match OR modality+anatomy+5yr) | 0.8338 | — |
| logistic regression on engineered features | 0.9256 | 0.9511 |
| **LightGBM (shipped)** | **0.9535** | **0.9826** |

The full write-up and next-step ideas are in [experiments.md](experiments.md).
