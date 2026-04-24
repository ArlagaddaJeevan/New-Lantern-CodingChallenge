# Relevant Priors

POST `/predict` endpoint that labels each prior radiology study as relevant
or not for a given current study.

Stack: Python 3.12, FastAPI, LightGBM.

## Run locally

```
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
python tests/eval_public.py --url http://127.0.0.1:8000/predict
```

Or Docker:

```
docker build -t relevant-priors .
docker run --rm -p 8000:8000 relevant-priors
```

## Layout

```
app/            FastAPI app, feature extractor, predictor
train/          training pipeline + a small EDA script
models/         bundle.joblib (tf-idf vectorizers + lightgbm + threshold)
tests/          eval_public.py - scores the running API on the public split
```

`data/relevant_priors_public.json` is used for training and local scoring but
is not bundled into the Docker image.

## Deploy

`render.yaml` for Render (primary), `fly.toml` for Fly.io. Both build from the
same `Dockerfile`. Health check on `/healthz`.

## Numbers (public split, 5-fold GroupKFold on case_id)

| model         | accuracy | AUC    |
|---------------|---------:|-------:|
| rules         |   0.8338 |      - |
| logreg        |   0.9256 | 0.9511 |
| lightgbm      |   0.9535 | 0.9826 |

See `experiments.md` for details.
