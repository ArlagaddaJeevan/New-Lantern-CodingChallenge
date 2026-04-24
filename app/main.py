from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.predictor import Predictor
from app.schemas import PredictRequest, PredictResponse, Prediction

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOG = logging.getLogger("api")

BUNDLE_PATH = Path(
    os.environ.get("MODEL_BUNDLE",
                   str(Path(__file__).resolve().parent.parent / "models" / "bundle.joblib"))
)

app = FastAPI(title="relevant-priors", version="1.0")
_predictor: Predictor | None = None


@app.on_event("startup")
def _load():
    global _predictor
    LOG.info("loading bundle from %s", BUNDLE_PATH)
    _predictor = Predictor(BUNDLE_PATH)
    LOG.info("ready")


@app.get("/")
def root():
    return {"status": "ok", "service": "relevant-priors", "version": app.version}


@app.get("/healthz")
def health():
    return {
        "status": "ok" if _predictor else "loading",
        "threshold": _predictor.threshold if _predictor else None,
        "meta": _predictor.meta if _predictor else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    t0 = time.time()
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]

    assert _predictor is not None, "predictor not ready"

    case_count = len(req.cases)
    prior_count = sum(len(c.prior_studies) for c in req.cases)
    LOG.info("rid=%s cases=%d priors=%d", rid, case_count, prior_count)

    predictions: list[Prediction] = []
    for c in req.cases:
        flags = _predictor.predict_case(
            c.current_study.model_dump(),
            [p.model_dump() for p in c.prior_studies],
        )
        for p, is_rel in zip(c.prior_studies, flags):
            predictions.append(Prediction(
                case_id=c.case_id,
                study_id=p.study_id,
                predicted_is_relevant=is_rel,
            ))

    dt = (time.time() - t0) * 1000
    LOG.info("rid=%s predictions=%d elapsed_ms=%.1f", rid, len(predictions), dt)
    return PredictResponse(predictions=predictions)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    LOG.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )
