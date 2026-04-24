from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

import joblib
import numpy as np

from app.features import FeatureExtractor, Study

LOG = logging.getLogger("predictor")


class Predictor:
    def __init__(self, bundle_path: Path | str):
        bundle = joblib.load(bundle_path)
        self.char_vec = bundle["char_vec"]
        self.word_vec = bundle["word_vec"]
        self.booster = bundle["booster"]
        self.threshold: float = float(bundle.get("threshold", 0.5))
        self.feature_names = bundle["feature_names"]
        self.meta = {
            k: bundle.get(k)
            for k in ("trained_rows", "trained_cases", "oof_accuracy",
                     "oof_auc", "rules_accuracy", "logreg_accuracy")
        }
        # cache on (norm_current, norm_prior, days) so retries are free
        self._cache: dict[tuple[str, str, int], float] = {}
        self._cache_lock = Lock()
        self._extractor = FeatureExtractor()
        self._extractor.char_vec = self.char_vec
        self._extractor.word_vec = self.word_vec
        self._extractor._fitted = True  # noqa: SLF001
        LOG.info("predictor loaded: threshold=%.3f meta=%s",
                 self.threshold, self.meta)

    def predict_case(self, current_dict: dict, priors: list[dict]) -> list[bool]:
        if not priors:
            return []
        cur = Study.from_dict(current_dict)
        pri_studies = [Study.from_dict(p) for p in priors]

        out: list[bool | None] = [None] * len(pri_studies)
        to_score_idx: list[int] = []
        for i, p in enumerate(pri_studies):
            days = 0
            if cur.date_parsed and p.date_parsed:
                days = max((cur.date_parsed - p.date_parsed).days, 0)
            key = (cur.norm, p.norm, days)
            with self._cache_lock:
                hit = self._cache.get(key)
            if hit is not None:
                out[i] = bool(hit >= self.threshold)
            else:
                to_score_idx.append(i)

        if to_score_idx:
            X = self._extractor.featurize_case(cur, pri_studies)
            proba = self.booster.predict(X)
            for i, prob in enumerate(proba):
                p = pri_studies[i]
                days = 0
                if cur.date_parsed and p.date_parsed:
                    days = max((cur.date_parsed - p.date_parsed).days, 0)
                key = (cur.norm, p.norm, days)
                with self._cache_lock:
                    self._cache[key] = float(prob)
                if i in to_score_idx:
                    out[i] = bool(prob >= self.threshold)

        return [bool(x) for x in out]
