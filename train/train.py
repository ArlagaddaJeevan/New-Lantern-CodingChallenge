"""Train and serialize the relevance classifier.

Run:  python train/train.py

Output:  models/bundle.joblib  (holds TF-IDF vectorizers + LightGBM booster
                                + chosen decision threshold).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import FEATURE_NAMES, FeatureExtractor, Study  # noqa: E402

DATA = ROOT / "data" / "relevant_priors_public.json"
OUT = ROOT / "models" / "bundle.joblib"

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_public() -> tuple[list[dict], dict[tuple[str, str], bool]]:
    with DATA.open(encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload["cases"]
    labels = {(t["case_id"], t["study_id"]): bool(t["is_relevant_to_current"])
              for t in payload["truth"]}
    return cases, labels


def build_xy(cases, labels, extractor: FeatureExtractor):
    X_blocks, y, groups, meta = [], [], [], []
    for c in cases:
        cur = Study.from_dict(c["current_study"])
        priors = [Study.from_dict(p) for p in c["prior_studies"]]
        if not priors:
            continue
        X = extractor.featurize_case(cur, priors)
        for i, p in enumerate(priors):
            key = (c["case_id"], p.study_id)
            if key not in labels:
                continue
            X_blocks.append(X[i])
            y.append(int(labels[key]))
            groups.append(c["case_id"])
            meta.append(key)
    return np.vstack(X_blocks), np.asarray(y), np.asarray(groups), meta


# --------------------------------------------------------------------------- #
# Baselines / evaluation
# --------------------------------------------------------------------------- #


def rules_predict(X: np.ndarray) -> np.ndarray:
    """Simple rule: positive iff exact match OR (modality match AND anatomy match
    AND within 5 years). Uses feature indexes so the model-free baseline stays
    aligned with the numeric columns in app.features.FEATURE_NAMES."""
    exact = X[:, FEATURE_NAMES.index("exact_match")]
    mm = X[:, FEATURE_NAMES.index("mod_match")]
    am = X[:, FEATURE_NAMES.index("anat_match")]
    w5 = X[:, FEATURE_NAMES.index("within_1825")]
    return ((exact > 0) | ((mm > 0) & (am > 0) & (w5 > 0))).astype(int)


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Grid-search threshold on accuracy."""
    grid = np.linspace(0.1, 0.9, 81)
    best_t, best_acc = 0.5, 0.0
    for t in grid:
        acc = accuracy_score(y_true, (proba >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    import joblib
    import lightgbm as lgb

    t0 = time.time()
    cases, labels = load_public()
    print(f"[load] cases={len(cases):,}  labels={len(labels):,}  "
          f"t={time.time()-t0:.1f}s")

    # Fit text vectorizers on ALL descriptions (current + priors).
    all_desc = []
    for c in cases:
        all_desc.append(c["current_study"]["study_description"])
        for p in c["prior_studies"]:
            all_desc.append(p["study_description"])
    extractor = FeatureExtractor().fit(all_desc)
    print(f"[fit-tfidf] char_vocab={len(extractor.char_vec.vocabulary_):,}  "
          f"word_vocab={len(extractor.word_vec.vocabulary_):,}  "
          f"t={time.time()-t0:.1f}s")

    X, y, groups, meta = build_xy(cases, labels, extractor)
    print(f"[features] X={X.shape}  pos_rate={y.mean():.3f}  "
          f"t={time.time()-t0:.1f}s")

    # --- Baseline 1: rules --------------------------------------------------
    rule_pred = rules_predict(X)
    rule_acc = accuracy_score(y, rule_pred)
    print(f"[baseline:rules] acc={rule_acc:.4f}")

    # --- Baseline 2: logistic regression (5-fold grouped CV) ---------------
    gkf = GroupKFold(n_splits=5)
    lr_oof = np.zeros(len(y), dtype=float)
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        lr = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        lr.fit(X[tr], y[tr])
        lr_oof[va] = lr.predict_proba(X[va])[:, 1]
    lr_thr, lr_acc = best_threshold(y, lr_oof)
    print(f"[baseline:logreg] auc={roc_auc_score(y, lr_oof):.4f}  "
          f"best_thr={lr_thr:.2f}  acc={lr_acc:.4f}")

    # --- LightGBM grouped CV (accuracy + tune threshold) -------------------
    lgb_params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=40,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        lambda_l2=1.0,
        verbosity=-1,
    )
    lgb_oof = np.zeros(len(y), dtype=float)
    best_iters = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=FEATURE_NAMES)
        dva = lgb.Dataset(X[va], label=y[va], feature_name=FEATURE_NAMES, reference=dtr)
        booster = lgb.train(
            lgb_params, dtr, num_boost_round=2000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        lgb_oof[va] = booster.predict(X[va], num_iteration=booster.best_iteration)
        best_iters.append(booster.best_iteration)
        acc_f = accuracy_score(y[va], (lgb_oof[va] >= 0.5).astype(int))
        print(f"  fold{fold} best_iter={booster.best_iteration}  val_acc@0.5={acc_f:.4f}")

    lgb_thr, lgb_acc = best_threshold(y, lgb_oof)
    print(f"[lightgbm:oof] auc={roc_auc_score(y, lgb_oof):.4f}  "
          f"best_thr={lgb_thr:.2f}  acc={lgb_acc:.4f}  "
          f"mean_iter={int(np.mean(best_iters))}")

    # --- Refit on ALL data with mean best_iteration ------------------------
    final_iters = int(np.mean(best_iters) * 1.1)
    dall = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    final_booster = lgb.train(lgb_params, dall, num_boost_round=final_iters)

    # Save bundle
    OUT.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump({
        "char_vec": extractor.char_vec,
        "word_vec": extractor.word_vec,
        "booster": final_booster,
        "threshold": lgb_thr,
        "feature_names": FEATURE_NAMES,
        "trained_rows": int(len(y)),
        "trained_cases": len(cases),
        "oof_accuracy": float(lgb_acc),
        "oof_auc": float(roc_auc_score(y, lgb_oof)),
        "rules_accuracy": float(rule_acc),
        "logreg_accuracy": float(lr_acc),
    }, OUT)
    print(f"[save] -> {OUT}   total_t={time.time()-t0:.1f}s")

    # --- Feature importance print (quick sanity) ---------------------------
    try:
        imp = final_booster.feature_importance(importance_type="gain")
        top = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:15]
        print("\n[top-15 features by gain]")
        for n, g in top:
            print(f"  {n:<25} {g:,.0f}")
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not print importance: {e}")


if __name__ == "__main__":
    main()
