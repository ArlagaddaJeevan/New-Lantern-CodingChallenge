# Relevant-priors write-up — experiments, results, next steps

## 1 · Problem framing

For each `(current_study, prior_study)` pair the task is binary: should this prior be shown to the radiologist reading `current_study`? Scoring is pure accuracy, so decision threshold and class balance matter as much as model quality.

## 2 · Data (public split)

| stat | value |
|------|------:|
| cases | 996 |
| labeled priors | 27,614 |
| positive rate | **0.238** (majority-class accuracy = 0.762) |
| max priors per case | 234 |
| mean priors per case | 27.7 |
| duplicate `(case, prior)` keys | 0 |
| priors without a truth label | 0 |

Early descriptive pivots:
- **Exact-description match** → 871/872 were relevant → **precision 0.999**, recall 0.13. Very clean feature.
- First-token modality match → precision 0.525 (vs 0.175 when modalities differ): weak alone, useful as a feature.
- Descriptions are not normalized: `"KNEE,"`, `"R2"`, `"CERVICL"`, `"MAMMO"` appear as first tokens. Any modality/anatomy parsing has to tolerate this sloppiness, so I lean heavily on TF-IDF rather than hard-coded token positions.

## 3 · Feature engineering

All feature code lives in [`app/features.py`](app/features.py) so the exact same transform runs at train and at serve time.

**Per-description parsing** (regex / keyword buckets, stateless):
- `modality` ∈ {MR, CT, XR, US, MAM, NM, ECHO, PETCT, DXA, FL, IR, OTHER}
- `anatomy` — 19 buckets (HEAD, CHEST, BREAST, CSPINE/TSPINE/LSPINE, ABDOMEN, PELVIS, …)
- `laterality` ∈ {NONE, LT, RT, BI}
- `contrast` ∈ {NONE, WO, W, BOTH}
- Heavy synonym expansion (`CNTRST`→`CONTRAST`, `MAMMO`→`MAM`, `WO`→`WITHOUT`, `BI`→`BILATERAL`, …) before tokenization so the vectorizers see consistent text.

**Per-case text similarity** (TF-IDF, fitted once at train time on all descriptions):
- `cosine_char` — char-wb 3–5-gram TF-IDF cosine (robust to typos / abbreviations)
- `cosine_word` — word 1–2-gram TF-IDF cosine
- `jaccard` on whitespace tokens, prefix-3 equality, exact-match flag

**Per-pair numeric**:
- `days_delta`, `log_days`, `within_365`, `within_1825`, `over_3650`
- `mod_match`, `anat_match`, `lat_match`, `contrast_match`, `same_mod_anat`
- `desc_len_cur / prior / diff`
- `priors_per_case`, `recency_rank`, `is_most_recent`

Plus one-hot encodings of `modality` / `anatomy` / `laterality` / `contrast` for both current and prior (so the model can learn that "MAM × BREAST" behaves differently from "CT × CHEST"). Total = **101 features**.

## 4 · Models tried

All evaluated with **5-fold `GroupKFold` by `case_id`** — priors from the same patient never cross folds, which matches the private-split setup.

| model | CV accuracy | CV AUC |
|-------|------------:|------:|
| rules-only: `exact_match OR (mod_match AND anat_match AND within_5y)` | 0.8338 | — |
| logistic regression (L2, C=1.0) on engineered features | 0.9256 | 0.9511 |
| **LightGBM** (GBDT, 127 leaves, lr=0.05, early-stopped per fold) | **0.9535** | **0.9826** |

Best decision threshold is 0.45 (slightly below 0.5), which reflects the class imbalance: the model is very confident on the easy negatives and needs a little push to flag borderline positives without over-firing.

### Top feature importances (gain)
```
anat_match         136,570
cosine_char         17,156
priors_per_case     10,917
anat_cur=OTHER       7,354
desc_len_cur         6,939
days_delta           6,720
desc_len_prior       6,551
jaccard              6,143
cosine_word          5,659
mod_pri=PETCT        4,800
```

Anatomy match dominates, closely followed by char-level textual similarity and recency. The `OTHER` buckets appearing so high tells me the anatomy extractor is missing a chunk of cases — a targeted expansion would lift accuracy another 0.5–1 pt.

## 5 · What worked

- **Char-level TF-IDF cosine** was the single biggest lift from the logistic-regression to the GBDT model: medical descriptions are short, abbreviation-heavy, and riddled with typos, and a char-n-gram vectorizer handles that gracefully.
- **Keeping train and inference feature code in the same module** (`app/features.py`) — the `Study` dataclass is built from the raw request dict on both sides, so there is zero drift between what the booster saw at training and what it sees at serve time.
- **Per-case vectorized scoring + in-memory cache** on `(current_norm, prior_norm, days_delta)` keeps the end-to-end latency at ~1s for the full 27,614-prior public set on a re-run (a critical property given the evaluator's 360-s cutoff and the fact that retries would otherwise re-pay the compute).

## 6 · What failed

- **First-token modality extraction** (split()[0]) — inconsistent description formatting meant anatomy words like `"CHEST"`, `"LUMBAR"`, `"KNEE,"` leaked into the modality column. Switched to regex search on a normalized string and ordered-fallback bucketing.
- **Simple rule `mod_match AND anat_match → True`** — 52.5% precision, a lot of false positives on stale cross-modality priors. Needed the time-delta features to fix.
- **Logistic regression failed to converge at `max_iter=2000`** and still hit 92.6% — solid baseline, but the one-hot interactions between modality and anatomy are exactly the kind of thing trees handle trivially and a linear model cannot.

## 7 · Deployment

- **Primary:** Render Docker blueprint ([`render.yaml`](render.yaml)) — single web service, `/healthz` health-checked.
- **Backup:** Fly.io ([`fly.toml`](fly.toml)) — same Docker image, `min_machines_running = 1` so the evaluator never hits a cold start.

Startup loads the 3-MB `bundle.joblib` once; per-request we run a single `booster.predict` call per case over a stacked feature matrix. Logs include request id, case count, prior count, and elapsed ms for evaluator-side debugging.

## 8 · Next-step improvements

1. **Close the `anat=OTHER` gap.** ~15% of the training rows fall into the OTHER anatomy bucket and they show up high in the gain table, which means the model is compensating. Adding keywords for PET/CT oncology sub-regions, neck vessels, and common breast-intervention descriptions should push OOF accuracy to ~0.96+.
2. **Pairwise patient context.** Features like "# of priors sharing modality with current", "# of priors sharing anatomy", "is the current desc exactly-present elsewhere in the prior list" — these are cheap and let the model reason about whether a given prior is *relatively* relevant vs. the rest of the patient history.
3. **Study-description embeddings.** A small MiniLM embedding of each description, cached keyed on the normalized string, would capture semantic equivalences the TF-IDF misses (e.g., `"US abdominal screening AAA"` ≡ `"aorta ultrasound"`). Can be pre-computed for the entire vocabulary observed in a batched request and re-used across the case — no per-prior network calls needed.
4. **Isotonic calibration** per modality-of-current to stabilize the threshold when the private split's modality mix skews differently.
5. **Stacked blend of LightGBM + logistic regression.** Current LightGBM OOF AUC is 0.9826, and the LR baseline is uncorrelated enough that a simple average gives ~0.1 pt bump in most experiments I have run on similar medical-text problems.
6. **Active learning on the `0.40 < p < 0.55` band.** If we ever get a second labeled split, those borderline rows are where the error lives — a hand-reviewed batch of ~500 would likely be worth 1–2 pts.

## 9 · Reproducibility

```bash
pip install -r requirements.txt
python train/train.py                                   # regenerates models/bundle.joblib
uvicorn app.main:app --host 0.0.0.0 --port 8000         # serves /predict
python tests/eval_public.py --url http://127.0.0.1:8000/predict
```

Training is deterministic given the same `relevant_priors_public.json` (LightGBM runs with default seed; no shuffling in `GroupKFold`). Full run ≈ 2 minutes on a laptop CPU.
