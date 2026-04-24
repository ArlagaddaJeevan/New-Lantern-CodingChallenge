# Experiments

Task: for each (current study, prior study) pair, predict whether the prior
should be surfaced to the radiologist. Scoring is accuracy, skipped rows
count as wrong.

## Public split at a glance

- 996 cases, 27,614 labeled priors
- positive rate 0.238 (majority-class baseline accuracy = 0.762)
- mean 27.7 priors per case, max 234
- no duplicate `(case, prior)` keys, no priors without a truth label

Two things I noticed early that shaped the approach:

1. Where current and prior descriptions match exactly, the prior is relevant
   871 times out of 872 (precision 0.999, recall ~0.13). Strong, clean rule,
   worth including as a feature.
2. Descriptions are not formatted consistently. First-token "modality" was
   often an anatomy word like `CHEST`, `KNEE,`, `CERVICL`. Hard-coded
   splitting didn't work; I moved to regex search on a normalized string.

## Features

All in `app/features.py` so the same code runs at train and at serve time.

- char-wb TF-IDF cosine (3-5 grams) and word TF-IDF cosine (1-2 grams) on the
  normalized description
- modality bucket (MR/CT/XR/US/MAM/NM/ECHO/PETCT/DXA/FL/IR/OTHER) via regex
- anatomy bucket (19 options: HEAD, CHEST, BREAST, CSPINE, ..., OTHER)
  via keyword lookup
- laterality (NONE/LT/RT/BI), contrast (NONE/WO/W/BOTH), and match flags for
  each of the above
- days between studies, log(days+1), within-1yr, within-5yr, over-10yr
- priors per case, recency rank within the case, is-most-recent flag
- one-hot of modality/anatomy/laterality/contrast for current and prior

101 features total. Everything besides the two TF-IDF vectorizers is
deterministic text parsing.

## Models

5-fold `GroupKFold` on `case_id` so priors from the same patient never cross
folds.

| model                                        | accuracy | AUC    |
|----------------------------------------------|---------:|-------:|
| rules: exact_match OR (mod+anat+within_5yr)  |   0.8338 |      - |
| logistic regression (L2, C=1.0)              |   0.9256 | 0.9511 |
| LightGBM (shipped)                           |   0.9535 | 0.9826 |

LightGBM: 127 leaves, lr=0.05, min_data_in_leaf=40, feature/bagging
fractions 0.9, early stopping at 50. Best accuracy threshold tuned on OOF
came out at 0.45, which matches what you'd expect given the 24%
positive rate.

Top features by gain (LightGBM):

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

## What worked

- Char n-gram TF-IDF was the biggest jump from logreg. Medical descriptions
  are short, abbreviation-heavy, and typo-riddled. Char n-grams shrug that off.
- Putting feature code in one module. I've made this mistake before where
  train-time and serve-time extractors drift. Keeping one `Study` dataclass
  built from the raw request dict on both sides fixed that by construction.
- Per-case batched scoring, plus an in-process cache keyed on
  `(norm_current, norm_prior, days)`. Full public split takes ~40s on the
  first call and ~1s on a repeat (matters for retries near the 360s
  evaluator limit).

## What didn't

- First-token modality extraction. `split()[0]` put anatomy words in the
  modality slot and introduced noise. Dropped in favor of regex.
- `mod_match AND anat_match` alone as a classifier. 52% precision, too many
  stale cross-modality priors. Time-delta features fixed most of that.
- Logreg with `max_iter=2000` didn't converge and capped at 0.9256. The
  modality x anatomy interactions are the kind of thing trees handle
  trivially and a linear model can't.

## Serving

FastAPI, one uvicorn worker, bundle loaded once at startup. Each request
runs one `booster.predict` per case over a stacked feature matrix. Logs
include request id, case count, prior count, elapsed ms so evaluator calls
are easy to pull up in the logs.

## Next steps

1. The `anatomy=OTHER` bucket is showing up in the top feature gains. That
   means ~15% of rows aren't being cleanly classified and the model is
   compensating around it. Expanding the anatomy keyword list should get
   another 0.5-1 point.
2. Patient-level features: how many priors share modality with the current
   study, how many share anatomy, is the current description also present in
   the prior list. Cheap to add.
3. Sentence embeddings (MiniLM) per normalized description, cached. Would
   probably help on semantic equivalences the TF-IDF misses, e.g. 
   `US abdominal screening AAA` vs `aorta ultrasound`.
4. Stacking LightGBM + logreg. On a quick check the two disagree enough
   that a simple average might add ~0.1 point.

## Reproducing

```
pip install -r requirements.txt
python train/train.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
python tests/eval_public.py --url http://127.0.0.1:8000/predict
```

Training is ~2 minutes on a laptop CPU.
