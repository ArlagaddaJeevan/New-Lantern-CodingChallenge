"""Quick EDA to understand the public data structure and label distribution."""
import json
from collections import Counter
from datetime import date
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "relevant_priors_public.json"

with DATA.open(encoding="utf-8") as f:
    payload = json.load(f)

truth = payload["truth"]
cases = payload["cases"]

print(f"truth rows: {len(truth):,}   cases: {len(cases):,}")

labels = Counter(t["is_relevant_to_current"] for t in truth)
print(f"label balance: true={labels[True]:,}  false={labels[False]:,}  "
      f"prevalence={labels[True]/sum(labels.values()):.3f}")

# index truth by (case_id, study_id)
label_index = {(t["case_id"], t["study_id"]): t["is_relevant_to_current"] for t in truth}

# sanity: every prior in `cases` has a label?
missing = 0
prior_counts = []
for c in cases:
    prior_counts.append(len(c["prior_studies"]))
    for p in c["prior_studies"]:
        if (c["case_id"], p["study_id"]) not in label_index:
            missing += 1
print(f"priors without a truth label: {missing:,}")
print(f"priors per case: min={min(prior_counts)}  max={max(prior_counts)}  "
      f"mean={sum(prior_counts)/len(prior_counts):.1f}")

# dedupe: do any cases repeat study_ids as priors?
dups = 0
for c in cases:
    seen = set()
    for p in c["prior_studies"]:
        if p["study_id"] in seen:
            dups += 1
        seen.add(p["study_id"])
print(f"duplicate prior study_ids within a case: {dups}")

# sample modality tokens (first word of description)
first_tokens = Counter()
for c in cases:
    first_tokens[c["current_study"]["study_description"].split()[0].upper()] += 1
    for p in c["prior_studies"]:
        first_tokens[p["study_description"].split()[0].upper()] += 1
print("\ntop 25 first tokens across all descriptions:")
for tok, n in first_tokens.most_common(25):
    print(f"  {tok:<12} {n:,}")

# date range / time delta distribution
def parse_date(s):
    return date.fromisoformat(s)

deltas = []
for c in cases:
    try:
        cur_d = parse_date(c["current_study"]["study_date"])
    except Exception:
        continue
    for p in c["prior_studies"]:
        try:
            pd_ = parse_date(p["study_date"])
            deltas.append((cur_d - pd_).days)
        except Exception:
            pass

if deltas:
    deltas.sort()
    def pct(x, q):
        i = max(0, min(len(x) - 1, int(q * len(x))))
        return x[i]
    print(f"\ntime deltas (days): min={deltas[0]}  p10={pct(deltas,.1)}  "
          f"p50={pct(deltas,.5)}  p90={pct(deltas,.9)}  max={deltas[-1]}")

# relevance vs same-description exact match
same_desc_true = same_desc_false = diff_desc_true = diff_desc_false = 0
for c in cases:
    cur = c["current_study"]["study_description"].strip().upper()
    for p in c["prior_studies"]:
        y = label_index.get((c["case_id"], p["study_id"]))
        if y is None:
            continue
        same = cur == p["study_description"].strip().upper()
        if same and y: same_desc_true += 1
        elif same: same_desc_false += 1
        elif y: diff_desc_true += 1
        else: diff_desc_false += 1
print(f"\nexact-desc-match:  rel={same_desc_true:,}  not={same_desc_false:,}   "
      f"precision_if_rule={same_desc_true/max(1,same_desc_true+same_desc_false):.3f}")
print(f"diff-desc:         rel={diff_desc_true:,}  not={diff_desc_false:,}")

# modality-first-token match
def modality(desc: str) -> str:
    return desc.split()[0].upper() if desc else ""

mm_true = mm_false = nm_true = nm_false = 0
for c in cases:
    cur_mod = modality(c["current_study"]["study_description"])
    for p in c["prior_studies"]:
        y = label_index.get((c["case_id"], p["study_id"]))
        if y is None:
            continue
        match = cur_mod == modality(p["study_description"])
        if match and y: mm_true += 1
        elif match: mm_false += 1
        elif y: nm_true += 1
        else: nm_false += 1
print(f"\nmod-match:   rel={mm_true:,}  not={mm_false:,}   "
      f"precision={mm_true/max(1,mm_true+mm_false):.3f}")
print(f"mod-diff:    rel={nm_true:,}  not={nm_false:,}   "
      f"precision={nm_true/max(1,nm_true+nm_false):.3f}")
