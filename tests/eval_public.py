"""Run the deployed predictor end-to-end against the full public split and
report accuracy. Simulates the evaluator by POSTing a single batch that
contains every case in the public JSON."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "relevant_priors_public.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--chunk", type=int, default=0,
                    help="If >0, split into chunks of N cases each (for test).")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, take only the first N cases (quick check).")
    args = ap.parse_args()

    with DATA.open(encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload["cases"]
    truth = {(t["case_id"], t["study_id"]): bool(t["is_relevant_to_current"])
             for t in payload["truth"]}
    if args.limit:
        cases = cases[: args.limit]

    def send(cases_batch):
        body = json.dumps({
            "challenge_id": "relevant-priors-v1",
            "schema_version": 1,
            "cases": cases_batch,
        }).encode()
        req = urllib.request.Request(
            args.url, data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=360) as r:
            data = json.loads(r.read())
        return data["predictions"], time.time() - t0

    all_preds: list[dict] = []
    total = 0.0
    if args.chunk and args.chunk < len(cases):
        for i in range(0, len(cases), args.chunk):
            preds, dt = send(cases[i:i + args.chunk])
            total += dt
            all_preds.extend(preds)
            print(f"  chunk {i//args.chunk}: {len(preds)} preds in {dt:.2f}s")
    else:
        all_preds, total = send(cases)

    # Score
    correct = incorrect = skipped = 0
    expected_keys = set(truth.keys())
    got_keys = set()
    for p in all_preds:
        key = (p["case_id"], p["study_id"])
        got_keys.add(key)
        y = truth.get(key)
        if y is None:
            continue  # prediction for unknown prior — will not help or hurt
        if bool(p["predicted_is_relevant"]) == y:
            correct += 1
        else:
            incorrect += 1
    skipped = len(expected_keys - got_keys)

    total_count = correct + incorrect + skipped
    acc = correct / total_count if total_count else 0.0
    print(f"\nelapsed: {total:.2f}s  preds_received: {len(all_preds):,}")
    print(f"correct={correct:,}  incorrect={incorrect:,}  "
          f"skipped={skipped:,}  total={total_count:,}")
    print(f"ACCURACY = {acc:.4f}")


if __name__ == "__main__":
    main()
