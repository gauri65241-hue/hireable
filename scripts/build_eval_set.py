"""
Build data/eval_set.json from test.csv.
Stratified sample: 20 per tier (60 total).
Run: python scripts/build_eval_set.py [--n 20] [--seed 42]
"""
import argparse
import json
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_MAP = {
    "Good Fit":      "Strong",
    "Potential Fit": "Moderate",
    "No Fit":        "Weak",
}
SCORE_RANGES = {
    "Strong":   [65, 90],
    "Moderate": [28, 64],
    "Weak":     [5,  34],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="/Users/gauricapuri/Work/test.csv")
    ap.add_argument("--n",    type=int, default=20, help="samples per tier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  default=os.path.join(BASE_DIR, "data", "eval_set.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"resume_text", "job_description_text", "label"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df["human_tier"] = df["label"].map(LABEL_MAP)
    unmapped = df["human_tier"].isna().sum()
    if unmapped:
        print(f"WARNING: {unmapped} rows have unmapped labels — skipped")
    df = df.dropna(subset=["human_tier"])

    samples = []
    for tier in ["Strong", "Moderate", "Weak"]:
        pool = df[df["human_tier"] == tier]
        n = min(args.n, len(pool))
        if n < args.n:
            print(f"WARNING: only {n} examples available for tier {tier} (wanted {args.n})")
        chunk = pool.sample(n=n, random_state=args.seed)
        for _, row in chunk.iterrows():
            samples.append({
                "id":                   f"eval_{len(samples)+1:03d}",
                "resume_text":          str(row["resume_text"]).strip(),
                "job_description":      str(row["job_description_text"]).strip(),
                "human_tier":           tier,
                "source_label":         row["label"],
                "expected_score_range": SCORE_RANGES[tier],
                "human_notes":          "",
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(samples)} examples → {args.out}")
    for tier in ["Strong", "Moderate", "Weak"]:
        n = sum(1 for s in samples if s["human_tier"] == tier)
        print(f"  {tier:10s}: {n}")


if __name__ == "__main__":
    main()
