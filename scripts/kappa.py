"""
Blind kappa labeling protocol — two steps:

  1. Export:  python scripts/kappa.py export --n 20 --out data/kappa_blind.csv
     Produces a CSV with job_id, title, description only — no scores.
     Randomized order. Give this file to both annotators independently.

  2. Join:    python scripts/kappa.py join --blind data/kappa_blind.csv \
                  --a1 data/kappa_labels_a1.csv --a2 data/kappa_labels_a2.csv
     Annotator CSVs must have columns: job_id, label  (fit/maybe/no)
     Computes and prints:
       - Inter-annotator kappa (a1 vs a2)
       - Model agreement against consensus label (majority vote)
     These are two separate numbers, reported separately.
"""

import argparse
import csv
import random
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "runs.db"
VALID_LABELS = {"fit", "maybe", "no"}


def export(n: int, out: Path, job_id_range: tuple[int, int] | None = None) -> None:
    con = sqlite3.connect(DB_PATH)
    # job_id_range filters by the jobs.job_id FK stored in runs; None = all runs.
    # runs joins back to jobs so we can filter by job_id range.
    if job_id_range:
        lo, hi = job_id_range
        rows = con.execute(
            """SELECT r.id, r.job_title, r.job_desc, r.overall_score
               FROM runs r
               WHERE r.job_desc IS NOT NULL AND r.job_id BETWEEN ? AND ?
               ORDER BY r.created_at DESC""",
            (lo, hi),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT id, job_title, job_desc, overall_score FROM runs WHERE job_desc IS NOT NULL ORDER BY created_at DESC LIMIT ?",
            (n * 3,),
        ).fetchall()
    con.close()

    if not rows:
        sys.exit("No scored runs in runs.db for the specified range. Score some JDs first.")

    random.shuffle(rows)
    rows = rows[:n]

    if len(rows) < n:
        print(f"Warning: only {len(rows)} runs available, requested {n}.")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "job_title", "job_description"])
        for run_id, title, desc, _score in rows:
            w.writerow([run_id, title or "", desc or ""])

    # Separately store the score mapping (for join step) — never shown to annotators
    score_path = out.with_suffix(".scores.csv")
    with open(score_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "model_score"])
        for run_id, _title, _desc, score in rows:
            w.writerow([run_id, score])

    print(f"Blind sheet:   {out}  ({len(rows)} rows)")
    print(f"Score mapping: {score_path}  (keep private until after labeling)")


def cohen_kappa(labels_a: list[str], labels_b: list[str], classes: list[str]) -> float:
    n = len(labels_a)
    assert n == len(labels_b) and n > 0

    # Observed agreement
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement
    p_e = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in classes
    )

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def load_labels(path: Path) -> dict[str, str]:
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().lower()
            if label not in VALID_LABELS:
                sys.exit(f"Invalid label '{label}' in {path}. Must be fit/maybe/no.")
            out[row["job_id"].strip()] = label
    return out


# Score bins pre-registered before any labeling data was collected — 2026-07-27.
# Do not adjust after seeing labels; that would be post-hoc threshold tuning.
# Semantics: ≥0.65 = clear fit, 0.45–0.65 = borderline, <0.45 = clear miss.
_FIT_THRESHOLD = 0.65
_NO_THRESHOLD = 0.45


def model_label(score: float) -> str:
    if score is None:
        return "maybe"
    if score >= _FIT_THRESHOLD:
        return "fit"
    if score >= _NO_THRESHOLD:
        return "maybe"
    return "no"


def join(blind: Path, a1_path: Path, a2_path: Path) -> None:
    score_path = blind.with_suffix(".scores.csv")
    if not score_path.exists():
        sys.exit(f"Score mapping not found: {score_path}")

    scores: dict[str, float] = {}
    with open(score_path) as f:
        for row in csv.DictReader(f):
            scores[row["job_id"]] = float(row["model_score"]) if row["model_score"] else None

    a1 = load_labels(a1_path)
    a2 = load_labels(a2_path)

    common_ids = sorted(set(a1) & set(a2) & set(scores))
    if len(common_ids) < len(scores):
        missing = set(scores) - set(a1) - set(a2)
        print(f"Warning: {len(missing)} job_ids missing from one or both label files.")

    if not common_ids:
        sys.exit("No overlapping job_ids between annotators and score mapping.")

    labels_a1 = [a1[jid] for jid in common_ids]
    labels_a2 = [a2[jid] for jid in common_ids]
    classes = list(VALID_LABELS)

    kappa = cohen_kappa(labels_a1, labels_a2, classes)

    # Consensus = majority vote; ties (all different) → "maybe"
    consensus = []
    for jid in common_ids:
        votes = [a1[jid], a2[jid]]
        for label in VALID_LABELS:
            if votes.count(label) >= 2:
                consensus.append(label)
                break
        else:
            consensus.append("maybe")

    model_labels = [model_label(scores[jid]) for jid in common_ids]
    model_agreement = sum(m == c for m, c in zip(model_labels, consensus)) / len(common_ids)

    print(f"\n{'='*50}")
    print(f"N pairs:                    {len(common_ids)}")
    print(f"Inter-annotator kappa:      {kappa:.3f}")
    print(f"Model vs consensus agree:   {model_agreement:.1%}")
    print(f"{'='*50}")
    print(f"\nKappa interpretation: <0.4 poor, 0.4-0.6 fair, 0.6-0.8 good, >0.8 excellent")
    print(f"These are two separate validity claims — report them separately.\n")

    # Per-label breakdown
    print("Model confusion vs consensus:")
    print(f"{'Consensus':<10} {'Model':<10} {'Count'}")
    from collections import Counter
    for (c, m), cnt in sorted(Counter(zip(consensus, model_labels)).items()):
        print(f"  {c:<10} {m:<10} {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kappa labeling protocol")
    sub = parser.add_subparsers(dest="cmd")

    exp = sub.add_parser("export", help="Export blind labeling sheet")
    exp.add_argument("--n", type=int, default=20)
    exp.add_argument("--out", type=Path, default=Path("data/kappa_blind.csv"))
    exp.add_argument(
        "--jobs", type=str, default=None,
        help="Job ID range to include, e.g. '1-12' for Rishabh's Java set or '13-31' for Gaurica's ML set. "
             "Omit to use all scored runs."
    )

    jn = sub.add_parser("join", help="Compute kappa from filled label sheets")
    jn.add_argument("--blind", type=Path, required=True)
    jn.add_argument("--a1", type=Path, required=True, help="Annotator 1 labels CSV (job_id, label)")
    jn.add_argument("--a2", type=Path, required=True, help="Annotator 2 labels CSV (job_id, label)")

    args = parser.parse_args()
    if args.cmd == "export":
        job_range = None
        if args.jobs:
            try:
                lo, hi = map(int, args.jobs.split("-"))
                job_range = (lo, hi)
            except ValueError:
                sys.exit("--jobs must be in format 'lo-hi', e.g. '1-12'")
        export(args.n, args.out, job_id_range=job_range)
    elif args.cmd == "join":
        join(args.blind, args.a1, args.a2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
