#!/usr/bin/env python3
"""Run repeated, grouped training-image-count experiments without changing the deployed model."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from threadpoolctl import threadpool_limits

from seiyuu_faceid.experiments.sample_size import (
    audit_sources,
    load_data,
    run_curve,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--maxima", type=int, nargs="+", default=[100, 50])
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.seed < 0
        or not 0 < args.target <= 1
        or not 0 <= args.threshold <= 1
    ):
        parser.error("Invalid repeats, seed, target or threshold.")
    levels = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 125, 150, 175, 200)
    if len(set(args.maxima)) != len(args.maxima) or any(n not in levels for n in args.maxima):
        parser.error("Cohort maxima must be unique tested training sizes up to 200.")
    args.output.mkdir(parents=True, exist_ok=False)
    training = _bootstrap.ROOT / "data/training"
    bundle = _bootstrap.ROOT / "models/selected"
    data = load_data(training)
    with threadpool_limits(limits=2):
        audit = audit_sources(data, training, bundle)
        write_json(args.output / "audit.json", audit)
        write_csv(args.output / "identity_inventory.csv", audit["profile"])
        summaries = {}
        for maximum in args.maxima:
            sizes = [n for n in levels if n <= maximum]
            summaries[str(maximum)] = run_curve(
                data,
                args.output / f"cohort_{maximum}",
                sizes=sizes,
                repeats=args.repeats,
                seed=args.seed,
                threshold=args.threshold,
                target=args.target,
                min_test_groups=10,
            )
        write_json(args.output / "summary.json", summaries)
    print(f"Results saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
