"""Source-group-aware learning curves for the frozen supervised classifier."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np


@dataclass
class StudyData:
    features: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    rows: list[dict[str, str]]
    fit_indices: np.ndarray
    test_indices: np.ndarray


def validate_data(data: StudyData) -> None:
    n = len(data.rows)
    if (
        data.features.ndim != 2
        or len(data.features) != n
        or data.labels.shape != (n,)
        or not np.isfinite(data.features).all()
        or data.labels.dtype.kind not in "iu"
        or len(set(data.class_names)) != len(data.class_names)
    ):
        raise ValueError("Invalid feature/label dimensions or values.")
    fit, test = data.fit_indices.tolist(), data.test_indices.tolist()
    if (
        len(set(fit)) != len(fit)
        or len(set(test)) != len(test)
        or set(fit) & set(test)
        or set(fit) | set(test) != set(range(n))
        or set(data.labels[fit]) != set(range(len(data.class_names)))
    ):
        raise ValueError("Invalid, incomplete or overlapping split indices/classes.")
    for i, row in enumerate(data.rows):
        if row["identity"] != data.class_names[int(data.labels[i])]:
            raise ValueError("Sample identity differs from the feature label.")
    if len({row["sample_id"] for row in data.rows}) != n:
        raise ValueError("Duplicate sample_id.")
    for field in ("sample_id", "source_sha256", "crop_sha256", "source_group"):
        if any(not isinstance(row.get(field), str) or not row[field] for row in data.rows):
            raise ValueError(f"Missing {field}.")
        if {data.rows[i][field] for i in fit} & {data.rows[i][field] for i in test}:
            raise ValueError(f"Train/holdout leakage: {field}.")


def load_data(directory: Path) -> StudyData:
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate sample manifest keys.")
    with np.load(directory / "features_and_splits.npz", allow_pickle=False) as a:
        data = StudyData(
            a["features"].copy(),
            a["labels"].copy(),
            a["class_names"].tolist(),
            [by_id[str(sample_id)] for sample_id in a["sample_ids"]],
            np.concatenate([a["train_indices"], a["validation_indices"]]),
            a["test_indices"].copy(),
        )
    if len(rows) != len(data.rows):
        raise ValueError("Feature and manifest counts differ.")
    validate_data(data)
    return data


def group_pool(data: StudyData, indices: np.ndarray, label: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index in indices:
        if data.labels[index] == label:
            groups.setdefault(data.rows[int(index)]["source_group"], []).append(int(index))
    return groups


def sample_order(data: StudyData, indices: np.ndarray, label: int, *, seed: int) -> list[int]:
    groups = group_pool(data, indices, label)
    rng = np.random.default_rng(np.random.SeedSequence([seed, label]))
    keys = sorted(groups)
    return [int(rng.choice(groups[keys[int(position)]])) for position in rng.permutation(len(keys))]


def select_fit_indices(data: StudyData, orders: dict[int, list[int]], *, size: int) -> np.ndarray:
    if size < 1 or any(len(order) < size for order in orders.values()):
        raise ValueError("Requested sample count exceeds available source groups.")
    background = [int(i) for i in data.fit_indices if int(data.labels[i]) not in orders]
    selected = background + [i for order in orders.values() for i in order[:size]]
    return np.asarray(sorted(selected), dtype=np.int64)


def measure_identity(
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    label: int,
    threshold: float,
) -> dict[str, int | float | None]:
    from scipy.stats import beta

    mask = truth == label
    count = int(mask.sum())
    correct = predictions[mask] == label
    accepted = probabilities[mask] >= threshold
    returned = int((correct & accepted).sum())
    wrong = int((~correct & accepted).sum())
    return {
        "test_count": count,
        "top1_correct": int(correct.sum()),
        "correct_returned": returned,
        "wrong_returned": wrong,
        "review": int((~accepted).sum()),
        "top1_rate": float(correct.mean()) if count else None,
        "correct_return_rate": returned / count if count else None,
        "accepted_precision": returned / (returned + wrong) if returned + wrong else None,
        "lower95": float(beta.ppf(0.05, returned, count - returned + 1)) if returned else 0.0,
    }


def minimum_sustained_size(sizes: list[int], passed: list[bool]) -> int | None:
    if len(sizes) != len(passed) or sorted(set(sizes)) != sizes:
        raise ValueError("Sizes must be ordered and unique, with one result per size.")
    for index, size in enumerate(sizes):
        if all(passed[index:]):
            return size
    return None


def profile(data: StudyData) -> list[dict[str, object]]:
    output = []
    for label, name in enumerate(data.class_names):
        fit = [data.rows[int(i)] for i in data.fit_indices if data.labels[i] == label]
        test = [data.rows[int(i)] for i in data.test_indices if data.labels[i] == label]
        output.append(
            {
                "identity": name,
                "label": label,
                "fit_crops": len(fit),
                "fit_photos": len({row["source_sha256"] for row in fit}),
                "fit_groups": len({row["source_group"] for row in fit}),
                "test_crops": len(test),
                "test_photos": len({row["source_sha256"] for row in test}),
                "test_groups": len({row["source_group"] for row in test}),
            }
        )
    return output


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_scores(data: StudyData, fit: np.ndarray, test: np.ndarray) -> np.ndarray:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1500, random_state=20261014
            ),
        ).fit(data.features[fit], data.labels[fit])
    np.testing.assert_array_equal(model[-1].classes_, np.arange(len(data.class_names)))
    return model.predict_proba(data.features[test])


def run_curve(
    data: StudyData,
    output: Path,
    *,
    sizes: list[int],
    repeats: int,
    seed: int,
    threshold: float,
    target: float,
    min_test_groups: int,
) -> dict[str, object]:
    validate_data(data)
    if not sizes or sorted(set(sizes)) != sizes or sizes[0] < 1 or repeats < 1:
        raise ValueError("Use ordered positive sizes and at least one repeat.")
    output.mkdir(parents=True, exist_ok=False)
    audit = profile(data)
    cohort = [
        int(row["label"])
        for row in audit
        if row["fit_groups"] >= max(sizes) and row["test_groups"] >= min_test_groups
    ]
    if not cohort:
        raise ValueError("No identities qualify for this fixed cohort.")
    test = np.asarray(
        [
            index
            for label in range(len(data.class_names))
            for index in sample_order(data, data.test_indices, label, seed=seed + 1_000_000)
        ]
    )
    truth = data.labels[test]
    cohort_mask = np.isin(truth, cohort)
    predictions = np.full((repeats, len(sizes), len(test)), -1, dtype=np.int16)
    probabilities = np.full(predictions.shape, np.nan, dtype=np.float64)
    records: list[dict[str, object]] = []
    sampling = []
    started = time.perf_counter()
    for repeat in range(repeats):
        orders = {
            label: sample_order(data, data.fit_indices, label, seed=seed + repeat)
            for label in cohort
        }
        sampling.append(
            {
                "seed": seed + repeat,
                "ordered_samples": {
                    data.class_names[label]: [
                        data.rows[i]["sample_id"] for i in order[: max(sizes)]
                    ]
                    for label, order in orders.items()
                },
            }
        )
        for position, size in enumerate(sizes):
            scores = fit_scores(data, select_fit_indices(data, orders, size=size), test)
            predictions[repeat, position] = scores.argmax(axis=1)
            probabilities[repeat, position] = scores.max(axis=1)
            for label in cohort:
                metrics = measure_identity(
                    truth,
                    predictions[repeat, position],
                    probabilities[repeat, position],
                    label=label,
                    threshold=threshold,
                )
                records.append(
                    {
                        "repeat": repeat,
                        "seed": seed + repeat,
                        "size": size,
                        "identity": data.class_names[label],
                        **metrics,
                    }
                )
        np.savez_compressed(
            output / "predictions.npz",
            sizes=sizes,
            seeds=np.arange(seed, seed + repeats),
            predictions=predictions[: repeat + 1],
            probabilities=probabilities[: repeat + 1],
            labels=truth,
            test_sample_ids=[data.rows[int(i)]["sample_id"] for i in test],
            cohort=cohort,
            class_names=data.class_names,
        )
        write_json(output / "sampling.json", sampling)
        print(
            f"{output.name}: {repeat + 1}/{repeats} repeats, "
            f"{(repeat + 1) * len(sizes)} fits, {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    per_identity = []
    for label in cohort:
        name = data.class_names[label]
        observed, supported = [], []
        values = []
        for size in sizes:
            selected = [row for row in records if row["identity"] == name and row["size"] == size]
            rates = [float(row["correct_return_rate"]) for row in selected]
            lower = [float(row["lower95"]) for row in selected]
            observed.append(min(rates) + 1e-12 >= target)
            supported.append(min(lower) + 1e-12 >= target)
            values.append(
                {
                    "size": size,
                    "identity": name,
                    "test_count": selected[0]["test_count"],
                    "mean_top1": float(np.mean([row["top1_rate"] for row in selected])),
                    "mean_correct_return": float(np.mean(rates)),
                    "p05_correct_return": float(np.quantile(rates, 0.05)),
                    "worst_correct_return": min(rates),
                    "worst_lower95": min(lower),
                    "passing_draw_fraction": sum(rate + 1e-12 >= target for rate in rates)
                    / repeats,
                }
            )
        per_identity.append(
            {
                "identity": name,
                "test_count": values[0]["test_count"],
                "observed_minimum": minimum_sustained_size(sizes, observed),
                "confidence_minimum": minimum_sustained_size(sizes, supported),
                "by_size": values,
            }
        )
    curve = []
    for position, size in enumerate(sizes):
        correct = predictions[:, position] == truth
        accepted = probabilities[:, position] >= threshold
        success = correct & accepted
        observed = [item["by_size"][position] for item in per_identity]
        curve.append(
            {
                "size": size,
                "target_identities": len(cohort),
                "test_count": int(cohort_mask.sum()),
                "mean_top1": float(correct[:, cohort_mask].mean()),
                "mean_correct_return": float(success[:, cohort_mask].mean()),
                "worst_identity_draw": min(row["worst_correct_return"] for row in observed),
                "passing_identities_all_draws": sum(
                    row["worst_correct_return"] + 1e-12 >= target for row in observed
                ),
                "all50_supported_mean_correct_return": float(success.mean()),
                "all50_supported_mean_top1": float(correct.mean()),
            }
        )
    summary = {
        "sizes": sizes,
        "repeats": repeats,
        "seed": seed,
        "threshold": threshold,
        "target": target,
        "cohort": [data.class_names[label] for label in cohort],
        "test_count": int(cohort_mask.sum()),
        "fit_count_background": int(np.sum(~np.isin(data.labels[data.fit_indices], cohort))),
        "curve": curve,
        "per_identity": per_identity,
        "joint_observed_minimum": minimum_sustained_size(
            sizes, [row["passing_identities_all_draws"] == len(cohort) for row in curve]
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    write_csv(output / "runs.csv", records)
    write_csv(output / "curve.csv", curve)
    write_csv(
        output / "identity_by_size.csv", [row for item in per_identity for row in item["by_size"]]
    )
    return summary


def audit_sources(data: StudyData, training: Path, bundle: Path) -> dict[str, object]:
    from seiyuu_faceid.vision.classifier import LinearIdentityHead

    head = LinearIdentityHead(bundle / "classifier.npz")
    if head.class_names != data.class_names:
        raise ValueError("Model and training class orders differ.")
    original_scores = head.predict_proba(data.features[data.test_indices])
    with np.load(bundle / "holdout_scores.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            saved["sample_ids"], [data.rows[int(i)]["sample_id"] for i in data.test_indices]
        )
        np.testing.assert_array_equal(saved["labels"], data.labels[data.test_indices])
        np.testing.assert_allclose(saved["scores"], original_scores, atol=2e-6, rtol=0)
    refit_scores = fit_scores(data, data.fit_indices, data.test_indices)
    np.testing.assert_array_equal(refit_scores.argmax(1), original_scores.argmax(1))
    np.testing.assert_allclose(refit_scores, original_scores, atol=2e-6, rtol=0)
    paths = [training / name for name in ("features_and_splits.npz", "samples.jsonl")]
    paths += [bundle / name for name in ("classifier.npz", "manifest.json", "holdout_scores.npz")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    protocol = json.loads((training / "holdout_protocol.json").read_text())
    stale = [
        path.name
        for path in paths[:2]
        if hashes[str(path)] != protocol["frozen_data_sha256"][path.name]
    ]
    return {
        "rows": len(data.rows),
        "identities": len(data.class_names),
        "fit_crops": len(data.fit_indices),
        "test_crops": len(data.test_indices),
        "hashes": hashes,
        "stale_protocol_hash_entries": stale,
        "refit_max_probability_difference": float(np.abs(refit_scores - original_scores).max()),
        "refit_predictions_match": True,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "scipy")},
        "profile": profile(data),
        "limitations": [
            "Exploratory reuse of a holdout already used for model selection.",
            "Aligned-face features exclude detector/quality failures and unknown identities.",
            "Source groups approximate independence; complete shoot independence is unverified.",
            "Repeated training draws do not increase independent holdout counts.",
            "Per-identity exact intervals are not simultaneous multiple-comparison guarantees.",
            "Data-rich cohort minima cannot be transferred to excluded identities.",
        ],
    }
