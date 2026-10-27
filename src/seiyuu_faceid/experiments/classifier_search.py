"""Development-only selection of portable linear heads on private frozen features."""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from seiyuu_faceid.experiments.sample_size import StudyData, validate_data, write_json
from seiyuu_faceid.validation import classification_metrics
from seiyuu_faceid.vision.classifier import LinearIdentityHead


@dataclass(frozen=True)
class SearchConfig:
    scaling: str
    C: float
    class_weight: str | None
    adaface_weight: float
    group_equalized: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def candidate_grid() -> list[SearchConfig]:
    """Fixed, bounded grid; the deployed recipe is the first candidate."""
    baseline = SearchConfig("standard", 1.0, "balanced", 0.5)
    grid = [
        SearchConfig("standard", c, w, a)
        for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        for w in ["balanced", None]
        for a in [0.25, 0.5, 0.75]
    ]
    grid += [
        SearchConfig("raw", c, w, 0.5)
        for c in [1.0, 10.0, 100.0, 1000.0]
        for w in ["balanced", None]
    ]
    grid += [
        SearchConfig("standard", c, w, 0.5, True)
        for c in [0.1, 1.0, 10.0]
        for w in ["balanced", None]
    ]
    return [baseline] + [candidate for candidate in grid if candidate != baseline]


def connected_components(data: StudyData) -> np.ndarray:
    """Join all duplicate keys transitively before inspecting development folds."""
    validate_data(data)
    if data.features.shape[1] % 2:
        raise ValueError("Expected two equally sized feature blocks.")
    parent = np.arange(len(data.rows))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    seen = {}
    for index, row in enumerate(data.rows):
        for field in (
            "sample_id",
            "source_group",
            "source_sha256",
            "crop_sha256",
            "known_book_group",
        ):
            value = row.get(field)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise ValueError(f"Invalid duplicate key: {field}.")
            key = (field, value)
            if key in seen:
                left, right = find(index), find(seen[key])
                parent[max(left, right)] = min(left, right)
            else:
                seen[key] = index
    components = np.array([find(i) for i in range(len(parent))])
    if set(components[data.fit_indices]) & set(components[data.test_indices]):
        raise ValueError("Duplicate component crosses development/test boundary.")
    for component in np.unique(components):
        if len(np.unique(data.labels[components == component])) != 1:
            raise ValueError("Duplicate component has conflicting identities.")
    return components


def make_folds(data: StudyData, components: np.ndarray, *, seed: int, folds: int = 3):
    """Greedy per-identity component balancing, pinning untestable identities."""
    if folds < 2:
        raise ValueError("At least two folds are required.")
    assignments = [[] for _ in range(folds)]
    pinned = []
    for label in range(len(data.class_names)):
        indices = data.fit_indices[data.labels[data.fit_indices] == label]
        groups = [indices[components[indices] == group] for group in np.unique(components[indices])]
        if len(groups) == 1:
            pinned.extend(indices.tolist())
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, label]))
        groups = [groups[i] for i in rng.permutation(len(groups))]
        groups.sort(key=len, reverse=True)
        counts = np.zeros(folds, dtype=int)
        for group in groups:
            target = int(np.argmin(counts))
            assignments[target].extend(group.tolist())
            counts[target] += len(group)
    result = []
    for assignment in assignments:
        valid = np.array(sorted(assignment), dtype=np.int64)
        train = np.setdiff1d(data.fit_indices, valid)
        if not len(valid) or set(data.labels[train]) != set(range(len(data.class_names))):
            raise ValueError("Every fold must validate samples and retain all training identities.")
        if set(components[train]) & set(components[valid]):
            raise ValueError("Fold component leakage.")
        result.append((train, valid))
    return result, np.array(sorted(pinned), dtype=np.int64)


def component_weights(labels: np.ndarray, components: np.ndarray, *, balanced: bool):
    """Equalize components, optionally equalizing total class mass, then mean one."""
    unique, counts = np.unique(components, return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.array([1.0 / sizes[int(group)] for group in components])
    if balanced:
        for label in np.unique(labels):
            mask = labels == label
            weights[mask] /= len(np.unique(components[mask]))
    return weights / weights.mean()


def _transform(features, mean, scale):
    values = np.array(features, dtype=np.float32, copy=True)
    values -= mean
    values /= scale
    return values


@dataclass
class FittedHead:
    mean: np.ndarray
    scale: np.ndarray
    classifier: object

    def predict_proba(self, features):
        return self.classifier.predict_proba(_transform(features, self.mean, self.scale))


def fit_head(data: StudyData, train: np.ndarray, components: np.ndarray, config: SearchConfig):
    """Fit preprocessing only on this fold, using the export's exact transform."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if config.scaling not in {"standard", "raw"} or not 0 < config.adaface_weight < 1:
        raise ValueError("Invalid feature transform configuration.")
    features = data.features[train]
    dimensions = features.shape[1]
    if config.scaling == "standard":
        scaler = StandardScaler().fit(features)
        mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
    else:
        mean, scale = np.zeros(dimensions), np.ones(dimensions)
    midpoint = dimensions // 2
    scale[:midpoint] /= np.sqrt(2 * config.adaface_weight)
    scale[midpoint:] /= np.sqrt(2 * (1 - config.adaface_weight))
    weights = None
    class_weight = config.class_weight
    if config.group_equalized:
        weights = component_weights(
            data.labels[train], components[train], balanced=class_weight == "balanced"
        )
        class_weight = None
    classifier = LogisticRegression(
        C=config.C, class_weight=class_weight, max_iter=1500, random_state=20261014
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(_transform(features, mean, scale), data.labels[train], sample_weight=weights)
    np.testing.assert_array_equal(classifier.classes_, np.arange(len(data.class_names)))
    return FittedHead(mean, scale, classifier)


def export_head(fitted: FittedHead, class_names: list[str], path: Path, verify: np.ndarray):
    """Write a new portable artifact and verify actual runtime probabilities."""
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            mean=fitted.mean,
            scale=fitted.scale,
            coefficients=fitted.classifier.coef_,
            intercept=fitted.classifier.intercept_,
            class_names=class_names,
        )
    expected = fitted.predict_proba(verify)
    actual = LinearIdentityHead(path).predict_proba(verify)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
    return float(np.max(np.abs(actual - expected)))


def score_metrics(labels, scores, *, detailed=False, components=None):
    metrics = classification_metrics(labels, scores)
    metrics["negative_log_likelihood"] = float(
        -np.log(np.clip(scores[np.arange(len(labels)), labels], 1e-15, 1)).mean()
    )
    predictions = scores.argmax(1)
    accepted = scores.max(1) >= 0.95
    correct = predictions == labels
    if components is not None:
        groups = np.unique(components)
        metrics["source_components"] = len(groups)
        metrics["group_macro_top1"] = float(
            np.mean([correct[components == group].mean() for group in groups])
        )
    count = int(accepted.sum())
    metrics["acceptance"] = dict(
        probability_threshold=0.95,
        accepted=count,
        review=int((~accepted).sum()),
        coverage=float(accepted.mean()),
        accepted_correct=int((accepted & correct).sum()),
        accepted_errors=int((accepted & ~correct).sum()),
        accepted_precision=float(correct[accepted].mean()) if count else None,
    )
    if detailed:
        metrics["per_class"] = []
        for label in range(scores.shape[1]):
            mask = labels == label
            tp = int((mask & correct).sum())
            actual, predicted = int(mask.sum()), int((predictions == label).sum())
            metrics["per_class"].append(
                dict(
                    label=label,
                    samples=actual,
                    correct=tp,
                    recall=tp / actual if actual else None,
                    f1=2 * tp / (actual + predicted) if actual + predicted else None,
                )
            )
    return metrics


def cross_validate(data, components, folds, config):
    scores = np.full((len(data.labels), len(data.class_names)), np.nan)
    for train, valid in folds:
        scores[valid] = fit_head(data, train, components, config).predict_proba(
            data.features[valid]
        )
    covered = np.flatnonzero(np.isfinite(scores).all(1))
    return (
        score_metrics(data.labels[covered], scores[covered], components=components[covered]),
        covered,
        scores[covered],
    )


def _rank(record):
    metrics = record["metrics"]
    return (
        -metrics["top1_accuracy"],
        -metrics["macro_f1_supported_classes"],
        metrics["negative_log_likelihood"],
        record["index"],
    )


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_search(
    data: StudyData,
    baseline_path: Path,
    output: Path,
    *,
    candidates: list[SearchConfig] | None = None,
    seed: int = 20261027,
    provenance: dict | None = None,
):
    """Lock development selection before any historical holdout prediction."""
    from threadpoolctl import threadpool_limits

    output.mkdir(parents=True, exist_ok=False)
    candidates = candidate_grid() if candidates is None else candidates
    if not candidates or candidates[0] != candidate_grid()[0]:
        raise ValueError("The baseline must be the first candidate.")
    components = connected_components(data)
    folds, pinned = make_folds(data, components, seed=seed)
    baseline = LinearIdentityHead(baseline_path)
    if baseline.class_names != data.class_names:
        raise ValueError("Baseline identity order differs from dataset.")
    protocol = dict(
        seed=seed,
        confirmation_seed=seed + 1,
        folds=3,
        candidates=[c.as_dict() for c in candidates],
        selection_order=[
            "pooled_oof_top1",
            "supported_class_macro_f1",
            "negative_log_likelihood",
            "candidate_order",
        ],
        historical_test_used_for_selection=False,
        historical_test_scope="exploratory reused holdout; closed-set classification",
        feature_sha256=_hash_arrays(data.features),
        labels_sha256=_hash_arrays(data.labels),
        metadata_sha256=hashlib.sha256(json.dumps(data.rows, sort_keys=True).encode()).hexdigest(),
        baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        provenance=provenance or {},
        pinned_indices=pinned.tolist(),
        pinned_classes=sorted(set(data.labels[pinned].tolist())),
        development_indices=data.fit_indices.tolist(),
        historical_test_indices=data.test_indices.tolist(),
        fold_indices=[dict(train=t.tolist(), validation=v.tolist()) for t, v in folds],
        class_names=data.class_names,
    )
    write_json(output / "protocol.json", protocol)
    records = []
    best_record, best_indices, best_scores = None, None, None
    with threadpool_limits(limits=2):
        for index, config in enumerate(candidates):
            start = time.perf_counter()
            record = dict(index=index, config=config.as_dict())
            try:
                metrics, indices, scores = cross_validate(data, components, folds, config)
                record.update(status="ok", metrics=metrics)
                if best_record is None or _rank(record) < _rank(best_record):
                    best_record, best_indices, best_scores = record, indices, scores
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                if index == 0:
                    write_json(output / "baseline_failure.json", record)
                    raise
            record["seconds"] = time.perf_counter() - start
            records.append(record)
            write_json(output / f"candidate_{index:02}.json", record)
            write_json(output / "candidates.json", records)
            progress = dict(
                candidate=index + 1,
                total=len(candidates),
                config=record["config"],
                status=record["status"],
                seconds=round(record["seconds"], 3),
            )
            if record["status"] == "ok":
                progress.update(
                    {
                        key: record["metrics"][key]
                        for key in (
                            "top1_accuracy",
                            "macro_f1_supported_classes",
                            "negative_log_likelihood",
                        )
                    }
                )
            else:
                progress["error"] = record["error"]
            print(json.dumps(progress), flush=True)
        chosen = min((r for r in records if r["status"] == "ok"), key=_rank)
        winner = candidates[chosen["index"]]
        selected = dict(
            winner=winner.as_dict(),
            candidate_index=chosen["index"],
            selection_metrics=chosen["metrics"],
            locked_before_holdout=True,
        )
        write_json(output / "winner.json", selected)
        np.savez_compressed(
            output / "selection_scores.npz",
            indices=best_indices,
            labels=data.labels[best_indices],
            scores=best_scores,
        )
        confirmation, confirmation_pinned = make_folds(data, components, seed=seed + 1)
        checks = dict(
            seed=seed + 1,
            pinned_indices=confirmation_pinned.tolist(),
            fold_indices=[dict(train=t.tolist(), validation=v.tolist()) for t, v in confirmation],
        )
        for name, config in [("baseline", candidates[0]), ("winner", winner)]:
            metrics, indices, scores = cross_validate(data, components, confirmation, config)
            checks[name] = metrics
            np.savez_compressed(
                output / f"confirmation_{name}_scores.npz",
                indices=indices,
                labels=data.labels[indices],
                scores=scores,
            )
        write_json(output / "confirmation.json", checks)
        fitted = fit_head(data, data.fit_indices, components, winner)
        parity = export_head(
            fitted, data.class_names, output / "classifier.npz", data.features[data.fit_indices]
        )
        # This is the only historical-test prediction stage; selection is already frozen.
        test = data.test_indices
        baseline_scores = baseline.predict_proba(data.features[test])
        winner_scores = LinearIdentityHead(output / "classifier.npz").predict_proba(
            data.features[test]
        )
        truth = data.labels[test]
        old_correct = baseline_scores.argmax(1) == truth
        new_correct = winner_scores.argmax(1) == truth
        holdout = dict(
            baseline=score_metrics(
                truth, baseline_scores, detailed=True, components=components[test]
            ),
            winner=score_metrics(truth, winner_scores, detailed=True, components=components[test]),
            corrected=int((~old_correct & new_correct).sum()),
            regressed=int((old_correct & ~new_correct).sum()),
            both_wrong=int((~old_correct & ~new_correct).sum()),
            scope=protocol["historical_test_scope"],
        )
        np.savez_compressed(
            output / "holdout_scores.npz",
            indices=test,
            sample_ids=np.array([data.rows[i]["sample_id"] for i in test]),
            labels=truth,
            baseline_scores=baseline_scores,
            winner_scores=winner_scores,
        )
        result = dict(
            **selected,
            confirmation=checks,
            holdout=holdout,
            export_max_probability_error=parity,
            classifier_sha256=hashlib.sha256((output / "classifier.npz").read_bytes()).hexdigest(),
        )
        write_json(output / "comparison.json", result)
    return result
