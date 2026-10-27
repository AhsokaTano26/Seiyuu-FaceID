# Classifier optimization implementation plan

**Goal:** Improve classification accuracy and speed using reproducible development-only model selection and measured runtime changes.

**Architecture:** Compare portable logistic heads on the existing frozen AdaFace/R50 features. Use source-disjoint development folds, retain singleton identities only in training, lock the winner before inspecting historical test predictions, and export a separate NumPy head compatible with the current runtime. Run the independent frozen backbones concurrently using a persistent executor, preserving feature order and allowing a sequential fallback.

**Tech stack:** Existing NumPy/scikit-learn environment; no new runtime dependencies.

## Design and constraints

- Optimize Top-1 first, pooled supported-class macro-F1 second, log loss third; keep deterministic candidate order as final tie-break.
- Development is original train + validation (4,064 faces). Historical test (1,051 faces) never contributes to preprocessing, candidate selection, or fitting.
- Group connected samples by source group, source checksum, crop checksum, sample ID and optional book group. Reject components spanning development/test or conflicting identities.
- Pin identities with one component in every training fold; exclude them from validation statistics. All folds must retain all identities in training.
- Compare standardized LR regularization/class weights, feature block weights applied after standardization, unstandardized LR, and equal source-component weighting. These export to the existing affine-softmax head. Nonlinear models would require a different runtime and are deferred from this bounded experiment.
- Write protocol, fold indices, candidate metrics, selected configuration, final head, holdout comparison and provenance under a new ignored `data/experiments/` directory. Do not overwrite the deployed bundle or publish private data.
- Existing historical test was previously used in model selection; any result on it remains exploratory. No unknown-identity accuracy claim is possible without labeled unknown examples.

## Work

- [x] Add tests for connected-group leakage, pinned singletons, complete OOF coverage, holdout-independent selection, fold-local preprocessing, export/runtime parity and output overwrite refusal.
- [x] Implement `experiments/classifier_search.py` and `tools/optimize_classifier.py` with a fixed 50-candidate grid and reproducible 3-fold selection (seed 20261027).
- [x] Run the search, freeze its result, compare winner and baseline on a second development fold seed (20261028), then evaluate the fixed winner once on historical test.
- [x] Benchmark sequential/parallel backbone inference with interleaved runs and probability parity checks.
- [x] Add optional persistent parallel execution, configuration validation, exception draining and API shutdown cleanup; enable it in the supplied configuration while preserving two intra-op threads per backbone.
- [x] Re-extract every historical holdout crop with the runtime optimization and verify recorded predictions and probability parity.
- [x] Verify candidate against `LinearIdentityHead`, run the full suite and lint, independently review changes and write a concise measured report.

## Review focus

Transitive duplicate grouping; mixed-label components; rare identities; multiclass probability export; validation metrics excluding pinned samples; source-group weighting fitted only within a fold; historical-test selection leakage; private artifacts remaining ignored.

## Progress

Baseline reproduced: 1,035/1,051 Top-1; 1,011 accepted at 0.95 with 4 wrong. Dataset audit found two singleton identities in the development pool and no development/test duplicate-component overlap.

User clarified accuracy and speed both matter. Fixed search selected standardized LR C=0.1, balanced classes, AdaFace weight 0.75. Selection OOF correct improved 3,977→3,986; second-seed OOF 3,981→3,988; historical test regressed 1,035→1,033. Keep deployed head. Historical test remains exploratory; the second development seed is not an independent test.

Local paired benchmark: same intra-op=2, sequential embedding mean 78.39 ms vs concurrent 54.22 ms on 16 development source groups × 3 runs; exact feature/probability parity. Enable the measured concurrent mode, retain a configuration switch for resource-constrained hosts, and verify full holdout through actual runtime before completion.

Actual parallel runtime verified on all 1,051 historical holdout crops: exact sequential/parallel features and probabilities; all historical Top-1 and threshold decisions preserved. Mean embedding latency 77.61→52.12 ms (−32.8%), P95 79.35→55.40 ms. Independent review of search and runtime found no important issues. Full suite 131 passed; deployed head checksum unchanged. Report: `docs/CLASSIFIER_OPTIMIZATION.md`.
