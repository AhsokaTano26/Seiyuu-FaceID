# Training image count study

[English](2026-10-21-sample-size.md) | [简体中文](2026-10-21-sample-size.zh-CN.md) | [日本語](2026-10-21-sample-size.ja.md)

This file records the protocol and findings from 2026-10-21.

The goal was to estimate how many training images per identity were needed for at least 95% correct-return rate under the frozen dual-backbone and logistic-regression pipeline at probability threshold 0.95.

The user requested the experiment and chose “at least 95% correct identification per identity.” This exploratory study uses the existing private data, with the production model, training pool and running API managed separately.

## Protocol

- Data are `data/training/features_and_splits.npz` and `samples.jsonl`, with pretrained features fixed.
- Fit on original train + validation and test on original test. Align sample IDs and verify indices, labels, source SHA, crop SHA and source-group isolation. Record current file SHA256.
- Two hashes in `holdout_protocol.json` differ from current files. Preserve that finding and establish this study's evidence using array/row consistency, alignment between deployed head and saved scores, prediction agreement after complete refitting, and a new hash snapshot.
- One training image means one reviewed face selected from a source_group. This limits overcounting burst/book images as independent sources. Groups approximate the current duplicate strategy and do not establish fully independent photo shoots.
- All runs train all 50 classes. Target identities use N images per person simultaneously. The other identities keep their complete fixed fit rows. Fit the scaler only on the selected rows in that run.
- The 100-image cohort has 9 identities with at least 100 training source groups and 10 test groups.
- The 50-image cohort has 12 identities with at least 50 training groups and 10 test groups.
- Analyze the cohorts separately because their populations differ.
- Counts are 1/2/3/5/8/10/15/20/30/50/75/100, truncated at 50 for the expanded cohort.
- Run 30 repeats per cohort with seeds starting at 20261021. Each repeat uses prefixes of the same randomized order across N, including fixed representatives of selected groups. Preserve traceable sampling records.
- Select one representative per test identity/source group with a separate fixed seed, identical across N, repeats and cohorts.
- Use StandardScaler + LogisticRegression(C=1, class_weight=balanced, max_iter=1500, random_state=20261014), matching the deployed head. Limit numerical threads to 2.
- Report Top-1, correct return at probability ≥0.95, false acceptance, review and accepted-result precision. Review counts as unsuccessful. The denominator is each identity's fixed test-source count.
- The observed minimum count is the smallest tested N where all 30 repeats and every larger tested N achieve ≥95% for that identity. Return null if none passes. Report individual identities and repeats.
- Also report the worst one-sided 95% Clopper–Pearson lower bound over runs. Repeats reuse the independent tests and are not pooled into an enlarged test count. Give a stricter count for lower bounds ≥95%.
- The holdout was previously explored. Intervals are conditional descriptions. Unknown rejection and full-API detection/quality failures require separate evaluation.

## Deliverables and checks

- [x] `src/seiyuu_faceid/experiments/sample_size.py` implements validation, grouped sampling, metrics and study loops.
- [x] `tools/sample_size_study.py` provides a repeatable CLI writing to an ignored local directory.
- [x] `tests/test_sample_size_study.py` covers source isolation, nested sampling, thresholds, denominators, intervals and sustained success.
- [x] Run failing tests, implement, then verify using the locked training extra.
- [x] Run 660 initial fits across two cohorts and 480 expanded fits, saving hashes, sampling, predictions, summaries and per-identity results.
- [x] Recalculate findings, review design/implementation, and run all tests (103 passed, Ruff passed).
- [x] Produce an HTML report with conclusions, scope, per-identity results, methods, limits and further collection plan.

## Progress

- The 50 classes contain 5,115 samples, with 4,064 fit and 1,051 test. Cross-pool intersections for all four key fields are 0.
- Complete refit takes 0.083 seconds, reproduces deployed Top-1 predictions, and differs in probability by at most about 2.1e-7.
- The head's original 1,051-face baseline is Top-1 98.4776%, correct-return rate 95.8135%, and 40 reviews.
- All 660 initial fits completed. Only 4/12 identities in the 50-image cohort and 4/9 in the 100-image cohort passed every repeat.
- After inspecting results, the five unsuccessful identities in the 100-image cohort each had at least 200 fit groups. An additional fixed five-person cohort tested 125/150/175/200, adding 480 fits, with all 50 classes and unchanged test representatives. This is a later exploratory extension.
- Independent review reproduced metrics, intervals, sustained success, nested sampling and fixed test representatives. No implementation defect was found. The original 1,051-crop baseline and source-group subcohorts use different populations, and passing 30/30 repeats does not establish 95% reliability for future samples.
- In the added cohort, only 近藤天音 passed all repeats at 200. The lowest tested sustained-success count was 175. 上坂堇, 爱美, 相羽爱奈 and 远藤祐里香 still had failed repeats at 200.
- Per-identity intervals, aggregates, worst repeats and minimum counts were cross-checked independently and against SQLite raw count aggregation.
- Before/after SHA256 values of data, deployed head and manifest agree.
- Outputs are `data/experiments/sample_size_20261021/report.html`, an `analysis.ipynb` executed top-to-bottom with four code cells, CSV, NPZ, samples and source fingerprints.
- HTML package and structure validation passed. Chromium headless-shell was unavailable, so semantic-table fallback was retained according to the packaging tool. Browser visuals and source-popover interactions were unverified. The report runs independently of network and local API services.
