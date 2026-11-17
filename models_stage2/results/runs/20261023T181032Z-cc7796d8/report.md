# CorVeritas Stage 2: executed evaluation

Run: `20261023T181032Z-cc7796d8`. Recommended family from validation: **neural**.

All values below are unweighted test-sample metrics for self-reported previous diagnosis history.
The test set was used after model and threshold selection. Scores are uncalibrated.

| Model | Condition | Positive / negative | Threshold | Precision | Sensitivity | Specificity | F1 | ROC-AUC | AP | PR-AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | coronary_heart_disease | 59 / 1103 | 0.1417 | 0.2564 | 0.5085 | 0.9211 | 0.3409 | 0.8765 | 0.2304 | 0.2200 | 0.0432 |
| baseline | heart_failure | 49 / 1113 | 0.1068 | 0.2017 | 0.4898 | 0.9146 | 0.2857 | 0.8542 | 0.2103 | 0.2044 | 0.0366 |
| baseline | heart_attack_history | 47 / 1115 | 0.1045 | 0.2121 | 0.5957 | 0.9067 | 0.3128 | 0.8695 | 0.1931 | 0.1856 | 0.0352 |
| neural | coronary_heart_disease | 59 / 1103 | 0.1483 | 0.2422 | 0.5254 | 0.9121 | 0.3316 | 0.8639 | 0.2103 | 0.2037 | 0.0432 |
| neural | heart_failure | 49 / 1113 | 0.1337 | 0.1919 | 0.3878 | 0.9281 | 0.2568 | 0.8489 | 0.1857 | 0.1769 | 0.0370 |
| neural | heart_attack_history | 47 / 1115 | 0.1541 | 0.2317 | 0.4043 | 0.9435 | 0.2946 | 0.8706 | 0.2243 | 0.2176 | 0.0345 |

## Interpretation

Average precision (AP) is shown separately from trapezoidal precision-recall AUC. The sample positive fraction is the no-skill AP reference. High ROC-AUC alone can coexist with low precision under strong class imbalance.

The saved recommended family is determined by validation macro AP. Test results do not change that selection. Differences between models should not be treated as significant without suitable uncertainty analysis and external replication.

Subgroup tables, confusion matrices, ECE and log loss are in test_evaluation.json. Age/sex cells require at least 30 participants and 5 of each class; smaller cells are explicitly withheld. Sparse positive cases make calibration and subgroup findings unstable.

Plots: `/Users/ritzmk/Documents/CorVeritas/models_stage2/results/plots/20261023T181032Z-cc7796d8`.

## Limitations

- Research prototype, not a validated diagnostic tool or a substitute for medical assessment.
- Scores estimate associations with self-reported previous diagnoses, not future disease onset.
- Uncalibrated model scores; positive weighting can materially alter probability calibration.
- Cross-sectional measurements may follow diagnosis, treatment or lifestyle changes; contributions are not causal.
- Unweighted NHANES sample evaluation is not a US population estimate; survey design is not modeled.
- Complete-label selection, missing measurements, small positive counts and US sampling limit generalization.
