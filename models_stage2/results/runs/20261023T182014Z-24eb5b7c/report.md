# CorVeritas Stage 2: executed evaluation

Run: `20261023T182014Z-24eb5b7c`. Recommended family from validation: **neural**.

All values below are unweighted test-sample metrics for self-reported previous diagnosis history.
The test set was used after model and threshold selection. Scores are uncalibrated.

| Model | Condition | Positive / negative | Threshold | Precision | Sensitivity | Specificity | F1 | ROC-AUC | AP | PR-AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | coronary_heart_disease | 59 / 1103 | 0.1417 | 0.2564 | 0.5085 | 0.9211 | 0.3409 | 0.8765 | 0.2304 | 0.2200 | 0.0432 |
| baseline | heart_failure | 49 / 1113 | 0.1068 | 0.2017 | 0.4898 | 0.9146 | 0.2857 | 0.8542 | 0.2103 | 0.2044 | 0.0366 |
| baseline | heart_attack_history | 47 / 1115 | 0.1045 | 0.2121 | 0.5957 | 0.9067 | 0.3128 | 0.8695 | 0.1931 | 0.1856 | 0.0352 |
| neural | coronary_heart_disease | 59 / 1103 | 0.1480 | 0.2424 | 0.5424 | 0.9093 | 0.3351 | 0.8651 | 0.2249 | 0.2164 | 0.0430 |
| neural | heart_failure | 49 / 1113 | 0.1474 | 0.2099 | 0.3469 | 0.9425 | 0.2615 | 0.8514 | 0.1956 | 0.1836 | 0.0367 |
| neural | heart_attack_history | 47 / 1115 | 0.1364 | 0.1961 | 0.4255 | 0.9265 | 0.2685 | 0.8727 | 0.2249 | 0.2184 | 0.0345 |

## Interpretation

Average precision (AP) is shown separately from trapezoidal precision-recall AUC. The sample positive fraction is the no-skill AP reference. High ROC-AUC alone can coexist with low precision under strong class imbalance.

The saved recommended family is determined by validation macro AP. Test results do not change that selection. Differences between models should not be treated as significant without suitable uncertainty analysis and external replication.

Subgroup tables, confusion matrices, ECE and log loss are in test_evaluation.json. Age/sex cells require at least 30 participants and 5 of each class; smaller cells are explicitly withheld. Sparse positive cases make calibration and subgroup findings unstable.

Plots: `/Users/ritzmk/Documents/CorVeritas/models_stage2/results/plots/20261023T182014Z-24eb5b7c`.

## Limitations

- Research prototype, not a validated diagnostic tool or a substitute for medical assessment.
- Scores estimate associations with self-reported previous diagnoses, not future disease onset.
- Uncalibrated model scores; positive weighting can materially alter probability calibration.
- Cross-sectional measurements may follow diagnosis, treatment or lifestyle changes; contributions are not causal.
- Unweighted NHANES sample evaluation is not a US population estimate; survey design is not modeled.
- Complete-label selection, missing measurements, small positive counts and US sampling limit generalization.
