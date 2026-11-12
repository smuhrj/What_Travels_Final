# What Travels?
## Projecting Mid-Major Pitcher Performance in Power Four Baseball

When a pitcher transfers from a mid-major program to a Power Four school, which parts of his performance carry over? This project measures how pitching outcomes change in the first Power Four season and tests whether pre-transfer statistics improve individual projections beyond simple benchmarks.

The dataset contains 241 Division I pitcher-seasons from 70 pitchers who transferred from the Atlantic 10, Big South, Coastal Athletic Association, Southern Conference, or America East into the ACC, Big Ten, Big 12, or SEC in the 2024–2026 entry cohorts. Every pitcher in the sample eventually recorded at least 10 Power Four innings. The main analysis uses each pitcher’s actual first Power Four season, including the three entry seasons below 10 innings.

## Run the project

1. Place `power4_pitcher_final_project_completed.ipynb` and the supplied Excel workbook in the same folder.
2. Open the notebook in VS Code or Jupyter and select a Python kernel.
3. If needed, uncomment and run the optional package installation cell.
4. Select **Run All**. The full run can take several minutes.
5. Start with `results_final/paper_findings.md`, then review the validation tables and figures.

The notebook contains the complete analysis and includes the tables and figures from the completed run. `results_final/run_manifest.json` records the workbook fingerprint, sample counts, analysis settings, and package versions used for your run.

## How the analysis works

- **Data audit.** The notebook checks player identifiers, season ordering, joins, missing values, and counting statistics. It recomputes pitching rates from the underlying counts and converts baseball innings notation to true decimal innings.
- **First-season comparisons.** The notebook pairs each pitcher’s source season with his first Power Four season. It estimates average changes and measures whether pitchers retain their relative standing on each statistic. A separate descriptive table uses the original qualifying-season pairs for comparison with the submitted abstract.
- **Predictive validation.** The notebook uses 10 repetitions of five-fold outer cross-validation, with three-fold inner cross-validation for model selection. The outer folds compare the selected workflow against the training mean, training median, unchanged pre-transfer statistic, average-change translation, and single-statistic ridge regression. Each comparison uses predictions for the same pitchers within a fold.
- **Feature comparisons.** Fixed ridge models compare the matching pre-transfer statistic alone, that statistic plus role and workload features, and a broader set of pitching statistics. The models use the same outer splits so their errors can be compared pitcher by pitcher.
- **Workload sensitivity.** The notebook refits the workflow on subsets of pitchers who recorded at least 10 or 20 innings in their first Power Four season. These thresholds define retrospective groups using observed outcomes.
- **Chronological check and prediction ranges.** The notebook trains on the earlier cohorts and evaluates predictions on the previously inspected 2026 cohort. A separate split-conformal model uses a calibration subset to estimate prediction ranges. The final point models are refit on all available paired data for optional candidate projections.

## Main findings

Pitchers in the sample threw **21.75 fewer innings on average** in their first Power Four season than in their source season. The 95% paired bootstrap interval corresponds to approximately **16.31 to 26.95 fewer innings**.

| Outcome | Selected workflow MAE | Training-median MAE | Interpretation |
|---|---:|---:|---|
| Innings pitched | 14.49 innings | 16.30 innings | The selected workflow reduced average error by 11%. The paired uncertainty interval includes zero, and the advantage narrowed in both workload subsets. |
| Walk percentage | 3.35 percentage points | 3.45 percentage points | A ridge model using prior walk percentage alone performed better than both, with an MAE of 3.19 percentage points. |
| Strikeout percentage | 5.55 percentage points | 5.39 percentage points | The training median had lower error. |
| ERA | 1.74 ERA units | 1.71 ERA units | The training median had lower error. |

In the fixed ridge comparison for innings pitched, adding role and pitching statistics reduced MAE from **15.87 to 13.80 innings** relative to using prior innings alone. The gain appeared in **43 of 50 outer folds**. These folds share pitchers and training data, so that count describes consistency across splits.

## Results files

The notebook writes these files to `results_final/`.

| File | Contents |
|---|---|
| `paper_findings.md` | Summary of the descriptive results, benchmark comparisons, and sensitivity checks. |
| `first_entry_changes_portability.csv` | Paired first-season changes and rank correlations with bootstrap intervals. |
| `qualifying_changes_portability.csv` | Descriptive comparisons using the original qualifying-season pairs. |
| `nested_cv_metrics.csv` | Error metrics for nested model selection, benchmarks, and fixed feature groups. |
| `nested_cv_paired_comparisons.csv` | Paired differences in MAE. Positive reductions favor the selected workflow. |
| `full_features_vs_benchmarks.csv` | Comparisons between fixed full-feature ridge and benchmark models. |
| `workload_feature_gains.csv` | Paired changes in innings prediction error from adding feature groups. |
| `model_selection_stability.csv` | Model selection counts and shares across the 50 outer folds per outcome. |
| `nested_repeat_stability.csv` | Variation in model error across the 10 validation repetitions. |
| `chronological_2026_metrics.csv` | Prediction errors for the chronological 2026 check. |
| `chronological_2026_paired_comparisons.csv` | Paired error differences for the 2026 cohort. |
| `interval_coverage.csv` | Observed coverage and average width of the separately calibrated prediction ranges. |
| `descriptive_sensitivity.csv` | Workload-threshold estimates and leave-one-pitcher-out influence ranges. |
| `predictive_workload_sensitivity.csv` | Model errors for the full sample and the 10- and 20-inning first-season subsets. |
| `figure_1_changes.png` through `figure_5_workload_sensitivity.png` | Five figures exported at 300 DPI. |
| `run_manifest.json` | Input fingerprint, sample counts, analysis settings, and software versions. |

Rate values such as K% and BB% are stored as proportions in the CSVs. For example, a walk-rate error of `0.0345` equals 3.45 percentage points. Innings are stored as true decimal innings.

## Project a new pitcher

The final projection cell is optional. To use it:

1. Open `results_final/new_pitcher_input_template.csv`.
2. Add one row per pitcher using statistics from his pre-transfer source season. Keep the template’s column names.
3. Save the filled file as `my_candidates.csv` beside the notebook, or use its full path in the cell below.

```python
candidates = pd.read_csv("my_candidates.csv")
projected = project_pitchers(candidates, results["bundle"])
projected.to_csv(OUT / "candidate_projections.csv", index=False)

projected[["Player", "projected_IP", "projected_BB%", "projected_K%", "projected_ERA"]]
```

Enter strikeout and walk percentages as proportions, such as `0.24` and `0.08`. Convert baseball innings notation before entering `source_IP`. For example, `60.2` in a box score means 60⅔ innings, so enter approximately `60.6667`. Calculate `source_start_rate` as games started divided by appearances and `source_ip_per_game` as true innings divided by appearances. Set `year_gap` to the first Power Four season minus the source season year.

The output contains point projections and separately calibrated prediction ranges. The ranges have their own model centers, which can differ from the final point projections.

## Interpretation limits

- **Sample selection.** The study includes pitchers who eventually reached 10 Power Four innings. The findings apply to this selected group. Estimating a recruit’s chance of earning a role would require data on eligible transfers who pitched little or never appeared.
- **Validation.** Nested cross-validation estimates performance across the observed cohorts. The 10 repetitions reuse the same pitchers. The 2026 cohort was previously inspected and serves as a chronological check.
- **Uncertainty.** Paired bootstrap intervals summarize variation across pitchers and capture only part of the uncertainty from fitting models on overlapping samples. The 11 outcomes are related and exploratory. Leave-one-out influence ranges describe how estimates change when a pitcher is omitted.
- **Prediction ranges.** Split-conformal coverage depends on exchangeability between calibration cases and future pitchers. Changes in cohort composition can affect coverage, and some ranges are wide. The all-data point models are fit separately. Because each target is modeled on its own, projected statistics may not reconcile into a single stat line.
- **Interpretation of changes.** The paired comparisons measure what changed after a transfer. They cannot isolate the effect of competition from changes in role, opportunity, development, health, or other factors.

The most valuable extension would be to add eligible transfers who recorded few or zero innings, verify their roster and season records, and reserve a future cohort for independent evaluation. Pitch-level measurements could then be tested for added predictive value.

## Reproducibility

Every cell in the completed notebook was run on the supplied 70-pitcher workbook. Verification covered training-only medians, matched pitcher predictions across models, workload subset membership, and model-selection counts. The workbook’s **Definitions & Sources** sheet contains its provenance links. The run manifest records reproducible settings, and the resampling seeds are fixed.