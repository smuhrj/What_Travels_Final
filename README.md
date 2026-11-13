# What Travels?
## Projecting Mid-Major Pitcher Performance in Power Four Baseball

When a pitcher transfers from a mid-major program to a Power Four school, which parts of his performance carry over? This project measures how pitching outcomes change in the first Power Four season and tests whether pre-transfer statistics improve individual projections beyond simple benchmarks.

The dataset contains 241 Division I pitcher-seasons from 70 pitchers who transferred from the Atlantic 10, Big South, Coastal Athletic Association, Southern Conference, or America East into the ACC, Big Ten, Big 12, or SEC in the 2024–2026 entry cohorts. Every pitcher in the sample eventually recorded at least 10 Power Four innings. The main analysis uses each pitcher’s actual first Power Four season, including the three entry seasons below 10 innings.

## Run the project

1. Keep [`power4_pitcher_final_project_reviewer.ipynb`](power4_pitcher_final_project_reviewer.ipynb) and the supplied Excel workbook in the same folder.
2. Open the notebook in VS Code or Jupyter and select a Python kernel. If packages are missing, uncomment and run the optional installation cell.
3. Choose a validation setting in the notebook. Leave `RUN_FULL_PAPER_VALIDATION = False` for the faster review run, which uses two outer repetitions. Set it to `True` for the ten-repeat configuration used for the paper.
4. Select **Run All**. The notebook creates `results_final/` and saves the analysis outputs there.
5. Open `results_final/paper_findings.md` for a summary, then review the validation tables and figures.

The notebook includes executed tables and figures from the faster review run. Validation metrics can differ between settings. Both settings write to the same output folder, so save a copy of the outputs if you want to compare runs. The run manifest records the workbook fingerprint, sample counts, settings, and package versions.

## How the analysis works

- **Data audit.** The notebook checks player identifiers, season ordering, joins, missing values, and counting statistics. It recomputes pitching rates from the underlying counts and converts baseball innings notation to true decimal innings.
- **First-season comparisons.** The notebook pairs each pitcher’s source season with his first Power Four season, estimates average changes, and measures whether pitchers retain their relative standing on each statistic. A separate descriptive table uses the original qualifying-season pairs for comparison with the submitted abstract.
- **Predictive validation.** Each repetition uses five outer folds and three inner folds for model selection. On the same held-out pitchers, the notebook compares the selected workflow with the training mean, training median, unchanged pre-transfer statistic, average-change translation, and single-statistic ridge regression.
- **Feature comparisons.** Fixed ridge models compare the matching pre-transfer statistic alone, that statistic plus role and workload features, and a broader set of pitching statistics. The models use identical outer splits for paired comparisons.
- **Workload sensitivity.** The notebook refits the workflow on subsets of pitchers who recorded at least 10 or 20 innings in their first Power Four season. These retrospective subsets are defined by observed workload.
- **Chronological check.** The notebook trains on earlier cohorts and evaluates predictions on the previously inspected 2026 cohort. A separate split-conformal model uses a calibration subset to estimate prediction ranges. The point models used in the pitcher demo are fit on all available paired data.

## Main findings

Pitchers in the sample threw **21.75 fewer innings on average** in their first Power Four season than in their source season. The 95% paired bootstrap interval corresponds to approximately **16.31 to 26.95 fewer innings**.

The predictive results below are from the **ten-repeat paper configuration**. MAE is mean absolute error.

| Outcome | Selected workflow MAE | Training-median MAE | Interpretation |
|---|---:|---:|---|
| Innings pitched | 14.49 innings | 16.30 innings | Average error fell by 11%. The paired uncertainty interval includes zero, and the advantage narrowed in both workload subsets. |
| Walk percentage | 3.35 percentage points | 3.45 percentage points | Ridge using prior walk percentage alone had the lowest MAE of these three comparisons, at 3.19 percentage points. |
| Strikeout percentage | 5.55 percentage points | 5.39 percentage points | The training median had lower error. |
| ERA | 1.74 ERA units | 1.71 ERA units | The training median had lower error. |

In the fixed ridge comparison for innings pitched, adding role and pitching statistics reduced MAE from **15.87 to 13.80 innings** relative to using prior innings alone. The broader model had lower error in **43 of 50 outer folds**. The folds share pitchers and training data, so the count describes consistency across splits.

## Results files

The notebook writes these files to `results_final/`.

| File | Contents |
|---|---|
| `paper_findings.md` | Summary of the descriptive results, benchmark comparisons, and sensitivity checks for the selected run. |
| `first_entry_changes_portability.csv` | Paired first-season changes and rank correlations with bootstrap intervals. |
| `qualifying_changes_portability.csv` | Descriptive comparisons using the original qualifying-season pairs. |
| `nested_cv_metrics.csv` | Error metrics for nested model selection, benchmarks, and fixed feature groups. |
| `nested_cv_paired_comparisons.csv` | Paired MAE reductions. Positive values favor the selected workflow. |
| `full_features_vs_benchmarks.csv` | Comparisons between fixed full-feature ridge and benchmark models. |
| `workload_feature_gains.csv` | Paired changes in innings prediction error from adding feature groups. |
| `model_selection_stability.csv` | Model selection counts and shares across the outer folds for each outcome. |
| `nested_repeat_stability.csv` | Variation in model error across validation repetitions. |
| `chronological_2026_metrics.csv` | Prediction errors for the chronological 2026 check. |
| `chronological_2026_paired_comparisons.csv` | Paired error differences for the 2026 cohort. |
| `interval_coverage.csv` | Observed coverage and average width of the separately calibrated split-conformal ranges. |
| `projection_residual_sd.csv` | Held-out residual standard deviations and pitcher counts used for the demo ranges. |
| `descriptive_sensitivity.csv` | Workload-threshold estimates and leave-one-pitcher-out influence ranges. |
| `predictive_workload_sensitivity.csv` | Model errors for the full sample and the 10- and 20-inning first-season subsets. |
| `figure_1_changes.png` through `figure_5_workload_sensitivity.png` | Five figures exported at 300 DPI. |
| `run_manifest.json` | Input fingerprint, sample counts, analysis settings, and software versions. |

The CSVs store rates such as K% and BB% as proportions. A walk-rate error of `0.0345` equals 3.45 percentage points. The CSVs store innings as true decimal innings.

## Project a new pitcher

1. Run the notebook once, then go to **Try a pitcher projection** at the bottom.
2. Replace **Joe Random** and his example values with the pitcher’s pre-transfer season totals. Enter innings in baseball notation. For example, `60.2` means 60 innings and two outs.
3. Rerun **only the final cell**. It calculates the pitching rates and displays projections for all 11 outcomes. Enter the totals directly in the cell.

The demo displays each point projection with a **±1 standard deviation range**. The standard deviation comes from held-out residuals in nested validation, averaged across repetitions for each historical pitcher before calculating the sample standard deviation. The endpoints are clipped to feasible bounds. This range describes historical prediction error and carries no fixed coverage guarantee. The split-conformal ranges reported in the evaluation use a separate calibration procedure.

## Interpretation limits

- **Sample selection.** The study includes pitchers who eventually reached 10 Power Four innings. Estimating a recruit’s chance of earning a role would require data on eligible transfers who pitched little or never appeared.
- **Validation.** Nested cross-validation estimates performance across the observed cohorts. Repetitions reuse the same pitchers. The previously inspected 2026 cohort serves as a chronological check.
- **Uncertainty.** Paired bootstrap intervals summarize variation across pitchers and capture only part of the uncertainty from fitting models on overlapping samples. The 11 outcomes are related and exploratory. Leave-one-out influence ranges describe how estimates change when a pitcher is omitted.
- **Prediction ranges.** Split-conformal coverage depends on exchangeability between calibration cases and future pitchers. Changes in cohort composition can affect coverage, and some ranges are wide. Each target is modeled separately, so projected statistics may not reconcile into a single stat line.
- **Interpretation of changes.** The paired comparisons measure what changed after a transfer. They cannot isolate the effect of competition from changes in role, opportunity, development, health, or other factors.

A future study could add eligible transfers who recorded few or zero innings, verify their roster and season records, and reserve a future cohort for independent evaluation. Pitch-level measurements could then be tested for added predictive value.

## Reproducibility

The reviewer notebook was run end to end on the supplied 70-pitcher workbook in the two-repeat configuration. A check against the previous implementation reproduced innings and walk-rate predictions on identical folds. The workbook’s **Definitions & Sources** sheet contains its provenance links. Resampling seeds are fixed, and the run manifest records the settings and software versions used.
