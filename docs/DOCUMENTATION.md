# Experiment documentation

This file explains how to validate, run, resume and interpret the bearing-prognosis experiment. This publication contains code and documentation. Supply benchmark data, trained checkpoints and detailed experiment artifacts separately; the code can validate them locally without publishing them.

## Installation and execution

Clone the repository, create a Python 3.12 virtual environment and install `requirements.txt`. Windows, Linux and macOS use the same Python entry points; the environment activation command differs as shown in the root README. The recorded full run used Windows, Python 3.12.14 and four configured computation threads. GPU hardware is unnecessary for the implemented CPU learners. Scientific versions are NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1, matplotlib 3.11.2 and joblib 1.6.0.

From the repository root:

```bash
# Default action: validate the data and return without fitting.
python run_experiments.py

# Recalculate metrics from a separately supplied extracted run.
python scripts/validate_results.py --run-dir path/to/extracted_run

# Fit only small models for deliberate interruption/recovery verification.
python run_experiments.py --verify-recovery --output runs/recovery_check

# Execute all four scopes with the fixed protocol.
python run_experiments.py --train --output runs/run_v1

# Audit a newly completed run.
python scripts/validate_results.py --run-dir runs/run_v1

# Run repository checks, including tiny checkpoint recovery fits.
python -m unittest discover -s tests -v
```

Run only one writer for a given output directory. To explore different scientific settings, use the engine's Python `run_all` interface with a configuration and a new output directory; this changes the experiment and must be documented separately from the reference protocol. The repository wrapper intentionally validates the exact recorded benchmark dataset. It rejects an altered CSV rather than silently treating different data as the same experiment.

The included notebook embeds the same experiment engine for a self-contained Colab workflow. Its configuration, inspection widgets and figures are interactive. `ipywidgets` is optional; a basic plot is displayed if widgets are unavailable. A local notebook environment additionally needs Jupyter/IPython installed; the command-line workflow uses only the pinned requirements. If changing the standalone engine, synchronise the embedded engine before treating the notebook as the same implementation.

## Dataset and target

The dataset has 15 independent bearings, five per operating condition. The conditions pair 2100 rpm with 12 kN, 2250 rpm with 11 kN, and 2400 rpm with 10 kN. Each original snapshot contains 32,768 horizontal and vertical vibration samples at 25.6 kHz; snapshots are recorded every minute.

The required feature dataset contains 9,216 derived rows with 58 columns and is not tracked in Git. Fifty columns are engineering/context predictors; eight contain labels or traceability metadata. Reserving the initial ten snapshots of each bearing leaves 9,066 eligible prediction rows. Initial history is used for one normalisation variant and excluded from evaluation for every variant.

The response is `max_recorded_cycle - current_cycle`, in minutes. `elapsed_minutes` starts at zero. Labels use the known retrospective endpoint; that endpoint is never provided to the model as an input. Bearing identity, run, source path/hash, true RUL and observed total lifetime are excluded from the predictor matrix. Inputs consist of current signal descriptors, elapsed observed time, rpm and radial load.

Feature CSV SHA256:

```text
31993f6277f598e99defc14d152e36aebf5d6d458a14a5cb26b2cc4f061a02d6
```

Raw ZIP SHA256:

```text
3cc815649a315ac7da202980c489f33db44ca2db0317bbe3bcb9dcf415375e10
```

Original provider: https://biaowang.tech/xjtu-sy-bearing-datasets/

Pinned acquisition mirror: `DavidNguyen/XJTU-SY_Bearing_Datasets` on Hugging Face, revision `2742263`. The manifest identifies the mirror as a retrieval route, not as the original measurement source. Each CSV row includes its raw snapshot path and SHA256.

### Optional raw reproduction

These commands download approximately 5.45 GB and process every waveform. They are unnecessary when a verified feature CSV has been supplied separately:

```bash
python scripts/download_raw.py
python scripts/extract_features.py
python run_experiments.py --validate-only
```

The downloader uses resumable range segments under `data/raw/`, checks the final archive hash and preserves the archive. Extraction reads from ZIP with CRC checks and stores resumable JSONL records in `data/feature_cache/`. It uses six worker processes and writes `data/XJTU_SY_features.csv` and its manifest. Preserve an existing local CSV before deliberately regenerating it; the wrapper expects the recorded byte-level checksum. Do not run multiple extraction processes against the same cache. Raw ZIPs, segments and cache records are excluded from Git.

## Representations

| Variant | Predictors | Construction |
|---|---:|---|
| `basic` | 9 | Mean, standard deviation and RMS for both channels, plus elapsed time, rpm and load |
| `engineered` | 50 | Basic context with statistical, impulsiveness, spectral, shaft-order and envelope descriptors, plus channel RMS ratio |
| `engineered_baseline_normalised` | 50 | Engineering signal descriptors normalised using that bearing's first ten snapshots; context unchanged |

Spectra use centred signals, a Hann window and real FFT. Broad power fractions cover 0–1, 1–5 and 5–12.8 kHz. Shaft-order power covers orders one through four with a ±0.2-order band. Envelope descriptors use the analytic-signal magnitude. Geometry-specific defect frequencies are not assumed.

For each signal feature, initial-history scaling uses `(x - median) / max(MAD, 0.05*abs(median), 1e-6)`. Only the first ten snapshots supply the median and MAD. The assumed initial reference is not independently verified healthy operation. The transformation uses no future trajectory extrema or target labels.

## Fitting and selection

The response is transformed with `log1p` during fitting. Predictions use `expm1`, a fixed numerical log ceiling of 15 and nonnegative clipping. No test lifetime is used to cap predictions. Each fitting bearing receives equal total sample weight; observations within a bearing receive equal weights.

Six candidates are compared for each representation and scope:

| Family | Candidate settings | Fixed budget |
|---|---|---|
| Ridge | Alpha 1 or 10, fitting-only weighted standardisation | Two candidates |
| Random Forest | Depth 8 / minimum leaf 3; unrestricted depth / minimum leaf 5 | 200 trees |
| XGBoost | Depth 2 / learning rate 0.05; depth 4 / learning rate 0.03 | 300 rounds |

Seed is 42 and thread budget is four. Boosting uses histogram trees, squared-error regression, no row or column subsampling. Configuration selection minimises unweighted mean per-bearing validation MAE in minutes. It occurs independently for each representation. Comparing their later test errors does not constitute validation-selected deployment of an overall winning representation.

### Evaluation boundaries

| Scope | Selection fitting | Validation | Final fitting | Test |
|---|---|---|---|---|
| `main` | Runs 1–3 from all conditions, 9 bearings | Run 4, 3 bearings | Runs 1–4, 12 bearings | Run 5, 3 bearings |
| Each `loco_N` | Source-condition runs 1–3, 6 bearings | Source run 4, 2 bearings | All source runs, 10 bearings | All 5 bearings in condition N |

Test-condition labels never participate in outer-fold selection. A run-five bearing may be a source example in a separate outer fold; these are different evaluations and are not pooled as independent repetitions. Splits are recorded in `*_split.json`.

Baselines are a training-only constant, calculated as the mean of per-bearing mean eligible RUL, and training condition-median lifetime minus observed age. For an unseen condition, the age prior uses the overall source median lifetime.

## Metrics and explanations

MAE, RMSE and R-squared are calculated per bearing. Macro MAE and macro RMSE are unweighted means of those per-bearing metrics. Macro RMSE is not the square root of a macro MSE. No significance claim is made from three main test bearings. Negative R-squared values and individual errors are retained.

Near-endpoint precision, recall and F1 use the condition `true RUL <= 30 minutes`, with the same threshold for predicted RUL. These are endpoint-proximity proxy metrics. They do not demonstrate fault-category diagnosis or independently verified early fault detection.

Descriptive bands use the pooled 90th percentile of absolute pre-refit validation residuals. They do not have a calibrated 90% coverage guarantee. Report actual coverage together with width; high coverage alone does not establish useful uncertainty precision.

Exact native Tree SHAP contributions explain the main engineered XGBoost model in `log(1 + remaining minutes)`. Up to 100 evenly spaced eligible examples per primary test bearing produce 242 explained examples. A bias term plus feature contributions reconstructs the log output; contributions do not add in minutes. The implementation checks the maximum additivity discrepancy against a fixed tolerance of `1e-3`.

Global mean absolute attribution gives the short bearing fewer examples than the longer two. Condition-wise rankings refer to this same fitted model. Correlated signal descriptors and linked rpm/load limit mechanical interpretation. Explanations were not used to select features or claim improved human decisions. No LLM is used.

## Maintenance scenarios

The first eligible prediction at or below 30 minutes triggers an assumed replacement alarm. Less than five true remaining minutes, or no alarm, counts as late. Unused-life fraction is remaining recorded time at the alarm divided by recorded endpoint lifetime.

The dimensionless cost is the planned cost 1 for a non-late alarm, or unplanned cost ratio `c` otherwise, plus `w * unused_life_fraction`. Ratios are 2, 5 and 10; weights are 0, 0.5 and 1. Nine fixed scenarios use the three primary test bearings, without test-based threshold optimisation.

Inspect the resulting reliability and unused-life quantities together. A lower regression MAE need not imply a lower scenario cost. Fractions are time proxies, not discarded material mass or lifecycle emissions. Detailed research outcomes are maintained separately from the public code.

## Checkpoint recovery and Drive

Local saves use temporary files and atomic replacement. Forests checkpoint every 25 trees; XGBoost checkpoints every 25 rounds. Completed fits use model checksums. Data/configuration hashes and scientific package versions guard run compatibility. An interrupted Ridge fit restarts that fit.

For a local restart, rerun the exact same training command with the same output directory. If data, settings or versions have changed, restore them or use a new run directory. Only load trusted checkpoint files produced by this project; joblib models use Python serialisation.

The CLI accepts a run directory on an already mounted/synchronised filesystem:

```bash
python run_experiments.py --train --output runs/run_v1 --drive-path /mounted/drive/bearing_runs/run_v1
```

On Windows, replace the example mount path with the actual local Drive location. This option does not authenticate or upload through a Drive API. The parent directory must already exist. Local output and backup must be separate, non-nested directories. If the local run directory is absent and the backup exists, the wrapper restores it before continuing. If both exist, local checkpoints are authoritative; reconcile them deliberately rather than overwriting a newer local run.

In Colab, leave `USE_GOOGLE_DRIVE=True`, mount Drive when prompted and retain `RUN_NAME`. The notebook restores a missing local directory from the same mounted Drive run folder and mirrors completed files every 30 seconds and after scopes. Run the final backup cell before closing and save your notebook copy. A RAR or ZIP uploaded to Drive must be extracted into the expected run directory before it can act as a checkpoint restore source.

Small RF/XGBoost interruption checks are available in the test suite. Actual Colab live mounting/mirroring remains to be verified in a Colab session. Network failure, runtime expiry or interruption before the latest cloud copy can lose progress. The full desktop run used local checkpoints plus separately uploaded immutable snapshots; those external backups are not a public GitHub artifact.

## Results and provenance files

`artifacts/reference_run/` is an optional location for a separately supplied recorded run; it is ignored by Git and is not included in this publication. It is not an output directory for new fitting. The wrapper refuses to write under `artifacts/`.

| Record | Purpose |
|---|---|
| `run_manifest.json` | Config/data hashes, package versions and original runtime platform |
| `completed.json` | Successful experiment completion marker |
| `*_search.json`, `*_selection.json` | All validation candidate scores and selected configurations |
| `*_split.json` | Bearing identities assigned to each stage |
| `*_metrics.csv`, `*_predictions.csv` | Per-bearing metrics and individual held-out predictions |
| `macro_summary.csv` | Four-scope, five-variant macro error summary |
| `shap_*.csv`, `shap_audit.json` | Attributions, ranks and numerical additivity audit |
| `maintenance_*.csv` | Alarm outcomes and all nine assumed scenarios |
| `resume_verification.json`, `run_audit.json` | Recorded recovery and result audits |
| `figures/` | Five result plots |

The read-only result validator recomputes MAE/RMSE and macro summaries, checks bearing boundaries, candidate minima, row uniqueness, finite predictions and saved recovery/SHAP checks. It does not retrain models, certify mechanical correctness or verify omitted model binaries. Repository tests additionally fit tiny synthetic models to check recovery. Data/result tests are skipped when those assets are absent; the GitHub Actions workflow does not claim to audit unpublished artifacts.

The recorded run encountered a pandas index-naming compatibility error when exporting SHAP tables after fitting. It was corrected before completion without changing the scientific protocol; saved models were reused. The published engine contains that correction.

## Troubleshooting

| Symptom | Action |
|---|---|
| Dataset checksum mismatch | Restore the checked-in feature CSV; do not bypass the check for a claimed reproduction |
| Different data/config or dependency versions | Restore the recorded environment or use a new output directory |
| Kernel restarted after dependency installation | Run notebook setup again after the requested restart |
| Drive backup error | Check mount/connection and inspect local checkpoints; do not assume the latest batch reached Drive |
| Notebook missing data | Run the configuration cell with internet access, or set the CSV path to a local verified copy |
| No notebook result plots yet | Train, or restore an extracted completed run; recorded results are supplied separately |
| Unexpectedly weak predictions | Inspect per-bearing and transfer results; do not retune against the published test labels |

## Scope and limitations

The benchmark contains only 15 independent accelerated-test bearings, three linked operating conditions and three primary test bearings. One seed and a fixed small tuning budget are used. The study does not isolate vibration-only versus context-only effects, verify initial health, annotate onset, measure human explanation benefit or validate a prospective maintenance policy. Condition-transfer performance must be assessed from the separately retained outputs; a primary average alone cannot establish transferability.

The code repository supports research reproduction. It is not a deployed asset-monitoring API, a certified fault detector or an industrial maintenance recommendation service. The thesis manuscript and administrative records are separate deliverables.

## References

- Wang, B., Lei, Y., Li, N., & Li, N. (2020). A hybrid prognostics approach for estimating remaining useful life of rolling element bearings. *IEEE Transactions on Reliability, 69*(1), 401–412. https://doi.org/10.1109/TR.2018.2882682
- Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*. https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- Lundberg, S. M., et al. (2019). Explainable AI for trees: From local explanations to global understanding. https://doi.org/10.48550/arXiv.1905.04610
