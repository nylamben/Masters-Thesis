# Explainable AI for Bearing Predictive Maintenance

Reproducible code for the MSc thesis **Explainable AI for Sustainable Predictive Maintenance and Fault Detection in Rolling Bearing**, by Nylam Benkerrache, Near East University, Master of Science in Artificial Intelligence.

This repository compares Ridge, Random Forest and XGBoost on all 15 XJTU-SY bearing trajectories. It includes one interactive notebook, checkpoint recovery, Tree SHAP explanations, maintenance scenarios and validation tools. Data, detailed results and model binaries are supplied separately and are not tracked in this repository. No LLM or API key is required.

**The prediction target is minutes to the final recorded snapshot.** Near-endpoint screening is a proxy; this experiment does not verify fault onset or fault type. Maintenance costs and unused-life fractions are assumed scenarios, not measured financial or environmental savings.

[Full documentation](docs/DOCUMENTATION.md) · [Frozen protocol](docs/experiment_protocol.json)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nylamben/Masters-Thesis/blob/main/notebooks/Bearing_Thesis_Interactive.ipynb)

## Quick start

Use Python 3.12. The recorded environment used Python 3.12.14; scientific package versions are pinned in `requirements.txt`.

```bash
git clone https://github.com/nylamben/Masters-Thesis.git
cd Masters-Thesis
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install dependencies. Once you have supplied the data and an optional recorded run, validate them without training:

```bash
python -m pip install -r requirements.txt
python run_experiments.py --validate-only
python scripts/validate_results.py --run-dir path/to/extracted_run
python -m unittest discover -s tests -v
```

The tests include small synthetic model interruptions and recovery. Data/result-specific checks are skipped when those separately supplied assets are absent. They do not rerun the full benchmark.

Start or resume the full experiment explicitly:

```bash
python run_experiments.py --train --output runs/run_v1
```

Rerun the same command after an interruption. Compatible completed fits are reused, forests resume from saved tree batches and boosting resumes from saved rounds. The wrapper validates data by default; importing the engine does not train a model.

## Supplying data and results

This is a code-and-documentation publication. Obtain the verified `XJTU_SY_features.csv` separately and place it under `data/`, or use the optional raw download and feature-extraction scripts described in the documentation. The entry point checks the recorded SHA256 before fitting.

To audit an existing experiment, supply its extracted run directory separately and pass it to `scripts/validate_results.py --run-dir`. The original dataset provider is credited below. Detailed experiment results and checkpoint files are not published here.

## Repository layout

```text
benchmark_engine.py               Frozen experiment implementation
run_experiments.py                Data validation and explicit training entry point
requirements.txt                  Pinned scientific environment
notebooks/                        One interactive Colab/local notebook
data/                             Locally supplied/generated data, ignored by Git
scripts/download_raw.py           Optional resumable raw benchmark download
scripts/extract_features.py       Optional resumable waveform feature extraction
scripts/validate_results.py       Read-only audit of recorded or new results
artifacts/reference_run/          Optional local results, ignored by Git
docs/DOCUMENTATION.md             Setup, methods, execution, recovery and limitations
docs/experiment_protocol.json     Protocol fixed before the recorded training run
tests/                            Data, artifact, notebook and recovery checks
.github/workflows/validate.yml    Automated validation and small recovery checks
```

Generated `runs/`, virtual environments, raw archives and model binaries are ignored by Git. The raw benchmark is approximately 5.45 GB; training from a separately supplied feature CSV does not require it. Trained checkpoints are retained separately from this repository. Separately supplied prediction/metric records can be audited without loading model files.

## Notebook and Google Drive

Open the Colab link above and run cells in order. Supply the feature CSV using the Colab upload prompt or your mounted Drive folder. The notebook verifies its SHA256 and offers Google Drive mounting. Keep `START_TRAINING=False` until ready; switch it to `True` to train. Save a personal notebook copy if you want to retain settings.

Mounted Drive checkpoints are mirrored every 30 seconds and after experiment scopes. Use the same `RUN_NAME`, mount the same Drive folder and rerun setup to restore a lost runtime. Actual Colab authentication/mounting still needs verification in a Colab session; desktop checkpoint recovery has passed. See [recovery details](docs/DOCUMENTATION.md#checkpoint-recovery-and-drive).

## Data provenance and citation

The [original XJTU-SY provider](https://biaowang.tech/xjtu-sy-bearing-datasets/) supplies the underlying measurements. The experiment expects a derived CSV containing descriptors from 9,216 snapshots, with source paths and per-snapshot hashes. It is not a raw-waveform archive. The extraction script records mirror provenance and checksums in a local `data/dataset_manifest.json`. The expected feature checksum is also recorded in the code and documentation.

Wang, B., Lei, Y., Li, N., & Li, N. (2020). A hybrid prognostics approach for estimating remaining useful life of rolling element bearings. *IEEE Transactions on Reliability, 69*(1), 401–412. https://doi.org/10.1109/TR.2018.2882682

Use the provider's terms for the underlying benchmark; this repository does not grant additional rights over third-party data or papers. `CITATION.cff` identifies this research software. Copyrighted research PDFs, personal administrative files and the thesis manuscript are not part of the code publication.
