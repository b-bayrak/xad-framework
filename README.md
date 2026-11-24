# XAD: What, How Confident, and Why

XAD is a modular framework for explainable anomaly detection in multivariate time series. It answers three questions for every window:

| Question | Component | Module |
|---|---|---|
| **What** is anomalous? | Autoencoder detectors: Conv-AE, LSTM-AE, LSTM-VAE | `xad/detectors.py` |
| **How confident** is the decision? | Conformal threshold with three policies (static, sliding window, Adaptive Conformal Inference), conformal p-value and confidence, four uncertainty factors | `xad/temporal_conformal_predictor.py`, `xad/uncertainty_explainer.py` |
| **Why** was it flagged? | Counterfactual explanations: GA-CE (proposed), DICE, PerCE | `xad/counterfactual_explainer.py`, `xad/dice_explainer.py`, `xad/perce_explainer.py` |

This repository accompanies the paper *What, How Confident, and Why: A Unified Framework for Explainable Anomaly Detection through Uncertainty Decomposition and Counterfactual Explanations* (B. Bayrak and K. Bach, submitted to Knowledge-Based Systems).

## Installation

```bash
git clone https://github.com/b-bayrak/xad-framework.git
cd xad-framework
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121   # or the CPU build
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
from xad.xad import XAD
from xad.data_loaders import DataLoaderFactory, BaseDataLoader

# Chronological split; windows never cross a split boundary
loader = DataLoaderFactory.create('skab', data_path='./data/SKAB/', sequence_length=50, window_overlap=0.8)
X_train, X_val, X_test, y_train, y_val, y_test = loader.load_and_split(test_size=0.2, val_size=0.2)

# The detector is trained and the threshold calibrated on normal windows only
X_train_normal, _ = BaseDataLoader.normal_only(X_train, y_train)
X_val_normal, _ = BaseDataLoader.normal_only(X_val, y_val)

xad = XAD(input_dim=X_train.shape[-1], sequence_length=X_train.shape[1],
          detector_type='lstm_vae', alpha=0.05)
xad.train(X_train_normal, X_val_normal, epochs=200, patience=20)

result = xad.detect_and_explain(X_test[0], generate_cf=True)
print(result['prediction'], result['confidence'], result['anomaly_score'])
```

SKAB is available with `git clone https://github.com/waico/SKAB.git data/SKAB`.

## Components

**Detectors.** Three reconstruction-based autoencoders. The nonconformity score is the reconstruction error of a window, optionally with a temporal term.

**Conformal threshold.** The threshold is the conformal quantile of the scores of normal calibration windows at significance level α. It is updated online by one of three policies:

* `static`: calibrated once;
* `sliding`: recalibrated on the most recent windows that were accepted as normal;
* `aci`: Adaptive Conformal Inference, which adjusts α after every decision.

Each decision comes with a conformal p-value and a confidence value.

**Uncertainty decomposition.** Four factors describe why a decision may be unreliable: distributional (distance to the training windows), density (local outlier factor), temporal (variability of the window) and feature (which sensors drive the score).

**Counterfactual explanations.** GA-CE searches with a genetic algorithm for a nearby window that the same conformal threshold would accept, balancing validity, proximity, sparsity and temporal smoothness. DICE and PerCE are adapted to the same score and threshold. GA-CE and PerCE only need the score, so they also work with non-differentiable detectors.

The threshold policies and the score-based explainers accept any scalar anomaly score, so the detector can be replaced, for example by Isolation Forest.

## Repository layout

```
xad/            framework
experiments/    experiments 1-8 and the shared training and calibration pipeline
datasets/       download and preparation of the ten additional datasets
jobs/           SLURM jobs for all runs
analysis/       statistical tests and the scripts that build the tables and figures
results/        tables, figures and summaries reported in the paper
docs/           reproduction guide, analysis plan, audit and run log
```

## Experiments

Each experiment is a standalone script. For example, on SKAB with one detector and seed:

```bash
python experiments/run_all_experiments.py --dataset skab --data_path ./data/SKAB/ \
    --detector lstm_vae --seed 42 --experiments 1 2 3 4 6
```

| Script | Experiment |
|---|---|
| `exp_1_detection_performance.py` | Detection performance (ROC-AUC, PR-AUC) |
| `exp_2_calibration_quality.py` | Coverage and calibration error across α |
| `exp_3_uncertainty_effectiveness.py` | Reliability by confidence level |
| `exp_4_counterfactual_quality.py` | Validity, proximity and sparsity of GA-CE, DICE and PerCE |
| `exp_5_gace_ablation.py` | GA-CE loss-term ablation |
| `exp_6_cp_ablation.py` | Threshold policies |
| `exp_7_policy_sensitivity.py` | Sensitivity to the ACI step size and the window size |
| `exp_8_classical_baselines.py` | Isolation Forest, LOF and One-Class SVM |

All scripts share one trained model per dataset, detector and seed through `experiments/pipeline.py`, so their results are consistent with each other.

## Reproducing the paper

The paper evaluates twelve datasets (SKAB, SWaT, 3W, Daphnet, Exathlon, GECCO, GHL, MetroPT-3, MSL, PSM, SMAP, SMD) with three detectors and five seeds. The datasets are not redistributed. `datasets/download.sh` and `datasets/prepare.py` obtain and prepare the ten additional ones, and SWaT is available on request from [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/).

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) lists the full sequence of jobs. After the runs,

```bash
bash analysis/run.sh paper_outputs.py
```

rebuilds every table, figure and number in `results/paper/`. The outputs of every run are attached to the GitHub release.

## Citation

```bibtex
@article{bayrak2026xad,
  title   = {What, How Confident, and Why: A Unified Framework for Explainable Anomaly Detection
             through Uncertainty Decomposition and Counterfactual Explanations},
  author  = {Bayrak, Bet{\"u}l and Bach, Kerstin},
  note    = {Submitted to Knowledge-Based Systems},
  year    = {2026}
}
```

## License

[CC BY-NC-SA 4.0](LICENSE). The datasets keep their own licenses.
