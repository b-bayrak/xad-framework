#!/usr/bin/env python
"""
Shared training/calibration pipeline used by every exp_*.py script.

Every exp_*.py script calls `get_or_train_detector()`, which trains once per
(dataset, detector, seed) and caches the weights to disk, so all experiments
for that combination are computed from the same trained model.

The module also enforces two data-hygiene rules:
  1. Anomalies are excluded from detector TRAINING (and from the data used
     for early-stopping) - only the normal subset of the training split is
     used to fit the autoencoder. Validation and test splits retain the
     dataset's natural anomaly rate, as they must to be useful.
  2. Conformal CALIBRATION also uses only normal instances (the normal
     subset of the validation split) - the coverage guarantee assumes
     exchangeability with the population it's meant to protect (normal
     instances), and calibrating on a contaminated set inflates the
     threshold and silently weakens the guarantee.

It also separates two metrics: overall classification ACCURACY (both
classes) and COVERAGE (the fraction of true-normal test instances correctly retained as
normal - the one-sided quantity conformal prediction actually bounds).
"""

import os
import json
import numpy as np
import torch

from xad.detectors import create_detector, DetectorTrainer
from xad.temporal_conformal_predictor import TemporalConformalPredictor
from xad.data_loaders import DataLoaderFactory


# Dataset-specific settings (see the paper's appendix on hyperparameters).
DATASET_CONFIG = {
    # smoothness_weight (lambda_temp, Eq. 9): selected on the VALIDATION
    # split's labels (used for nothing else). On SKAB the selection is
    # unambiguous - val ROC-AUC decreases monotonically in lambda
    # (lstm_ae: 0.622 at 0.0 -> 0.550 at 0.1; lstm_vae: 0.597 -> 0.547),
    # because the TVD term's raw magnitude is 15-150x the base
    # reconstruction score and swamps it. SWaT's validation split contains
    # no anomalies (all attacks are in the chronological tail), so no
    # per-dataset selection is possible there; the SKAB-selected value is
    # transferred. exp_6 sweeps {0, 0.01, 0.1} on both datasets so the
    # published ablation still reports the TVD term's effect explicitly.
    'skab': {
        'sequence_length': 50,
        'window_overlap': 0.8,
        'calibration_window': 500,
        'smoothness_weight': 0.0,
    },
    'swat': {
        'sequence_length': 100,
        'window_overlap': 0.5,
        'calibration_window': 1000,
        'smoothness_weight': 0.0,
    },
}

DETECTORS = ['conv_ae', 'lstm_ae', 'lstm_vae']


def load_dataset(dataset: str, data_path: str, test_size: float = 0.2, val_size: float = 0.2):
    """
    Chronological load+split (see data_loaders.BaseDataLoader.load_and_split).
    Returns X_train, X_val, X_test, y_train, y_val, y_test - all in
    chronological order within each split, with the dataset's native
    anomaly rate preserved in val/test.
    """
    cfg = DATASET_CONFIG[dataset]
    loader = DataLoaderFactory.create(
        dataset,
        data_path=data_path,
        sequence_length=cfg['sequence_length'],
        window_overlap=cfg['window_overlap'],
    )
    return loader.load_and_split(test_size=test_size, val_size=val_size, temporal_split=True)


def _cuda_usable() -> bool:
    """
    True only if CUDA is available AND actually runs a kernel. On nodes whose
    GPU is older than the installed torch wheel supports (e.g. a P100/sm_60
    with a cu12 wheel built for sm_70+), torch.cuda.is_available() returns
    True but every kernel launch raises - probe once instead of crashing
    mid-experiment.
    """
    if not torch.cuda.is_available():
        return False
    try:
        (torch.zeros(1, device='cuda') + 1).item()
        return True
    except Exception:
        return False


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_or_train_detector(
    detector_type: str,
    dataset: str,
    X_train_normal: np.ndarray,
    X_val_normal: np.ndarray,
    input_dim: int,
    sequence_length: int,
    config: dict,
    cache_dir: str = './results/cache',
    force_retrain: bool = False,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Train (or load a cached checkpoint of) a detector for (dataset, detector_type).
    Trained ONLY on X_train_normal; X_val_normal is used purely for early
    stopping. Both must already be filtered to y==0 by the caller
    (see `BaseDataLoader.normal_only`).
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Cache is keyed by seed so multi-seed runs (for mean +/- std tables)
    # never overwrite each other. Legacy un-tagged checkpoints (all trained
    # with the default seed=42) are still honored for seed 42 only.
    ckpt_path = os.path.join(cache_dir, f'{dataset}_{detector_type}_seed{seed}.pt')
    legacy_path = os.path.join(cache_dir, f'{dataset}_{detector_type}.pt')
    if seed == 42 and not os.path.exists(ckpt_path) and os.path.exists(legacy_path):
        ckpt_path = legacy_path

    # Forward every architecture kwarg present in the config (e.g. the Conv-AE
    # filters [64, 32] and kernel size 7). create_detector() drops kwargs a
    # given architecture does not accept.
    arch_kwargs = {k: config[k] for k in
                   ('hidden_dim', 'latent_dim', 'num_layers', 'dropout',
                    'filters', 'kernel_size', 'hidden_channels',
                    'bidirectional', 'alpha_score') if k in config}
    detector = create_detector(
        detector_type=detector_type,
        input_dim=input_dim,
        sequence_length=sequence_length,
        **arch_kwargs,
    )

    if os.path.exists(ckpt_path) and not force_retrain:
        state = torch.load(ckpt_path, map_location='cpu')
        detector.load_state_dict(state['model_state_dict'])
        # Evaluate on the same device a freshly-trained model would be on
        # (DetectorTrainer moves it to CUDA when available). Scoring the same
        # weights on CPU vs GPU gives float-level score differences, which
        # the adaptive sliding-window threshold then amplifies through its
        # feedback loop into visibly different coverage numbers (observed:
        # SWaT lstm_vae 0.7854 vs 0.7896 between exp_1 [fresh, GPU] and
        # exp_2 [cached, CPU] on identical weights).
        if _cuda_usable():
            detector = detector.cuda()
        if verbose:
            print(f"Loaded cached {detector_type}/{dataset} from {ckpt_path}")
        return detector, state.get('history', {})

    _set_seed(seed)
    trainer = DetectorTrainer(
        detector=detector,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        kl_weight=config['kl_weight'],
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train_normal).float()),
        batch_size=config['batch_size'], shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_val_normal).float()),
        batch_size=config['batch_size'], shuffle=False,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        patience=config['patience'],
        verbose=verbose,
    )

    torch.save({'model_state_dict': detector.state_dict(), 'history': history}, ckpt_path)
    if verbose:
        print(f"Trained and cached {detector_type}/{dataset} -> {ckpt_path}")

    return detector, history


def make_conformal_predictor(
    detector,
    X_val_normal: np.ndarray,
    dataset: str,
    alpha: float = 0.05,
    adaptive: bool = True,
    smoothness_weight: float = None,
    cp_mode: str = None,
    aci_gamma: float = 0.005,
    verbose: bool = False,
) -> TemporalConformalPredictor:
    """
    Calibrate a TemporalConformalPredictor on the normal subset of validation.

    `cp_mode` in {'static', 'sliding', 'aci'} selects the threshold policy
    (see TemporalConformalPredictor docstring); when None, the legacy
    `adaptive` flag decides between 'sliding' and 'static'.

    `smoothness_weight=None` uses the per-dataset default from
    DATASET_CONFIG (selected on the validation split's labels, which are
    used for nothing else - see 'smoothness_weight' comments there).
    """
    window_size = DATASET_CONFIG[dataset]['calibration_window']
    if smoothness_weight is None:
        smoothness_weight = DATASET_CONFIG[dataset]['smoothness_weight']
    cp = TemporalConformalPredictor(
        alpha=alpha, window_size=window_size,
        smoothness_weight=smoothness_weight, adaptive=adaptive,
        mode=cp_mode, aci_gamma=aci_gamma,
    )
    cp.calibrate(X_val_normal, detector, verbose=verbose)
    return cp


def predict_all(cp: TemporalConformalPredictor, detector, X_test: np.ndarray):
    """
    Run predict_with_confidence over X_test IN ORDER (required for the
    sliding-window recalibration to be meaningful) and collect predictions,
    scores, and confidences.
    """
    predictions, scores, confidences = [], [], []
    detector.eval()
    for i in range(len(X_test)):
        result = cp.predict_with_confidence(X_test[i], detector)
        predictions.append(1 if result['prediction'] == 'anomaly' else 0)
        scores.append(result['score'])
        confidences.append(result['confidence'])
    return np.array(predictions), np.array(scores), np.array(confidences)


def coverage_and_accuracy(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> dict:
    """
    Two DISTINCT metrics that must not be reported under the same name:

    - accuracy: overall classification accuracy over BOTH classes. Depends on
      detection power on true anomalies, which conformal prediction makes NO
      guarantee about.
    - coverage: fraction of TRUE-NORMAL test instances correctly retained as
      normal (specificity). This is the one-sided quantity the guarantee
      P(y_hat=normal | y=normal) >= 1-alpha actually bounds; it says nothing
      about anomalies. calibration_error is computed against this, not
      against accuracy.
    """
    accuracy = float(np.mean(y_pred == y_true))

    normal_mask = (y_true == 0)
    if normal_mask.sum() > 0:
        coverage = float(np.mean(y_pred[normal_mask] == 0))
    else:
        coverage = float('nan')

    expected_coverage = 1 - alpha
    calibration_error = abs(coverage - expected_coverage) if normal_mask.sum() > 0 else float('nan')

    return {
        'accuracy': accuracy,
        'coverage': coverage,
        'expected_coverage': expected_coverage,
        'calibration_error': calibration_error,
        'n_test': len(y_true),
        'n_true_normal': int(normal_mask.sum()),
        'n_true_anomaly': int((~normal_mask).sum()),
    }


def default_config(args) -> dict:
    return {
        'hidden_dim': args.hidden_dim,
        'latent_dim': args.latent_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'kl_weight': args.kl_weight,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        # Conv-AE only (ignored by LSTM-AE and LSTM-VAE): filters [64, 32],
        # kernel size 7.
        'filters': [64, 32],
        'kernel_size': 7,
    }


def prepare_split(dataset: str, data_path: str, test_size: float, val_size: float, verbose: bool = True):
    """
    Load + chronologically split, then derive the normal-only train/val
    subsets used for training and calibration. Returns a dict with every
    array a downstream experiment script needs.
    """
    from xad.data_loaders import BaseDataLoader

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(dataset, data_path, test_size, val_size)
    X_train_normal, y_train_normal = BaseDataLoader.normal_only(X_train, y_train)
    X_val_normal, y_val_normal = BaseDataLoader.normal_only(X_val, y_val)

    if verbose:
        print(f"  Train: {X_train.shape} (anomaly rate {y_train.mean():.2%}) "
              f"-> normal-only for training: {X_train_normal.shape}")
        print(f"  Val:   {X_val.shape} (anomaly rate {y_val.mean():.2%}) "
              f"-> normal-only for calibration: {X_val_normal.shape}")
        print(f"  Test:  {X_test.shape} (anomaly rate {y_test.mean():.2%}, kept as-is)")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'X_train_normal': X_train_normal, 'y_train_normal': y_train_normal,
        'X_val_normal': X_val_normal, 'y_val_normal': y_val_normal,
        'input_dim': X_train.shape[2],
        'sequence_length': X_train.shape[1],
    }
