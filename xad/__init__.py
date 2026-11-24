from .detectors import (
    BaseDetector,
    ConvAE,
    LSTM_AE,
    LSTM_VAE,
    create_detector,
    DetectorTrainer
)

from .temporal_conformal_predictor import TemporalConformalPredictor
from .counterfactual_explainer import CounterfactualExplainer
from .uncertainty_explainer import UncertaintyExplainer, UncertaintyExplanation, UncertaintyVisualizer
from .xad import XAD, DETECTOR_TYPES
from .data_loaders import (
    DataLoaderFactory,
    SKABDataLoader,
    SWaTDataLoader,
    CustomDataLoader,
    quick_load
)

# Available detectors. Performance numbers are deliberately not hardcoded here -
# they depend on dataset, seed, and hyperparameters and must come from an actual
# run of experiments/run_all_experiments.py, not from a stale constant in source.
DETECTOR_INFO = {
    'conv_ae': {'name': 'Conv-AE', 'description': 'Convolutional autoencoder - local temporal patterns'},
    'lstm_ae': {'name': 'LSTM-AE', 'description': 'LSTM autoencoder - long-range temporal dependencies'},
    'lstm_vae': {'name': 'LSTM-VAE', 'description': 'LSTM variational autoencoder - probabilistic latent space'}
}

__all__ = [
    # Detectors
    'BaseDetector',
    'ConvAE',
    'LSTM_AE', 
    'LSTM_VAE',
    'create_detector',
    'DetectorTrainer',
    'DETECTOR_INFO',
    
    # Core components
    'TemporalConformalPredictor',
    'CounterfactualExplainer',
    'UncertaintyExplainer',
    'UncertaintyExplanation',
    'UncertaintyVisualizer',
    'XAD',
    'DETECTOR_TYPES',
    
    # Data loading
    'DataLoaderFactory',
    'SKABDataLoader',
    'SWaTDataLoader',
    'CustomDataLoader',
    'quick_load'
]
