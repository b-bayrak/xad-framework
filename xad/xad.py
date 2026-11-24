"""
This module implements the eXplainable Anomaly Detection framework,
bring the components together including multiple detector architectures,
however, this is not used in the experiments.
Can be used as a unified interface for anomaly detection with explanations.

1. Multiple detector architectures (Conv-AE, LSTM-AE, LSTM-VAE)
2. Temporal Conformal Prediction for uncertainty quantification
3. Counterfactual Explanations for interpretability
4. Uncertainty Decomposition (epistemic, aleatoric, temporal)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime
import json
import warnings

# Import all detector types
from .detectors import ConvAE, LSTM_AE, LSTM_VAE
from .temporal_conformal_predictor import TemporalConformalPredictor
from .counterfactual_explainer import CounterfactualExplainer
from .uncertainty_explainer import UncertaintyExplainer


DETECTOR_TYPES = {
    'conv_ae': 'Conv-AE',
    'lstm_ae': 'LSTM-AE',
    'lstm_vae': 'LSTM-VAE',
}


class XAD:
    """
    Explainable Anomaly Detection Framework.

    Args:
        input_dim: Number of input features
        sequence_length: Length of input sequences
        detector_type: Type of detector ('conv_ae', 'lstm_ae', 'lstm_vae')
        hidden_dim: Hidden dimension (interpretation varies by detector)
        latent_dim: Latent space dimension
        dropout: Dropout rate
        num_layers: Number of layers (for LSTM-based detectors)
        alpha: Significance level for conformal prediction
        device: Device to use ('cuda', 'cpu', or 'auto')
        **detector_kwargs: Additional arguments for specific detector
    
    Attributes:
        detector: Anomaly detection model
        conformal: Temporal conformal predictor
        explainer: Counterfactual explainer
        device: Computing device
        is_trained: Whether the model has been trained
        is_calibrated: Whether conformal predictor is calibrated
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        detector_type: str = 'conv_ae',
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.2,
        num_layers: int = 2,
        alpha: float = 0.1,
        device: str = 'auto',
        **detector_kwargs
    ):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.detector_type = detector_type.lower()
        self.alpha = alpha
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._init_detector(
            detector_type=self.detector_type,
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            num_layers=num_layers,
            **detector_kwargs
        )
        
        self.conformal = TemporalConformalPredictor(alpha=alpha)
        
        self.explainer: Optional[CounterfactualExplainer] = None
        self.uncertainty_explainer: Optional[UncertaintyExplainer] = None
        
        self.is_trained = False
        self.is_calibrated = False
        self.training_history: List[Dict] = []
        
        self.hyperparameters = {
            'input_dim': input_dim,
            'sequence_length': sequence_length,
            'detector_type': self.detector_type,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'dropout': dropout,
            'num_layers': num_layers,
            'alpha': alpha
        }
    
    def _init_detector(
        self,
        detector_type: str,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        num_layers: int,
        **kwargs
    ):

        detector_type = detector_type.lower().replace('-', '_')
        
        if detector_type == 'conv_ae':
            hidden_channels = kwargs.get('hidden_channels', [32, 64, 128])
            self.detector = ConvAE(
                input_dim=input_dim,
                sequence_length=sequence_length,
                latent_dim=latent_dim,
                hidden_channels=hidden_channels
            )
        
        elif detector_type == 'lstm_ae':
            self.detector = LSTM_AE(
                input_dim=input_dim,
                sequence_length=sequence_length,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=kwargs.get('bidirectional', True)
            )
        
        elif detector_type == 'lstm_vae':
            self.detector = LSTM_VAE(
                input_dim=input_dim,
                sequence_length=sequence_length,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_layers=num_layers,
                dropout=dropout
            )


        else:
            raise ValueError(
                f"Unknown detector type: {detector_type}. "
                f"Available: {list(DETECTOR_TYPES.keys())}"
            )
        
        self.detector.to(self.device)
        print(f"Initialized {self.detector.name if hasattr(self.detector, 'name') else detector_type} detector")

    def train(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        X_val: Union[np.ndarray, torch.Tensor],
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 15,
        kl_weight: float = 0.001,
        weight_decay: float = 1e-5,
        optimizer: str = 'adamw',
        model_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(X_val, np.ndarray):
            X_val = torch.from_numpy(X_val).float()
        
        # Load pre-trained model if provided
        if model_path is not None and os.path.exists(model_path):
            self.load_model(model_path)
            if verbose:
                print(f"Loaded pre-trained model from {model_path}")
        else:
            train_dataset = TensorDataset(X_train)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                drop_last=True if len(X_train) > batch_size else False
            )
            
            if optimizer.lower() == 'adamw':
                opt = optim.AdamW(
                    self.detector.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay
                )
            else:
                opt = optim.Adam(
                    self.detector.parameters(), 
                    lr=lr, 
                    weight_decay=weight_decay
                )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=5, verbose=verbose
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            history = {'train_loss': [], 'val_loss': [], 'lr': []}
            
            if verbose:
                detector_name = self.detector.name if hasattr(self.detector, 'name') else self.detector_type
                print(f"\n{'-'*60}")
                print(f"Training {detector_name}")
                print(f"{'-'*60}")
                print(f"  Device: {self.device}")
                print(f"  Training samples: {len(X_train)}")
                print(f"  Validation samples: {len(X_val)}")
                print(f"  Batch size: {batch_size}")
                print(f"  Learning rate: {lr}")
                print(f"{'-'*60}\n")
            
            for epoch in range(epochs):
                # Training phase
                self.detector.train()
                train_losses = []
                
                for batch in train_loader:
                    x = batch[0].to(self.device)
                    opt.zero_grad()
                    
                    # Forward pass
                    outputs = self.detector(x)
                    
                    loss = self._compute_loss(x, outputs, kl_weight, epoch)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.detector.parameters(), max_norm=1.0
                    )
                    
                    opt.step()
                    train_losses.append(loss.item())
                
                self.detector.eval()
                val_losses = []
                
                with torch.no_grad():
                    for i in range(0, len(X_val), batch_size):
                        x = X_val[i:i+batch_size].to(self.device)
                        outputs = self.detector(x)
                        loss = self._compute_loss(x, outputs, kl_weight, epoch)
                        val_losses.append(loss.item())
                
                avg_train_loss = np.mean(train_losses)
                avg_val_loss = np.mean(val_losses)
                current_lr = opt.param_groups[0]['lr']
                
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['lr'].append(current_lr)
                
                # LR scheduling
                scheduler.step(avg_val_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1:3d}/{epochs} | "
                          f"Train Loss: {avg_train_loss:.6f} | "
                          f"Val Loss: {avg_val_loss:.6f} | "
                          f"LR: {current_lr:.2e}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self._best_model_state = self.detector.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
            
            # Restore best model
            if hasattr(self, '_best_model_state'):
                self.detector.load_state_dict(self._best_model_state)
            
            self.training_history = history
        
        self.is_trained = True
        
        if verbose:
            print("\nCalibrating conformal predictor...")
        self.conformal.calibrate(X_val, self.detector, verbose=verbose)
        self.is_calibrated = True
        
        if verbose:
            print("Initializing counterfactual explainer...")
        self._fit_explainer(X_val)
        
        if verbose:
            print("\nTraining completed!")
        
        return self.training_history
    
    def _compute_loss(
        self, 
        x: torch.Tensor, 
        outputs: Tuple, 
        kl_weight: float,
        epoch: int = 0
    ) -> torch.Tensor:

        if self.detector_type == 'lstm_vae':
            if len(outputs) == 3:
                reconstruction, mu, logvar = outputs
                
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstruction, x)
                
                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                # KL weight
                kl_weight_a = min(kl_weight, kl_weight * (epoch + 1) / 10)

                loss = recon_loss + kl_weight_a * kl_loss
            else:
                # Fallback for non-VAE
                reconstruction = outputs[0]
                loss = F.mse_loss(reconstruction, x)
        else:
            # Non-VAE detectors: (reconstruction, latent)
            reconstruction = outputs[0]
            loss = F.mse_loss(reconstruction, x)
        
        # Add temporal smoothness regularization
        if reconstruction.size(1) > 1:
            temporal_diff = reconstruction[:, 1:] - reconstruction[:, :-1]
            smoothness_loss = 0.01 * torch.mean(temporal_diff ** 2)
            loss = loss + smoothness_loss
        
        return loss

    def _fit_explainer(self, X_val: torch.Tensor) -> None:
        normal_indices = []
        
        for i in range(len(X_val)):
            # update_state=False: X_val was already consumed by calibrate();
            # re-scoring it here to find normal indices must not double-count
            # these points into the adaptive rolling window a second time.
            result = self.conformal.predict_with_confidence(X_val[i], self.detector, update_state=False)
            if result['prediction'] == 'normal':
                normal_indices.append(i)
        
        if normal_indices:
            X_normal = X_val[normal_indices]
            
            # Initialize counterfactual explainer
            self.explainer = CounterfactualExplainer(
                model=self.detector,
                conformal_predictor=self.conformal
            )
            self.explainer.fit(X_normal)
            
            self.uncertainty_explainer = UncertaintyExplainer(
                training_data=X_normal,
                n_neighbors=min(20, len(X_normal) - 1)
            )
        else:
            warnings.warn("No normal instances found in validation set. Counterfactual and uncertainty explainers not initialized.")

    def detect_and_explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        generate_cf: bool = False,
        explain_uncertainty: bool = True,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Returns:
            Dictionary containing:
                - prediction: 'normal' or 'anomaly'
                - confidence: Prediction confidence
                - anomaly_score: Reconstruction-based score
                - threshold: Decision threshold
                - uncertainty: Prediction uncertainty
                - feature_importance: Per-feature error contribution
                - uncertainty_explanation: Detailed uncertainty breakdown
                - counterfactual: (optional) Counterfactual explanation
        """
        if not self.is_trained or not self.is_calibrated:
            raise RuntimeError("Model not trained or calibrated. Call train() first.")
        
        self.detector.eval()
        
        if isinstance(x, torch.Tensor):
            x_numpy = x.detach().cpu().numpy()
        else:
            x_numpy = np.array(x)
        
        result = self.conformal.predict_with_confidence(x_numpy, self.detector)
        
        with torch.no_grad():
            anomaly_score, score_details = self.detector.compute_anomaly_score(x_numpy)
        
        # Extract feature importance from score_details dict
        if isinstance(score_details, dict):
            feature_importance = score_details.get('feature_importance', None)
            if feature_importance is None:
                reconstruction = score_details.get('reconstruction', None)
                if reconstruction is not None:
                    if isinstance(x_numpy, torch.Tensor):
                        x_np = x_numpy.detach().cpu().numpy()
                    else:
                        x_np = x_numpy
                    if x_np.ndim == 2:
                        x_np = x_np[np.newaxis, :]
                    if reconstruction.ndim == 2:
                        reconstruction = reconstruction[np.newaxis, :]
                    feature_errors = (reconstruction - x_np) ** 2
                    feature_importance = np.mean(feature_errors, axis=(0, 1))
                else:
                    feature_importance = np.zeros(self.input_dim)
            elif isinstance(feature_importance, torch.Tensor):
                feature_importance = feature_importance.detach().cpu().numpy()
            else:
                feature_importance = np.array(feature_importance)
        else:
            feature_errors = score_details
            if isinstance(feature_errors, torch.Tensor):
                feature_errors = feature_errors.detach().cpu().numpy()
            if feature_errors.ndim == 3:
                feature_importance = np.mean(feature_errors, axis=(0, 1))
            else:
                feature_importance = np.mean(feature_errors, axis=0).squeeze()
        
        # Ensure feature_importance is 1D
        feature_importance = np.atleast_1d(feature_importance).flatten()
        
        # Rank features by importance
        top_features = np.argsort(feature_importance)[::-1][:5]
        
        explanation = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'anomaly_score': result['score'],
            'threshold': result['threshold'],
            'uncertainty': result['uncertainty'],
            'p_value': result['p_value'],
            'feature_importance': feature_importance.tolist(),
            'top_features': top_features.tolist(),
            'top_feature_names': [
                feature_names[i] if feature_names else f"Feature_{i}" 
                for i in top_features
            ]
        }
        
        if explain_uncertainty and self.uncertainty_explainer is not None:
            # Get reconstruction errors for uncertainty analysis
            if isinstance(score_details, dict):
                reconstruction = score_details.get('reconstruction', None)
                if reconstruction is not None:
                    if isinstance(x_numpy, torch.Tensor):
                        x_np = x_numpy.detach().cpu().numpy()
                    else:
                        x_np = x_numpy
                    if x_np.ndim == 2:
                        x_np = x_np[np.newaxis, :]
                    if reconstruction.ndim == 2:
                        reconstruction = reconstruction[np.newaxis, :]
                    reconstruction_errors = (reconstruction - x_np) ** 2
                else:
                    reconstruction_errors = np.zeros((1, self.sequence_length, self.input_dim))
            else:
                reconstruction_errors = score_details if not isinstance(score_details, dict) else np.zeros((1, self.sequence_length, self.input_dim))
            
            unc_explanation = self.uncertainty_explainer.explain_uncertainty(
                x=x_numpy,
                anomaly_score=result['score'],
                confidence=result['confidence'],
                uncertainty=result['uncertainty'],
                reconstruction_errors=reconstruction_errors
            )
            
            # Add to explanation dict
            explanation['uncertainty_breakdown'] = {
                'total': unc_explanation.total_uncertainty,
                'epistemic': unc_explanation.epistemic_uncertainty,
                'aleatoric': unc_explanation.aleatoric_uncertainty,
                'temporal': unc_explanation.temporal_uncertainty
            }
            explanation['uncertainty_sources'] = unc_explanation.uncertainty_sources
            explanation['uncertainty_explanation'] = unc_explanation.summary
            explanation['uncertainty_reasons'] = unc_explanation.detailed_reasons
            explanation['uncertainty_recommendations'] = unc_explanation.recommendations
            explanation['prediction_reliability'] = unc_explanation.prediction_reliability
            explanation['should_flag_for_review'] = unc_explanation.should_flag_for_review
            explanation['feature_uncertainties'] = unc_explanation.feature_uncertainties.tolist()
            explanation['top_uncertain_features'] = unc_explanation.top_uncertain_features
        else:
            # Fallback to simple explanation
            if result['uncertainty'] > self.alpha:
                explanation['uncertainty_explanation'] = (
                    'High uncertainty - prediction may be unreliable'
                )
                explanation['uncertainty_reasons'] = [
                    'Data point is far from calibration distribution'
                ]
            else:
                explanation['uncertainty_explanation'] = (
                    'Low uncertainty - prediction is reliable'
                )
                explanation['uncertainty_reasons'] = [
                    'Data point is similar to calibration data'
                ]
        
        # Generate counterfactual if requested and sample is anomaly
        if generate_cf and result['prediction'] == 'anomaly' and self.explainer:
            try:
                cf_explanation = self.explainer.generate_counterfactual(x_numpy)
                explanation['counterfactual'] = cf_explanation
            except Exception as e:
                explanation['counterfactual_error'] = str(e)
        
        return explanation

    def evaluate(
        self,
        X_test: Union[np.ndarray, torch.Tensor],
        y_test: Optional[np.ndarray] = None,
        generate_cf: bool = False,
        save_results: bool = True,
        output_dir: str = './results'
    ) -> Dict[str, float]:

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Convert to tensor if needed
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        
        results = {
            'predictions': [],
            'confidences': [],
            'scores': [],
            'uncertainties': [],
            'p_values': [],
            'epistemic_uncertainties': [],
            'aleatoric_uncertainties': [],
            'temporal_uncertainties': [],
            'reliability_ratings': [],
            'flagged_for_review': []
        }
        
        has_labels = y_test is not None
        if has_labels:
            results['coverage'] = []
        
        results_list = []
        
        print(f"\nEvaluating on {len(X_test)} samples...")
        
        for idx in range(len(X_test)):
            if (idx + 1) % 100 == 0:
                print(f"  Processing {idx + 1}/{len(X_test)}...")
            
            x = X_test[idx]
            
            # Get explanation with uncertainty breakdown
            explanation = self.detect_and_explain(
                x, 
                generate_cf=generate_cf and (y_test is None or y_test[idx] == 1),
                explain_uncertainty=True
            )
            
            # Store results
            is_anomaly = explanation['prediction'] == 'anomaly'
            results['predictions'].append(is_anomaly)
            results['confidences'].append(explanation['confidence'])
            results['scores'].append(explanation['anomaly_score'])
            results['uncertainties'].append(explanation['uncertainty'])
            results['p_values'].append(explanation['p_value'])
            
            # Store uncertainty breakdown if available
            if 'uncertainty_breakdown' in explanation:
                results['epistemic_uncertainties'].append(
                    explanation['uncertainty_breakdown']['epistemic']
                )
                results['aleatoric_uncertainties'].append(
                    explanation['uncertainty_breakdown']['aleatoric']
                )
                results['temporal_uncertainties'].append(
                    explanation['uncertainty_breakdown']['temporal']
                )
                results['reliability_ratings'].append(
                    explanation.get('prediction_reliability', 'unknown')
                )
                results['flagged_for_review'].append(
                    explanation.get('should_flag_for_review', False)
                )
            
            # Check coverage if labels available
            if has_labels:
                is_correct = is_anomaly == bool(y_test[idx])
                results['coverage'].append(is_correct)
            
            # Evaluate counterfactual quality
            cf_valid = None
            cf_proximity = None
            
            if 'counterfactual' in explanation:
                if 'cf_quality' not in results:
                    results['cf_quality'] = []
                
                cf = explanation['counterfactual']['counterfactual']
                # update_state=False: cf is a synthetic counterfactual, not a
                # real chronological observation - must not enter the window.
                cf_result = self.conformal.predict_with_confidence(cf, self.detector, update_state=False)
                cf_valid = cf_result['prediction'] == 'normal'
                
                x_numpy = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                if x_numpy.ndim == 3:
                    x_numpy = x_numpy.squeeze(0)
                cf_proximity = float(np.mean((cf - x_numpy) ** 2))
                
                results['cf_quality'].append({
                    'valid': cf_valid,
                    'proximity': cf_proximity
                })
            
            # Store detailed results
            result_entry = {
                'sample_id': idx,
                'prediction': explanation['prediction'],
                'confidence': explanation['confidence'],
                'true_label': int(y_test[idx]) if has_labels else None,
                'anomaly_score': explanation['anomaly_score'],
                'uncertainty': explanation['uncertainty'],
                'p_value': explanation['p_value'],
                'cf_validity': cf_valid,
                'cf_proximity': cf_proximity,
                'correct': is_correct if has_labels else None
            }
            
            # Add uncertainty breakdown
            if 'uncertainty_breakdown' in explanation:
                result_entry['epistemic_uncertainty'] = explanation['uncertainty_breakdown']['epistemic']
                result_entry['aleatoric_uncertainty'] = explanation['uncertainty_breakdown']['aleatoric']
                result_entry['temporal_uncertainty'] = explanation['uncertainty_breakdown']['temporal']
                result_entry['prediction_reliability'] = explanation.get('prediction_reliability', 'unknown')
                result_entry['flagged_for_review'] = explanation.get('should_flag_for_review', False)
            
            results_list.append(result_entry)
        
        # Compute metrics
        metrics = {
            'anomaly_rate': np.mean(results['predictions']),
            'avg_confidence': np.mean(results['confidences']),
            'avg_score': np.mean(results['scores']),
            'avg_uncertainty': np.mean(results['uncertainties']),
            'n_samples': len(X_test),
            'total_samples': len(X_test)
        }
        
        # Uncertainty breakdown metrics
        if results['epistemic_uncertainties']:
            metrics['avg_epistemic_uncertainty'] = np.mean(results['epistemic_uncertainties'])
            metrics['avg_aleatoric_uncertainty'] = np.mean(results['aleatoric_uncertainties'])
            metrics['avg_temporal_uncertainty'] = np.mean(results['temporal_uncertainties'])
            metrics['samples_flagged_for_review'] = sum(results['flagged_for_review'])
            
            # Reliability distribution
            reliability_counts = {}
            for r in results['reliability_ratings']:
                reliability_counts[r] = reliability_counts.get(r, 0) + 1
            metrics['reliability_distribution'] = reliability_counts
        
        # Supervised metrics
        if has_labels:
            from sklearn.metrics import (
                f1_score, precision_score, recall_score, 
                roc_auc_score, accuracy_score, confusion_matrix
            )
            
            predictions = np.array(results['predictions']).astype(int)
            
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            metrics['f1'] = f1_score(y_test, predictions, zero_division=0)
            metrics['precision'] = precision_score(y_test, predictions, zero_division=0)
            metrics['recall'] = recall_score(y_test, predictions, zero_division=0)
            
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, results['scores'])
            except:
                metrics['roc_auc'] = 0.0
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            
            # Conformal coverage
            metrics['empirical_coverage'] = np.mean(results['coverage'])
        
        # Counterfactual metrics
        if 'cf_quality' in results and results['cf_quality']:
            metrics['cf_validity_rate'] = np.mean([
                cf['valid'] for cf in results['cf_quality']
            ])
            metrics['cf_avg_proximity'] = np.mean([
                cf['proximity'] for cf in results['cf_quality']
            ])
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save detailed results
            df = pd.DataFrame(results_list)
            df.to_csv(f'{output_dir}/evaluation_results_{timestamp}.csv', index=False)
            
            # Save metrics
            with open(f'{output_dir}/metrics_{timestamp}.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"\nResults saved to {output_dir}/")
        
        return metrics

    def save_model(self, path: str) -> None:
        """Save model and configuration."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.detector.state_dict(),
            'hyperparameters': self.hyperparameters,
            'conformal_threshold': self.conformal.threshold,
            'conformal_scores': self.conformal.calibration_scores,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'is_calibrated': self.is_calibrated
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model and configuration."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.detector.load_state_dict(checkpoint['model_state_dict'])
        self.hyperparameters = checkpoint['hyperparameters']
        self.conformal.threshold = checkpoint['conformal_threshold']
        self.conformal.calibration_scores = checkpoint['conformal_scores']
        
        if checkpoint['conformal_scores'] is not None:
            self.conformal.score_mean = np.mean(checkpoint['conformal_scores'])
            self.conformal.score_std = np.std(checkpoint['conformal_scores'])
        
        self.training_history = checkpoint.get('training_history', [])
        self.is_trained = checkpoint.get('is_trained', True)
        self.is_calibrated = checkpoint.get('is_calibrated', True)
        
        print(f"Model loaded from {path}")

    def summary(self) -> str:
        total_params = sum(p.numel() for p in self.detector.parameters())
        trainable_params = sum(
            p.numel() for p in self.detector.parameters() if p.requires_grad
        )
        
        summary = f"""
            {'-'*60}
            XAD Framework Summary
            {'-'*60}
            
            Architecture:
              Input dimension:    {self.input_dim}
              Sequence length:    {self.sequence_length}
              Hidden dimension:   {self.hyperparameters['hidden_dim']}
              Latent dimension:   {self.hyperparameters['latent_dim']}
              LSTM layers:        {self.hyperparameters['num_layers']}
              Dropout:            {self.hyperparameters['dropout']}
            
            Conformal Prediction:
              Alpha (significance): {self.alpha}
              Threshold:           {self.conformal.threshold if self.conformal.threshold else 'Not calibrated'}
              Calibration samples: {len(self.conformal.calibration_scores) if self.conformal.calibration_scores is not None else 0}
            
            Model Statistics:
              Total parameters:     {total_params:,}
              Trainable parameters: {trainable_params:,}
              Device:               {self.device}
            
            Status:
              Trained:     {self.is_trained}
              Calibrated:  {self.is_calibrated}
              Explainer:   {'Initialized' if self.explainer else 'Not initialized'}
            
            {'-'*60}
            """
        return summary
