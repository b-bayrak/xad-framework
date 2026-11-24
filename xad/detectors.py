import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings


@dataclass
class DetectionResult:
    score: float
    is_anomaly: bool
    threshold: float
    reconstruction: np.ndarray
    latent: np.ndarray
    feature_importance: np.ndarray
    temporal_error: np.ndarray


class BaseDetector(ABC, nn.Module):
    """
    Abstract class for all  detectors.
    """
    
    def __init__(self, input_dim: int, sequence_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.name = "BaseDetector"
        self.threshold = None  # Will be set during training
        self._training_errors = []
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pass
    
    @abstractmethod
    def compute_anomaly_score(
        self, 
        x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[float, Dict[str, Any]]:
        pass
    
    def get_feature_importance(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor
    ) -> np.ndarray:
        """Compute per-feature reconstruction error."""
        error = (x - reconstruction) ** 2

        feature_importance = error.mean(dim=1).squeeze().detach().cpu().numpy()
        return feature_importance
    
    def set_threshold(self, percentile: float = 99.0):
        """Set anomaly threshold"""
        if len(self._training_errors) > 0:
            self.threshold = np.percentile(self._training_errors, percentile)
        else:
            warnings.warn("No training errors recorded. Using default threshold.")
            self.threshold = 0.1
    
    def is_anomaly(self, score: float) -> bool:
        if self.threshold is None:
            warnings.warn("Threshold not set. Call set_threshold() after training.")
            return False
        return score > self.threshold


class ConvAE(BaseDetector):
    def __init__(
        self, 
        input_dim: int, 
        sequence_length: int,
        latent_dim: int = 16,
        filters: List[int] = None,
        hidden_channels: List[int] = None,  # Alias for filters
        kernel_size: int = 7,
        dropout: float = 0.2
    ):
        super().__init__(input_dim, sequence_length)
        
        self.name = "Conv-AE"
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        
        if hidden_channels is not None:
            filters = hidden_channels
        if filters is None:
            filters = [32, 16]
        
        self.filters = filters
        
        padding = kernel_size // 2
        
        # ENCODER
        encoder_layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(filters):
            encoder_layers.extend([
                nn.Conv1d(
                    in_channels, out_channels, 
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate encoded size
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, sequence_length)
            encoded = self.encoder(dummy)
            self.encoded_length = encoded.shape[2]
            self.encoded_channels = encoded.shape[1]
        
        # Latent projection
        self.flatten_size = self.encoded_channels * self.encoded_length
        self.fc_latent = nn.Linear(self.flatten_size, latent_dim)
        self.fc_expand = nn.Linear(latent_dim, self.flatten_size)
        
        # DECODER
        decoder_layers = []
        reversed_filters = list(reversed(filters))
        in_channels = reversed_filters[0]
        
        for i, out_channels in enumerate(reversed_filters[1:] + [input_dim]):
            # Use ConvTranspose1d for upsampling
            decoder_layers.extend([
                nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=1
                ),
                nn.BatchNorm1d(out_channels) if out_channels != input_dim else nn.Identity(),
                nn.ReLU(inplace=True) if out_channels != input_dim else nn.Identity(),
                nn.Dropout(dropout) if out_channels != input_dim else nn.Identity()
            ])
            in_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply encoder
        encoded = self.encoder(x)
        
        # Flatten and project to latent
        flat = encoded.view(encoded.size(0), -1)
        latent = self.fc_latent(flat)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        expanded = self.fc_expand(latent)
        expanded = expanded.view(-1, self.encoded_channels, self.encoded_length)
        
        decoded = self.decoder(expanded)
        
        if decoded.shape[2] != self.sequence_length:
            decoded = F.interpolate(
                decoded, 
                size=self.sequence_length, 
                mode='linear', 
                align_corners=False
            )
        
        # (batch, features, seq_len) -> (batch, seq_len, features)
        reconstruction = decoded.permute(0, 2, 1)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def compute_anomaly_score(
        self, 
        x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[float, Dict[str, Any]]:
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        device = next(self.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            reconstruction, latent = self.forward(x)
            
            # Per-sample MSE (reconstruction error)
            mse = torch.mean((reconstruction - x) ** 2, dim=(1, 2))
            score = mse.item()
            
            # Feature importance
            feature_importance = self.get_feature_importance(x, reconstruction)
            
            # Temporal error profile
            temporal_error = torch.mean((reconstruction - x) ** 2, dim=2)
            temporal_error = temporal_error.squeeze().cpu().numpy()
        
        return score, {
            'reconstruction': reconstruction.squeeze().cpu().numpy(),
            'latent': latent.squeeze().cpu().numpy(),
            'feature_importance': feature_importance,
            'temporal_error': temporal_error,
            'is_anomaly': self.is_anomaly(score) if self.threshold else None
        }



class LSTM_AE(BaseDetector):
    def __init__(
        self, 
        input_dim: int, 
        sequence_length: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__(input_dim, sequence_length)
        
        self.name = "LSTM-AE"
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # ENCODER
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.encoder_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        
        self.fc_latent = nn.Linear(hidden_dim * self.num_directions, latent_dim)
        
        # DECODER
        self.fc_expand = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Decoder is unidirectional
        )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        
        outputs, (hidden, cell) = self.encoder_lstm(x)
        
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.encoder_norm(hidden)
        latent = self.fc_latent(hidden)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.size(0)
        
        expanded = self.fc_expand(latent)
        expanded = expanded.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        outputs, _ = self.decoder_lstm(expanded)
        
        outputs = self.decoder_norm(outputs)
        reconstruction = self.fc_out(outputs)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def compute_anomaly_score(
        self, 
        x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[float, Dict[str, Any]]:
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        device = next(self.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            reconstruction, latent = self.forward(x)
            
            mse = torch.mean((reconstruction - x) ** 2, dim=(1, 2))
            score = mse.item()
            
            feature_importance = self.get_feature_importance(x, reconstruction)
            temporal_error = torch.mean((reconstruction - x) ** 2, dim=2).squeeze().cpu().numpy()
        
        return score, {
            'reconstruction': reconstruction.squeeze().cpu().numpy(),
            'latent': latent.squeeze().cpu().numpy(),
            'feature_importance': feature_importance,
            'temporal_error': temporal_error,
            'is_anomaly': self.is_anomaly(score) if self.threshold else None
        }


class LSTM_VAE(BaseDetector):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        alpha_score: float = 0.1
    ):
        super().__init__(input_dim, sequence_length)

        self.name = "LSTM-VAE"
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # Weight on the KL term in the anomaly score (Eq. 4 in the paper):
        # s_VAE(x) = ||x - x_hat||^2 + alpha_score * KL(q(z|x) || p(z)).
        # This is distinct from the training-time KL weight (beta, Eq. 3) which
        # regularizes the latent space; alpha_score only affects scoring/detection.
        self.alpha_score = alpha_score
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.encoder_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        outputs, (hidden, _) = self.encoder_lstm(x)
        
        # Concatenate bidirectional hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.encoder_norm(hidden)
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        
        # Expand to sequence
        hidden = self.fc_decode(z)
        hidden = hidden.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # LSTM decoding
        outputs, _ = self.decoder_lstm(hidden)
        outputs = self.decoder_norm(outputs)
        
        # Project to output
        reconstruction = self.fc_out(outputs)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def compute_anomaly_score(
        self, 
        x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[float, Dict[str, Any]]:
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        device = next(self.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            reconstruction, mu, logvar = self.forward(x)
            
            # Reconstruction error
            recon_error = torch.mean((reconstruction - x) ** 2, dim=(1, 2))

            # KL divergence (measure of latent uncertainty)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            # Score incorporates latent-space regularization (Eq. 4):
            # s_VAE(x) = ||x - x_hat||^2 + alpha_score * KL(q(z|x) || p(z)).
            # This lets LSTM-VAE flag instances that reconstruct well but occupy
            # atypical regions of latent space, unlike Conv-AE/LSTM-AE which only
            # see reconstruction error.
            score = (recon_error + self.alpha_score * kl_div).item()

            feature_importance = self.get_feature_importance(x, reconstruction)
            temporal_error = torch.mean((reconstruction - x) ** 2, dim=2).squeeze().cpu().numpy()

        return score, {
            'reconstruction': reconstruction.squeeze().cpu().numpy(),
            'latent_mu': mu.squeeze().cpu().numpy(),
            'latent_logvar': logvar.squeeze().cpu().numpy(),
            'feature_importance': feature_importance,
            'temporal_error': temporal_error,
            'reconstruction_error': recon_error.item(),
            'kl_divergence': kl_div.item(),
            'is_anomaly': self.is_anomaly(score) if self.threshold else None
        }


def compute_vae_loss(
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 0.001
) -> Tuple[ torch.Tensor, torch.Tensor, torch.Tensor ]:
    recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


DETECTOR_REGISTRY = {
    'conv_ae': ConvAE,
    'lstm_ae': LSTM_AE,
    'lstm_vae': LSTM_VAE,

}

DETECTOR_INFO = {
    'conv_ae': {'name': 'Conv-AE'},
    'lstm_ae': {'name': 'LSTM-AE'},
    'lstm_vae': {'name': 'LSTM-VAE'}
    }


# Constructor kwargs each architecture actually accepts. create_detector()
# is commonly called with one shared config dict (hidden_dim, latent_dim,
# num_layers, dropout, ...) across all three detector types for convenience,
# but ConvAE has no hidden_dim/num_layers concept - passing them through
# unfiltered raises TypeError. Route only what each class supports.
_DETECTOR_ALLOWED_KWARGS = {
    'conv_ae': {'latent_dim', 'filters', 'hidden_channels', 'kernel_size', 'dropout'},
    'lstm_ae': {'hidden_dim', 'latent_dim', 'num_layers', 'dropout', 'bidirectional'},
    'lstm_vae': {'hidden_dim', 'latent_dim', 'num_layers', 'dropout', 'alpha_score'},
}


def create_detector(
    detector_type: str,
    input_dim: int,
    sequence_length: int,
    **kwargs
) -> BaseDetector:
    detector_type = detector_type.lower().replace('-', '_')

    if detector_type not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector type: {detector_type}. "
            f"Available: {list(DETECTOR_REGISTRY.keys())}"
        )

    allowed = _DETECTOR_ALLOWED_KWARGS[detector_type]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    return DETECTOR_REGISTRY[detector_type](
        input_dim=input_dim,
        sequence_length=sequence_length,
        **filtered_kwargs
    )


def list_detectors() -> Dict[str, Dict]:
    return DETECTOR_INFO.copy()


class DetectorTrainer:
    """
    - Training loop with early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Threshold calibration
    """
    
    def __init__(
        self,
        detector: BaseDetector,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        kl_weight: float = 0.001,
        device: str = None
    ):
        self.detector = detector
        self.kl_weight = kl_weight
        
        if device is None:
            device = 'cpu'
            if torch.cuda.is_available():
                # Probe that CUDA actually runs a kernel: on nodes whose GPU
                # is older than the installed wheel supports (e.g. P100 with
                # a sm_70+ build), is_available() is True but every launch
                # raises - fall back to CPU instead of crashing mid-training.
                try:
                    (torch.zeros(1, device='cuda') + 1).item()
                    device = 'cuda'
                except Exception:
                    device = 'cpu'
        self.device = torch.device(device)
        self.detector.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            detector.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.detector(x)
        
        if isinstance(self.detector, LSTM_VAE):
            reconstruction, mu, logvar = outputs
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstruction, x)
            
            # KL divergence with annealing
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_weight = min(self.kl_weight, self.kl_weight * (epoch + 1) / 10)
            
            loss = recon_loss + kl_weight * kl_loss
            
            return loss, {'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item()}
        else:
            reconstruction = outputs[0]
            loss = F.mse_loss(reconstruction, x)
            return loss, {'recon_loss': loss.item()}
    
    def train_epoch(
        self, 
        train_loader: torch.utils.data.DataLoader,
        epoch: int = 0
    ) -> Tuple[float, List[float]]:
        self.detector.train()
        total_loss = 0.0
        n_batches = 0
        all_errors = []
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Compute loss
            loss, _ = self.compute_loss(x, epoch)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.detector.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Record reconstruction errors for threshold calibration
            with torch.no_grad():
                outputs = self.detector(x)
                reconstruction = outputs[0]
                errors = torch.mean((reconstruction - x) ** 2, dim=(1, 2))
                all_errors.extend(errors.cpu().numpy().tolist())
        
        return total_loss / n_batches, all_errors
    
    def validate(
        self, 
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        self.detector.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                
                outputs = self.detector(x)
                reconstruction = outputs[0]
                
                loss = F.mse_loss(reconstruction, x)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 100,
        patience: int = 15,
        threshold_percentile: float = 99.0,
        verbose: bool = True
    ) -> Dict[str, list]:
        """Full training loop with early stopping and threshold calibration."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        all_training_errors = []
        
        if verbose:
            print(f"\nTraining {self.detector.name}")
            print(f"Device: {self.device}")
            print("-" * 50)
        
        for epoch in range(epochs):
            train_loss, epoch_errors = self.train_epoch(train_loader, epoch)
            all_training_errors.extend(epoch_errors)
            
            history['train_loss'].append(train_loss)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.detector.state_dict().items()}
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f}")
        
        # Restore best model
        if best_state is not None:
            self.detector.load_state_dict(best_state)
        
        # Calibrate threshold
        self.detector._training_errors = all_training_errors
        self.detector.set_threshold(threshold_percentile)
        
        if verbose:
            print(f"\nThreshold set to {self.detector.threshold:.6f} "
                  f"({threshold_percentile}th percentile)")
        
        return history


