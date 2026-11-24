import numpy as np
import torch
from collections import deque
from typing import Dict, Union, Optional, List
from dataclasses import dataclass


@dataclass
class PredictionResult:
    prediction: str  # 'normal' or 'anomaly'
    confidence: float
    score: float
    threshold: float
    uncertainty: float
    p_value: float


class TemporalConformalPredictor:
    """
    Split conformal prediction with a temporal-aware nonconformity measure
    (Eq. 9-10) and one of three threshold policies (`mode`):

    - 'static':  plain split conformal - one threshold fixed at calibration.
    - 'sliding': sliding-window recalibration (Eq. 8): the threshold
      q_hat_alpha is recomputed from only the most recent `window_size`
      observed nonconformity scores. Only instances classified normal are fed
      back into the rolling window; a point flagged anomalous is not treated
      as drawn from the reference ("normal") distribution, so it must not
      contaminate the threshold used for subsequent points. Known failure
      mode: under sustained upward drift of *normal* scores, drifted normals
      get flagged and are then excluded from the very window that should
      adapt to them - a self-locking under-coverage loop.
    - 'aci':     Adaptive Conformal Inference (Gibbs & Candes, 2021). The
      nominal level alpha_t is updated online from the realized error
      sequence, alpha_{t+1} = alpha_t + gamma * (alpha - err_t) with
      err_t = 1[x_t flagged anomalous], and the threshold is the
      (1 - alpha_t) conformal quantile of the FIXED calibration scores.
      This directly targets long-run coverage under arbitrary distribution
      shift (their Prop. 4.1: the average flag rate converges to alpha),
      with the honest caveat that during a sustained burst of true
      anomalies ACI will raise the threshold to keep the flag rate near
      alpha, trading detection power for false-alarm control - the
      opposite failure mode of 'sliding'. Report both; that comparison is
      part of the paper's contribution.

    Both 'sliding' and 'aci' require `predict_with_confidence` to be called
    on test instances **in chronological order** (batch_predict already does
    this); calling out of order silently breaks the recalibration.

    Independent of the mode, the nonconformity measure (`_compute_tvd`,
    Eq. 9-10) penalizes reconstructions whose *first differences* (local
    dynamics) don't match the true signal's, even when point-wise
    reconstruction error is low; `smoothness_weight=0.0` disables it.
    """

    MODES = ('static', 'sliding', 'aci')

    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 500,
        smoothness_weight: float = 0.1,
        adaptive: bool = True,
        mode: Optional[str] = None,
        aci_gamma: float = 0.005,
    ):
        # `mode` supersedes the legacy `adaptive` flag; when mode is not
        # given, adaptive=True maps to 'sliding' and adaptive=False to
        # 'static', preserving existing call sites' behavior exactly.
        if mode is None:
            mode = 'sliding' if adaptive else 'static'
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.adaptive = (mode == 'sliding')  # kept for backward compat

        self.alpha = alpha
        self.window_size = window_size
        self.smoothness_weight = smoothness_weight
        self.aci_gamma = aci_gamma

        self.calibration_scores: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.score_mean: Optional[float] = None
        self.score_std: Optional[float] = None

        # Rolling state for sliding-window recalibration (Eq. 8). Initialized
        # in calibrate() from the tail of the calibration set. `window` holds
        # the most recent up-to-`window_size` nonconformity scores that are
        # currently in force; `threshold_history` records how q_hat_alpha
        # evolved, for diagnostics/plotting.
        self.window: Optional[deque] = None
        self.threshold_history: List[float] = []

        # ACI state: the online nominal level alpha_t and its trajectory.
        self.alpha_t: float = alpha
        self.alpha_history: List[float] = []

    def calibrate(self, X_calib: Union[np.ndarray, torch.Tensor], model: torch.nn.Module, verbose: bool = True) -> None:
        """
        Calibrate conformal predictor on a held-out calibration set.

        `X_calib` must contain only instances exchangeable with the
        population the coverage guarantee is meant to hold for - i.e. normal
        instances. Passing a calibration set contaminated with anomalies
        inflates q_hat_alpha and silently weakens the guarantee; callers are
        responsible for filtering (see `data_loaders.BaseDataLoader` normal
        subset helpers).
        """
        model.eval()
        nonconformity_scores = []

        with torch.no_grad():
            for i in range(len(X_calib)):
                score = self._compute_nonconformity(X_calib[i], model)
                nonconformity_scores.append(score)

        self.calibrate_from_scores(np.array(nonconformity_scores), verbose=verbose)

    def calibrate_from_scores(self, nonconformity_scores: np.ndarray, verbose: bool = False) -> None:
        """
        Calibrate directly from precomputed nonconformity scores. Used by the
        CP-mode ablation (exp_6) so the expensive model forward passes are
        done once per nonconformity configuration and every threshold policy
        (static/sliding/aci) x alpha is evaluated through the exact same
        class logic on the same scores.
        """
        self.calibration_scores = np.asarray(nonconformity_scores, dtype=float)

        self.score_mean = np.mean(self.calibration_scores)
        self.score_std = np.std(self.calibration_scores)

        self.threshold = self._quantile(self.calibration_scores, self.alpha)
        self.threshold_history = [self.threshold]

        # Initialize the rolling window from the most recent `window_size`
        # calibration scores (Eq. 8's D_cal(t) at the start of the test
        # stream). If the calibration set is smaller than window_size, use
        # all of it.
        tail = self.calibration_scores[-self.window_size:]
        self.window = deque(tail.tolist(), maxlen=self.window_size)

        # ACI starts at the nominal level.
        self.alpha_t = self.alpha
        self.alpha_history = [self.alpha_t]

        if verbose:
            print(f"\n{'-'*50}")
            print("Conformal Predictor Calibration")
            print(f"{'-'*50}")
            print(f"  Calibration samples: {len(self.calibration_scores)}")
            print(f"  Significance level (alpha): {self.alpha}")
            print(f"  Threshold policy (mode): {self.mode}"
                  + (f" (gamma={self.aci_gamma})" if self.mode == 'aci' else ""))
            print(f"  Window size: {self.window_size}")
            print(f"  Initial threshold: {self.threshold:.6f}")
            print(f"  Score statistics: mean={self.score_mean:.6f}, std={self.score_std:.6f}")
            print(f"{'-'*50}\n")

    @staticmethod
    def _quantile(scores: np.ndarray, alpha: float) -> float:
        """(m+1)(1-alpha)/m empirical quantile, per Eq. 6-7."""
        m = len(scores)
        q_level = np.ceil((m + 1) * (1 - alpha)) / m
        q_level = min(q_level, 1.0)
        return float(np.quantile(scores, q_level))

    @staticmethod
    def _quantile_aci(scores: np.ndarray, alpha_t: float) -> float:
        """
        Conformal quantile for ACI, where the online alpha_t can leave (0,1).
        Following Gibbs & Candes (2021): if the required order statistic
        ceil((m+1)(1-alpha_t)) exceeds m the threshold is +inf (never flag -
        err_t=0, so alpha_t recovers upward); if it is < 1 the threshold is
        -inf (always flag - err_t=1, so alpha_t recovers downward). Clipping
        to the sample max/min instead (as `_quantile` does) would stall the
        recovery dynamics ACI's guarantee relies on.
        """
        m = len(scores)
        k = int(np.ceil((m + 1) * (1 - alpha_t)))
        if k > m:
            return float('inf')
        if k < 1:
            return float('-inf')
        return float(np.sort(scores)[k - 1])

    def reset_window(self) -> None:
        """Reset all online state back to right after calibrate()."""
        if self.calibration_scores is None:
            raise RuntimeError("Predictor not calibrated. Call calibrate() first.")
        tail = self.calibration_scores[-self.window_size:]
        self.window = deque(tail.tolist(), maxlen=self.window_size)
        self.threshold = self._quantile(np.array(self.window), self.alpha)
        self.threshold_history = [self.threshold]
        self.alpha_t = self.alpha
        self.alpha_history = [self.alpha_t]

    def _compute_tvd(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> float:
        """
        Total variation distance between reconstructed and true dynamics
        (Eq. 10): sum over t of || (x_hat_{t+1}-x_hat_t) - (x_{t+1}-x_t) ||_2.

        This compares first differences (how fast the signal moves), not raw
        values or reconstruction error - a reconstruction can have low
        point-wise error yet fail to track the true dynamics, and vice versa.
        """
        if x.dim() == 3:
            x = x.squeeze(0)
        if reconstruction.dim() == 3:
            reconstruction = reconstruction.squeeze(0)

        if x.shape[0] < 2:
            return 0.0

        true_diff = x[1:] - x[:-1]
        recon_diff = reconstruction[1:] - reconstruction[:-1]
        per_step_norm = torch.linalg.norm(recon_diff - true_diff, dim=-1)
        return float(torch.sum(per_step_norm).item())

    def _compute_nonconformity(
        self,
        x: Union[np.ndarray, torch.Tensor],
        model: torch.nn.Module
    ) -> float:
        """
        Nonconformity measure s~(x) = s(x) + lambda_temp * TVD(x, x_hat)
        (Eq. 9), combining the detector's own anomaly score with the temporal
        dynamics penalty (Eq. 10, see `_compute_tvd`).
        """
        if isinstance(x, torch.Tensor):
            x_t = x.clone()
            if x_t.dim() == 2:
                x_t = x_t.unsqueeze(0)
        else:
            x_t = torch.from_numpy(np.array(x)).float()
            if x_t.dim() == 2:
                x_t = x_t.unsqueeze(0)

        device = next(model.parameters()).device
        x_t = x_t.to(device)

        with torch.no_grad():
            anomaly_score, score_details = model.compute_anomaly_score(x_t)

        if isinstance(anomaly_score, torch.Tensor):
            base_score = anomaly_score.detach().cpu().numpy().mean()
        else:
            base_score = float(np.mean(anomaly_score))

        smoothness_penalty = 0.0
        reconstruction = score_details.get('reconstruction', None) if isinstance(score_details, dict) else None
        if reconstruction is not None:
            recon_t = reconstruction if isinstance(reconstruction, torch.Tensor) else torch.from_numpy(np.asarray(reconstruction)).float()
            x_cpu = x_t.squeeze(0).detach().cpu()
            recon_t = recon_t.to(x_cpu.device).float()
            if recon_t.shape == x_cpu.shape:
                smoothness_penalty = self._compute_tvd(x_cpu, recon_t)

        nonconformity = base_score + self.smoothness_weight * smoothness_penalty

        return float(nonconformity)

    def predict_with_confidence(
        self,
        x: Union[np.ndarray, torch.Tensor, None],
        model: Optional[torch.nn.Module] = None,
        update_state: bool = True,
        precomputed_score: Optional[float] = None,
    ) -> Dict[str, Union[str, float]]:
        """
        Predict on a single instance. If `self.adaptive` is True, instances
        must be passed in chronological order: the threshold used here is
        recomputed from the current rolling window (Eq. 8) *before* scoring
        x, and x's score is folded into the window afterward only if x is
        classified normal.

        `update_state=False` scores x against the CURRENT window/threshold
        without mutating anything - use this for any "what if" scoring that
        is not a genuine, chronologically-next observation. In particular,
        GA-CE (and any other counterfactual search) calls this many times per
        candidate solution while optimizing; those candidates are synthetic
        and must never be folded into the calibration window or overwrite
        `self.threshold`, or the adaptive recalibration silently drifts based
        on search candidates instead of real observations.
        """
        if self.threshold is None:
            raise RuntimeError("Predictor not calibrated. Call calibrate() first.")

        # `precomputed_score` lets callers (exp_6 CP-mode ablation) score the
        # stream once and replay it under several threshold policies without
        # repeating the model forward passes; the policy logic below is
        # identical either way.
        if precomputed_score is not None:
            score = float(precomputed_score)
        else:
            score = self._compute_nonconformity(x, model)

        if self.mode == 'sliding' and self.window is not None and len(self.window) > 0:
            current_threshold = self._quantile(np.array(self.window), self.alpha)
            current_scores = np.array(self.window)
        elif self.mode == 'aci':
            current_threshold = self._quantile_aci(self.calibration_scores, self.alpha_t)
            current_scores = self.calibration_scores
        else:
            current_threshold = self.threshold
            current_scores = self.calibration_scores

        is_anomaly = score > current_threshold

        p_value = self._compute_p_value(score, current_scores)
        # Confidence needs a finite reference threshold; when ACI's alpha_t
        # has left (0,1) the decision threshold is +/-inf, so fall back to
        # the static calibration threshold for the *confidence* computation
        # only (the decision itself still uses the ACI threshold).
        finite_threshold = current_threshold if np.isfinite(current_threshold) else self.threshold
        confidence = self._compute_confidence(score, finite_threshold, current_scores)
        uncertainty = self._compute_uncertainty(score, current_scores)

        if update_state:
            if self.mode == 'sliding':
                self.threshold = current_threshold
                if self.window is not None and not is_anomaly:
                    self.window.append(score)
                    self.threshold_history.append(current_threshold)
            elif self.mode == 'aci':
                # alpha_{t+1} = alpha_t + gamma * (alpha - err_t); err_t = 1
                # iff this (real, chronologically-next) instance was flagged.
                err_t = 1.0 if is_anomaly else 0.0
                self.alpha_t = self.alpha_t + self.aci_gamma * (self.alpha - err_t)
                self.alpha_history.append(self.alpha_t)
                self.threshold_history.append(current_threshold)

        return {'prediction': 'anomaly' if is_anomaly else 'normal',
            'confidence': confidence,
            'score': score,
            'threshold': current_threshold,
            'uncertainty': uncertainty,
            'p_value': p_value}

    def _compute_p_value(self, score: float, reference_scores: Optional[np.ndarray] = None) -> float:
        # The p-value is the proportion of (reference) calibration scores
        # that are greater than or equal to the current score.
        scores = reference_scores if reference_scores is not None else self.calibration_scores
        if scores is None or len(scores) == 0:
            return 0.5
        return float(np.mean(scores >= score))

    def _compute_confidence(
        self,
        score: float,
        threshold: Optional[float] = None,
        reference_scores: Optional[np.ndarray] = None
    ) -> float:
        thr = threshold if threshold is not None else self.threshold
        scores = reference_scores if reference_scores is not None else self.calibration_scores

        if thr is None or scores is None or len(scores) == 0:
            return 0.5

        std = np.std(scores)
        distance = abs(score - thr)  # normalized distance

        normalized_distance = distance / std if std > 0 else distance

        # Sigmoid transformation for smoother confidence
        confidence = 1 / (1 + np.exp(-normalized_distance))

        return float(min(0.99, confidence))

    def _compute_uncertainty(self, score: float, reference_scores: Optional[np.ndarray] = None) -> float:
        scores = reference_scores if reference_scores is not None else self.calibration_scores
        if scores is None or len(scores) == 0:
            return 1.0

        # Distance from nearest calibration scores
        distances = np.abs(scores - score)
        nearest_distance = np.min(distances)

        std = np.std(scores)
        uncertainty = nearest_distance / std if std > 0 else nearest_distance

        return float(min(1.0, uncertainty))

    def get_prediction_interval(
        self,
        x: Union[np.ndarray, torch.Tensor],
        model: torch.nn.Module
    ) -> Dict[str, float]:
        if self.calibration_scores is None:
            raise RuntimeError("Predictor not calibrated.")

        score = self._compute_nonconformity(x, model)

        reference = np.array(self.window) if (self.adaptive and self.window is not None) else self.calibration_scores

        # Prediction interval based on calibration scores
        lower_quantile = self.alpha / 2
        upper_quantile = 1 - self.alpha / 2

        lower_bound = np.quantile(reference, lower_quantile)
        upper_bound = np.quantile(reference, upper_quantile)

        return {
            'score': score,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'in_interval': lower_bound <= score <= upper_bound
        }

    def batch_predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        model: torch.nn.Module,
        return_details: bool = False
    ) -> Union[np.ndarray, List[Dict]]:
        """
        Predict over X in order. If `self.adaptive` is True, X must be in
        chronological order for the sliding-window recalibration to be
        meaningful (see class docstring).
        """
        predictions = []
        details = []

        for i in range(len(X)):
            result = self.predict_with_confidence(X[i], model)
            predictions.append(1 if result['prediction'] == 'anomaly' else 0)
            if return_details:
                details.append(result)

        if return_details:
            return details

        return np.array(predictions)

    def get_calibration_statistics(self) -> Dict[str, float]:
        if self.calibration_scores is None:
            return {}

        return {
            'n_samples': len(self.calibration_scores),
            'mean': float(self.score_mean) if self.score_mean else 0,
            'std': float(self.score_std) if self.score_std else 0,
            'min': float(np.min(self.calibration_scores)),
            'max': float(np.max(self.calibration_scores)),
            'median': float(np.median(self.calibration_scores)),
            'threshold': float(self.threshold) if self.threshold else 0,
            'alpha': self.alpha,
            'mode': self.mode,
            'adaptive': self.adaptive,
            'aci_gamma': self.aci_gamma,
            'current_alpha_t': self.alpha_t,
            'window_size': self.window_size,
            'current_window_len': len(self.window) if self.window is not None else 0
        }
