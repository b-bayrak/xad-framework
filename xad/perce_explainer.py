"""
PerCE (Perturbation-based Counterfactual Explanations), Bayrak & Bach 2025,
per Section 3.5 / Appendix B.2 (Eq. 26-29).

This implementation follows the authors' own prior implementation
(xad_f10/experiments/counterfactual_experiments.py::PerCEExplainer), not a
from-scratch reading of Appendix B.2 alone, once it became available - two
concrete places where the paper's prose and the actual working code disagree:

1. Appendix B.2 says the InSample candidate is selected via DTW distance
   ("k-nearest neighbors... using DTW distance... select the candidate with
   minimum DTW"). The prior implementation uses plain Euclidean k-NN and
   takes the single nearest neighbor - no DTW anywhere. Adopted the simpler,
   actually-used Euclidean version: faster, and faithful to what evidently
   produced prior results, rather than a DTW variant that was apparently
   never actually run.
2. Channel/segment importance (Eq. 26-27) is computed as the *unsigned*
   change in anomaly score when a channel/segment is zeroed
   (abs(perturbed_score - original_score)), not a signed "drop" - a channel
   whose removal *increases* the anomaly score is still ranked as important,
   not deprioritized. Segment importance is computed once globally (not
   recomputed per-channel), and channels/segments are then tried in a single
   nested loop (channel order outer, the same global segment order inner),
   stopping the moment a perturbation achieves validity - simpler than a
   per-channel re-ranking scheme.

Difference from the original PerCE code: there, validity was checked against
a plain reconstruction-MSE score and a static threshold. Here validity uses
the same nonconformity measure and threshold as detection, GA-CE and DICE,
so all explainers share one definition of a valid counterfactual. This
implementation therefore routes every
validity/importance computation through `conformal_predictor._compute_nonconformity`
(read-only, matches what actually flagged the instance as anomalous and what
GA-CE/DICE are validated against) instead.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Union


class PerCEExplainer:
    """Perturbation-based Counterfactual Explanations (Bayrak & Bach, 2025), Eq. 26-29."""

    def __init__(
        self,
        model: torch.nn.Module,
        conformal_predictor,
        n_segments: int = 10,
        k_neighbors: int = 5,
        alpha_segment: float = 0.75,
        beta_channel: float = 0.9,
    ):
        self.model = model
        self.conformal_predictor = conformal_predictor
        self.n_segments = n_segments
        self.k_neighbors = k_neighbors
        self.alpha_segment = alpha_segment
        self.beta_channel = beta_channel

        self.normal_data: Optional[np.ndarray] = None
        self.normal_data_flat: Optional[np.ndarray] = None
        self.nn_model: Optional[NearestNeighbors] = None

    def fit(self, X_normal: Union[np.ndarray, torch.Tensor]) -> "PerCEExplainer":
        self.normal_data = X_normal.detach().cpu().numpy() if isinstance(X_normal, torch.Tensor) else np.array(X_normal)
        self.normal_data_flat = self.normal_data.reshape(len(self.normal_data), -1)
        self.nn_model = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(self.normal_data)))
        self.nn_model.fit(self.normal_data_flat)
        return self

    def _score(self, x: np.ndarray) -> float:
        """Real nonconformity score (base + TVD, Eq. 9) - see module docstring
        for why this replaces the prior code's plain-MSE score."""
        return self.conformal_predictor._compute_nonconformity(x, self.model)

    def _channel_importance(self, x: np.ndarray) -> np.ndarray:
        """I_ch(c) = |s(x) - s(x^(c))| (Eq. 26, unsigned per prior implementation)."""
        base_score = self._score(x)
        d = x.shape[1]
        importance = np.zeros(d)
        for c in range(d):
            x_perturbed = x.copy()
            x_perturbed[:, c] = 0.0
            importance[c] = abs(base_score - self._score(x_perturbed))
        return importance

    def _segment_importance(self, x: np.ndarray, T: int, seg_len: int) -> np.ndarray:
        """I_seg(s) = |s(x) - s(x^[s_i:s_{i+1}])| (Eq. 27), computed once globally."""
        base_score = self._score(x)
        importance = np.zeros(self.n_segments)
        for s in range(self.n_segments):
            start, end = s * seg_len, min((s + 1) * seg_len, T)
            x_perturbed = x.copy()
            x_perturbed[start:end, :] = 0.0
            importance[s] = abs(base_score - self._score(x_perturbed))
        return importance

    def _find_nearest_normal(self, x: np.ndarray) -> np.ndarray:
        """InSample candidate: nearest normal neighbor by Euclidean distance."""
        _, indices = self.nn_model.kneighbors(x.reshape(1, -1))
        return self.normal_data[indices[0, 0]].copy()

    def generate_counterfactual(self, x_anomaly: Union[np.ndarray, torch.Tensor], verbose: bool = False) -> Dict:
        if self.normal_data is None:
            raise ValueError("Must call fit() first with normal data")

        if isinstance(x_anomaly, torch.Tensor):
            x_anomaly = x_anomaly.detach().cpu().numpy()
        else:
            x_anomaly = np.array(x_anomaly)
        if x_anomaly.ndim == 3:
            x_anomaly = x_anomaly.squeeze(0)

        T, d = x_anomaly.shape
        seg_len = max(1, T // self.n_segments)
        threshold_score = getattr(self.conformal_predictor, 'threshold', None) or 1e-6

        channel_importance = self._channel_importance(x_anomaly)
        segment_importance = self._segment_importance(x_anomaly, T, seg_len)
        x_cand = self._find_nearest_normal(x_anomaly)

        channel_order = np.argsort(channel_importance)[::-1]
        segment_order = np.argsort(segment_importance)[::-1]

        x_cf = x_anomaly.copy()
        modified_segments, modified_channels = set(), set()
        found_valid = False

        for c in channel_order:
            for s in segment_order:
                start, end = s * seg_len, min((s + 1) * seg_len, T)
                x_cf[start:end, c] = (
                    (1 - self.alpha_segment) * x_anomaly[start:end, c]
                    + self.alpha_segment * x_cand[start:end, c]
                )
                modified_segments.add(s)
                modified_channels.add(c)

                if self._score(x_cf) < threshold_score:
                    found_valid = True
                    break
            if found_valid:
                break

        if not found_valid:
            # Step 4 (Eq. 29): channel-wide refinement if segment-level wasn't enough.
            for c in channel_order:
                x_cf[:, c] = (1 - self.beta_channel) * x_anomaly[:, c] + self.beta_channel * x_cand[:, c]
                modified_channels.add(c)
                if self._score(x_cf) < threshold_score:
                    found_valid = True
                    break

        if verbose:
            print(f"  PerCE: found_valid={found_valid}, score={self._score(x_cf):.4f} "
                  f"(threshold={threshold_score:.4f}), channels_modified={len(modified_channels)}")

        result = self.conformal_predictor.predict_with_confidence(x_cf, self.model, update_state=False)
        changes = x_cf - x_anomaly
        change_threshold = 0.01 * np.std(x_anomaly)
        sparsity = float(np.mean(np.abs(changes) <= change_threshold))

        return {
            'counterfactual': x_cf,
            'changes': changes,
            'distance': float(np.mean(changes ** 2)),
            'proximity': float(np.mean(changes ** 2)),
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'validity': result['prediction'] == 'normal',
            'is_valid': result['prediction'] == 'normal',
            'sparsity': sparsity,
            'insample_candidate': x_cand,
        }
