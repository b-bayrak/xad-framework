"""
DICE (Diverse Counterfactual Explanations), adapted from Mothilal et al. 2020
per Section 3.5 / Appendix B.1 (Eq. 24-25):

    L_DICE = sum_i [L_pred(x'_i) + lambda_prox ||x - x'_i||^2] - lambda_div sum_{i<j} ||x'_i - x'_j||^2

Note on hyperparameters (Appendix C lists "diversity/proximity/sparsity weights
[1.0, 0.5, 0.2]" for DICE, but Eq. 24 - the equation this class implements
exactly - has no sparsity term. This is an inconsistency in the source
document, not resolved silently here: this implementation follows Eq. 24
verbatim (prediction + proximity - diversity, no sparsity term), uses
lambda_div=1.0 and lambda_prox=0.5 from that list for the terms that do
appear, and does not invent a sparsity term to make the two sources agree.

L_pred is implemented as a differentiable hinge-style validity loss against a
differentiable reconstruction of the SAME nonconformity measure the
conformal predictor actually thresholds against (base score + TVD penalty,
Eq. 9-10) - not just the raw reconstruction score. This matters: the
conformal threshold q_hat is calibrated on the TVD-augmented measure, which
can be on a meaningfully different scale than the raw reconstruction score
alone. Comparing a raw-score-only proxy against a TVD-inclusive
threshold risks the optimizer thinking a candidate is "already valid" when
it isn't (or the reverse), independent of gradient descent. The conformal predictor's
own `_compute_nonconformity` cannot be reused directly here since it wraps
the detector forward pass in `torch.no_grad()`; `_differentiable_score`
below recomputes the identical formula (Eq. 4, Eq. 9-10) without it.
Reported validity is still always checked against the real conformal
predictor after optimization, not against this proxy, so no scoping choice
here can make validity look better than it is.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Union


def _differentiable_tvd(x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Differentiable version of TemporalConformalPredictor._compute_tvd (Eq. 10)."""
    if x.dim() == 3:
        x = x.squeeze(0)
    if reconstruction.dim() == 3:
        reconstruction = reconstruction.squeeze(0)
    if x.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    true_diff = x[1:] - x[:-1]
    recon_diff = reconstruction[1:] - reconstruction[:-1]
    return torch.sum(torch.linalg.norm(recon_diff - true_diff, dim=-1))


def _differentiable_score(detector: torch.nn.Module, x: torch.Tensor, smoothness_weight: float = 0.1) -> torch.Tensor:
    """
    Differentiable nonconformity score: matches
    TemporalConformalPredictor._compute_nonconformity (base score + Eq. 9-10
    TVD penalty) but without the torch.no_grad() wrapper, so gradients w.r.t.
    x flow. x: (1, T, d) or (T, d) tensor, requires_grad-compatible.
    """
    x_in = x.unsqueeze(0) if x.dim() == 2 else x

    outputs = detector(x_in)
    if len(outputs) == 3:  # LSTM_VAE: (reconstruction, mu, logvar)
        reconstruction, mu, logvar = outputs
        recon_error = torch.mean((reconstruction - x_in) ** 2, dim=(1, 2))
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        alpha_score = getattr(detector, 'alpha_score', 0.1)
        base_score = (recon_error + alpha_score * kl_div).squeeze()
    else:  # ConvAE / LSTM_AE: (reconstruction, latent)
        reconstruction = outputs[0]
        base_score = torch.mean((reconstruction - x_in) ** 2, dim=(1, 2)).squeeze()

    tvd = _differentiable_tvd(x, reconstruction)
    return base_score + smoothness_weight * tvd


class DICEExplainer:
    """
    Diverse Counterfactual Explanations (Mothilal et al., 2020), Eq. 24-25.

    Hyperparameters (lambda_div=0.2, lambda_prox=0.5, max_iterations=100)
    match the authors' own prior implementation
    (xad_f10/experiments/counterfactual_experiments.py::DICEExplainer), not
    the earlier guess of lambda_div=1.0 inferred from the paper's ambiguous
    Appendix C wording ("diversity/proximity/sparsity weights [1.0, 0.5,
    0.2]", read literally in that order). The prior code's actual defaults
    make clear the "[1.0, 0.5, 0.2]" list is [validity (implicit, always 1),
    proximity, diversity] - not [diversity, proximity, sparsity] - and that
    there is no diversity=1.0 anywhere in the working implementation. That
    matters mechanically: an unnormalized diversity term with weight 1.0,
    summed over all K*(K-1)/2 pairs, can dominate the gradient over the
    single-candidate validity/proximity terms and prevent convergence to
    validity at all - which is exactly the failure mode observed before this
    fix (0% validity, anomalously large proximity, on every architecture).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        conformal_predictor,
        k_counterfactuals: int = 5,
        n_neighbors: int = 10,
        lambda_prox: float = 0.5,
        lambda_div: float = 0.2,
        learning_rate: float = 2e-2,
        max_iterations: int = 300,
    ):
        self.model = model
        self.conformal_predictor = conformal_predictor
        self.k = k_counterfactuals
        self.n_neighbors = n_neighbors
        self.lambda_prox = lambda_prox
        self.lambda_div = lambda_div
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

        self.normal_data: Optional[np.ndarray] = None
        self.normal_data_flat: Optional[np.ndarray] = None
        self.nn_model: Optional[NearestNeighbors] = None

    def fit(self, X_normal: Union[np.ndarray, torch.Tensor]) -> "DICEExplainer":
        self.normal_data = X_normal.detach().cpu().numpy() if isinstance(X_normal, torch.Tensor) else np.array(X_normal)
        self.normal_data_flat = self.normal_data.reshape(len(self.normal_data), -1)
        self.nn_model = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self.normal_data)))
        self.nn_model.fit(self.normal_data_flat)
        return self

    def generate_counterfactual(self, x_anomaly: Union[np.ndarray, torch.Tensor], verbose: bool = False) -> Dict:
        """Generates K diverse counterfactuals (Eq. 24-25); returns the best one
        (lowest proximity among those the real conformal predictor accepts as
        valid, or lowest final loss if none are valid) in the same result
        schema GA-CE uses, plus `all_counterfactuals` with the full set."""
        if self.normal_data is None:
            raise ValueError("Must call fit() first with normal data")

        if isinstance(x_anomaly, torch.Tensor):
            x_anomaly = x_anomaly.detach().cpu().numpy()
        else:
            x_anomaly = np.array(x_anomaly)
        if x_anomaly.ndim == 3:
            x_anomaly = x_anomaly.squeeze(0)

        device = next(self.model.parameters()).device
        x_t = torch.from_numpy(x_anomaly).float().to(device)

        x_flat = x_anomaly.reshape(1, -1)
        _, indices = self.nn_model.kneighbors(x_flat)
        neighbor_pool = indices[0]

        # Initialize K counterfactuals by interpolating with distinct normal neighbors
        cfs = []
        for i in range(self.k):
            neighbor_idx = neighbor_pool[i % len(neighbor_pool)]
            neighbor = self.normal_data[neighbor_idx]
            alpha = np.random.uniform(0.3, 0.7)
            init = alpha * x_anomaly + (1 - alpha) * neighbor
            cfs.append(torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True))

        q_hat = getattr(self.conformal_predictor, 'threshold', None) or 1e-6
        smoothness_weight = getattr(self.conformal_predictor, 'smoothness_weight', 0.1)
        optimizer = torch.optim.Adam(cfs, lr=self.learning_rate)

        # Detector set to train() mode (needed for some LSTM cuDNN backward
        # paths) with its own parameters frozen, matching the prior
        # implementation - we only ever want gradients w.r.t. the
        # counterfactual candidates, never w.r.t. the (already-trained,
        # cached) detector weights.
        was_training = self.model.training
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(False)

        try:
            for it in range(self.max_iterations):
                optimizer.zero_grad()

                l_pred = sum(torch.relu(_differentiable_score(self.model, cf, smoothness_weight) - q_hat) for cf in cfs)
                l_prox = sum(torch.mean((cf - x_t) ** 2) for cf in cfs)
                l_div = sum(
                    torch.mean((cfs[i] - cfs[j]) ** 2)
                    for i in range(self.k) for j in range(i + 1, self.k)
                )

                loss = l_pred + self.lambda_prox * l_prox - self.lambda_div * l_div
                loss.backward()
                optimizer.step()

                if verbose and (it + 1) % 20 == 0:
                    print(f"  DICE iter {it+1}/{self.max_iterations}, loss={loss.item():.6f}")
        finally:
            for p in self.model.parameters():
                p.requires_grad_(True)
            self.model.train(was_training)

        # Evaluate each of the K counterfactuals against the REAL conformal
        # predictor (update_state=False: these are search candidates, not
        # genuine chronological observations - see TemporalConformalPredictor
        # docstring).
        results = []
        for cf in cfs:
            cf_np = cf.detach().cpu().numpy()
            r = self.conformal_predictor.predict_with_confidence(cf_np, self.model, update_state=False)
            changes = cf_np - x_anomaly
            threshold = 0.01 * np.std(x_anomaly)
            sparsity = float(np.mean(np.abs(changes) <= threshold))
            results.append({
                'counterfactual': cf_np,
                'changes': changes,
                'proximity': float(np.mean(changes ** 2)),
                'sparsity': sparsity,
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'validity': r['prediction'] == 'normal',
            })

        valid_results = [r for r in results if r['validity']]
        best = min(valid_results, key=lambda r: r['proximity']) if valid_results else min(results, key=lambda r: r['proximity'])

        return {
            'counterfactual': best['counterfactual'],
            'changes': best['changes'],
            'distance': best['proximity'],
            'proximity': best['proximity'],
            'prediction': best['prediction'],
            'confidence': best['confidence'],
            'validity': best['validity'],
            'is_valid': best['validity'],
            'sparsity': best['sparsity'],
            'all_counterfactuals': results,
        }
