"""
Uncertainty Types:
1. Epistemic Uncertainty: Model's lack of knowledge (can be reduced with more data)
2. Aleatoric Uncertainty: Inherent data noise (cannot be reduced)
3. Temporal Uncertainty: Unusual temporal patterns
4. Feature Uncertainty: Which features contribute to uncertainty

NOTE:
    This file is for demonstration purposes only and is not yet complete.
    It uses simplified, fixed thresholds and a different scoring function than
    the paper to better illustrate the core concepts for future work (e.g., user studies).
    For the full implementation matching the paper (with adaptive thresholds),
    see the experiments folder.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class UncertaintyExplanation:
    total_uncertainty: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    temporal_uncertainty: float
    
    uncertainty_sources: List[Dict[str, Any]]
    feature_uncertainties: np.ndarray
    top_uncertain_features: List[int]
    
    # Human-readable explanation
    summary: str
    detailed_reasons: List[str]
    recommendations: List[str]
    
    # Reliability
    prediction_reliability: str  # h/m/l
    should_flag_for_review: bool


class UncertaintyExplainer:
    """
    Explains WHY the model is uncertain about predictions.
    
    This goes beyond just reporting an uncertainty score - it decomposes
    the uncertainty into interpretable components and provides actionable
    explanations.
    
    Key insight from literature:
    "Decision-makers need not only the prediction and confidence, but also
    the REASONS for varying levels of confidence to improve trust in AI systems"
    
    Components:
    1. Epistemic Uncertainty: Is this sample similar to training data?
    2. Aleatoric Uncertainty: Is this region inherently noisy?
    3. Temporal Uncertainty: Are there unusual temporal patterns?
    4. Feature Uncertainty: Which features are most uncertain?
    """
    
    def __init__(
        self,
        training_data: np.ndarray,
        n_neighbors: int = 20,
        feature_names: Optional[List[str]] = None
    ):
        if isinstance(training_data, torch.Tensor):
            training_data = training_data.detach().cpu().numpy()
        
        self.training_data = training_data
        self.n_neighbors = min(n_neighbors, len(training_data) - 1)
        self.feature_names = feature_names  # Optional names for features
        

        self.training_flat = training_data.reshape(len(training_data), -1) # Flatten
        
        # Initialize nearest neighbor model
        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors, 
            algorithm='ball_tree'
        )
        self.nn_model.fit(self.training_flat)
        
        # Initialize Local Outlier Factor for density estimation
        self.lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        self.lof.fit(self.training_flat)
        
        # Compute training statistics (for comparison)
        self._compute_training_statistics()
    
    def _compute_training_statistics(self, n_samples: int = 1000):
        # Sample if dataset is large
        if len(self.training_data) > n_samples:
            sample_indices = np.random.choice(len(self.training_data), n_samples, replace=False)
        else:
            sample_indices = np.arange(len(self.training_data))
        
        # Distance distribution in training data
        distances = []
        for i in sample_indices:
            dist, _ = self.nn_model.kneighbors(self.training_flat[i:i+1])
            distances.append(np.mean(dist))
        self.training_distances = np.array(distances)
        self.distance_percentiles = {
            50: np.percentile(self.training_distances, 50),
            75: np.percentile(self.training_distances, 75),
            90: np.percentile(self.training_distances, 90),
            95: np.percentile(self.training_distances, 95),
        }
        
        # Temp var distribution
        temporal_vars = []
        for i in sample_indices:
            sample = self.training_data[i]
            if len(sample.shape) > 1 and sample.shape[0] > 1:
                # Variance of first differences
                temporal_vars.append(np.mean(np.var(np.diff(sample, axis=0), axis=0)))
        self.temporal_variances = np.array(temporal_vars) if temporal_vars else np.array([0])
        self.temporal_percentiles = {
            50: np.percentile(self.temporal_variances, 50),
            75: np.percentile(self.temporal_variances, 75),
            90: np.percentile(self.temporal_variances, 90),
        }
        
        # Feature-level statistics
        self.feature_means = np.mean(self.training_data, axis=(0, 1))
        self.feature_stds = np.std(self.training_data, axis=(0, 1))
        self.feature_ranges = (
            np.min(self.training_data, axis=(0, 1)),
            np.max(self.training_data, axis=(0, 1))
        )


    def explain_uncertainty(
        self,
        x: np.ndarray,
        anomaly_score: float,
        confidence: float,
        uncertainty: float,
        reconstruction_errors: Optional[np.ndarray] = None
    ) -> UncertaintyExplanation:

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        
        epistemic = self._compute_epistemic_uncertainty(x)
        aleatoric = self._compute_aleatoric_uncertainty(x)
        temporal = self._compute_temporal_uncertainty(x)
        feature_unc = self._compute_feature_uncertainty(x, reconstruction_errors)
        
        # total = 0.4 * epistemic['score'] + 0.3 * aleatoric['score'] + 0.3 * temporal['score']
        total = (epistemic['score'] + aleatoric['score'] + temporal['score']) / 3.0

        # Collect all uncertainty sources
        sources = []
        detailed_reasons = []
        recommendations = []
        
        # Epistemic uncertainty analysis
        if epistemic['score'] > 0.5:
            sources.append({
                'type': 'epistemic',
                'score': epistemic['score'],
                'description': 'Pattern not well represented in training data'
            })
            detailed_reasons.append(
                f"This sample is {epistemic['description']} from the training distribution "
                f"(distance percentile: {epistemic['percentile']:.0f}%)"
            )
            recommendations.append(
                "Consider collecting more training data for similar patterns"
            )
        
        # Aleatoric uncertainty analysis
        if aleatoric['score'] > 0.5:
            sources.append({
                'type': 'aleatoric',
                'score': aleatoric['score'],
                'description': 'Sample is in a sparse/noisy region'
            })
            detailed_reasons.append(
                f"This sample is in a {aleatoric['density_description']} density region "
                f"(LOF score: {aleatoric['lof_score']:.2f})"
            )
            recommendations.append(
                "This region has inherent variability - consider manual review"
            )
        
        # Temporal uncertainty analysis
        if temporal['score'] > 0.5:
            sources.append({
                'type': 'temporal',
                'score': temporal['score'],
                'description': 'Unusual temporal dynamics'
            })
            detailed_reasons.append(
                f"Temporal pattern is {temporal['description']} compared to training "
                f"(variance percentile: {temporal['percentile']:.0f}%)"
            )
            recommendations.append(
                "Check for sensor issues or process changes causing temporal anomalies"
            )
        
        # Feature-specific uncertainty
        top_uncertain = np.argsort(feature_unc)[::-1][:5]
        feature_sources = []
        for idx in top_uncertain:
            if feature_unc[idx] > 0.3:
                feat_name = self.feature_names[idx] if self.feature_names else f"Feature {idx}"
                feature_sources.append({
                    'feature_idx': int(idx),
                    'feature_name': feat_name,
                    'uncertainty': float(feature_unc[idx])
                })
                if len(feature_sources) <= 3:
                    detailed_reasons.append(
                        f"{feat_name} shows unusual values (uncertainty: {feature_unc[idx]:.2f})"
                    )
        
        if feature_sources:
            sources.append({
                'type': 'feature',
                'score': np.max(feature_unc),
                'description': 'Specific features have unusual values',
                'features': feature_sources
            })
        
        # Determine reliability
        if total < 0.3 and confidence > 0.8:
            reliability = 'high'
            flag_for_review = False
        elif total < 0.6 or confidence > 0.7:
            reliability = 'medium'
            flag_for_review = uncertainty > 0.5
        else:
            reliability = 'low'
            flag_for_review = True
        
        # Generate summary
        summary = self._generate_summary(
            total, epistemic['score'], aleatoric['score'], 
            temporal['score'], reliability, anomaly_score
        )
        
        # Add general recommendations based on reliability
        if reliability == 'low':
            recommendations.append(
                "Low reliability prediction. Manual verification strongly recommended"
            )
        elif reliability == 'medium':
            recommendations.append(
                "Consider cross-checking with domain expert for borderline cases"
            )
        
        return UncertaintyExplanation(
            total_uncertainty=float(total),
            epistemic_uncertainty=float(epistemic['score']),
            aleatoric_uncertainty=float(aleatoric['score']),
            temporal_uncertainty=float(temporal['score']),
            uncertainty_sources=sources,
            feature_uncertainties=feature_unc,
            top_uncertain_features=[int(i) for i in top_uncertain],
            summary=summary,
            detailed_reasons=detailed_reasons,
            recommendations=recommendations,
            prediction_reliability=reliability,
            should_flag_for_review=flag_for_review
        )
    
    def _compute_epistemic_uncertainty(self, x: np.ndarray) -> Dict:
        """
        Compute epistemic uncertainty (model's lack of knowledge).
        
        High epistemic uncertainty means:
        - Sample is far from training data
        - Model hasn't seen similar patterns
        - Can potentially be reduced with more data
        """
        x_flat = x.reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(x_flat)
        avg_distance = np.mean(distances)
        
        # Compare to training distribution
        percentile = np.mean(avg_distance > self.training_distances) * 100
        
        # Normalize to [0, 1]
        if percentile >= 95:
            score = 1.0
            description = "very far"
        elif percentile >= 90:
            score = 0.8
            description = "far"
        elif percentile >= 75:
            score = 0.5
            description = "moderately far"
        elif percentile >= 50:
            score = 0.3
            description = "slightly far"
        else:
            score = 0.1
            description = "close"
        
        return {
            'score': score,
            'distance': float(avg_distance),
            'percentile': percentile,
            'description': description,
            'nearest_indices': indices[0].tolist()
        }
    
    def _compute_aleatoric_uncertainty(self, x: np.ndarray) -> Dict:
        """
        Compute aleatoric uncertainty (inherent data noise).
        
        High aleatoric uncertainty means:
        - Sample is in a sparse/variable region
        - Even with more data, predictions would be uncertain
        - Inherent to the data, not the model
        """
        x_flat = x.reshape(1, -1)
        
        # Use LOF score as proxy for density
        try:
            lof_score = -self.lof.score_samples(x_flat)[0]
        except:
            lof_score = 1.0
        
        # LOF > 1 indicates outlier, normalize
        if lof_score > 2.0:
            score = 1.0
            density_description = "very low"
        elif lof_score > 1.5:
            score = 0.7
            density_description = "low"
        elif lof_score > 1.0:
            score = 0.4
            density_description = "moderate"
        else:
            score = 0.1
            density_description = "high"
        
        return {
            'score': score,
            'lof_score': float(lof_score),
            'density_description': density_description
        }
    
    def _compute_temporal_uncertainty(self, x: np.ndarray) -> Dict:
        """
        Compute temporal uncertainty (unusual time dynamics).
        
        High temporal uncertainty means:
        - Unusual patterns over time
        - Sudden changes or unexpected smoothness
        - May indicate sensor issues
        """
        if x.shape[0] < 2:
            return {'score': 0.0, 'variance': 0.0, 'percentile': 0.0, 'description': 'N/A'}
        
        # Compute temporal variance
        temporal_var = np.mean(np.var(np.diff(x, axis=0), axis=0))
        
        # Compare to training distribution
        percentile = np.mean(temporal_var > self.temporal_variances) * 100
        
        if percentile >= 90:
            score = 1.0
            description = "highly variable"
        elif percentile >= 75:
            score = 0.6
            description = "variable"
        elif percentile >= 50:
            score = 0.3
            description = "moderately variable"
        elif percentile <= 10:
            # Very low variance can also be unusual
            score = 0.4
            description = "unusually smooth"
        else:
            score = 0.1
            description = "typical"
        
        return {
            'score': score,
            'variance': float(temporal_var),
            'percentile': percentile,
            'description': description
        }
    
    def _compute_feature_uncertainty(self, x: np.ndarray, reconstruction_errors = None) -> np.ndarray:
        """
        Compute per-feature uncertainty.
        
        Identifies which features contribute most to overall uncertainty.
        """
        n_features = x.shape[-1]
        feature_unc = np.zeros(n_features)
        
        for i in range(n_features):
            feature_values = x[:, i]
            
            # Check if values are outside typical range
            feat_min, feat_max = self.feature_ranges[0][i], self.feature_ranges[1][i]
            range_size = feat_max - feat_min
            if range_size > 0:
                out_of_range = (
                    np.mean(feature_values < feat_min - 0.1 * range_size) +
                    np.mean(feature_values > feat_max + 0.1 * range_size)
                )
            else:
                out_of_range = 0
            
            # Check deviation from mean
            if self.feature_stds[i] > 0:
                z_scores = np.abs((feature_values - self.feature_means[i]) / self.feature_stds[i])
                deviation = np.mean(z_scores > 2)  # Proportion > 2 std
            else:
                deviation = 0
            
            # If reconstruction errors available, use them
            if reconstruction_errors is not None:
                if reconstruction_errors.ndim > 1:
                    recon_unc = np.mean(reconstruction_errors[:, i]) / (
                        np.mean(reconstruction_errors) + 1e-8
                    )
                else:
                    recon_unc = reconstruction_errors[i] / (
                        np.mean(reconstruction_errors) + 1e-8
                    )
            else:
                recon_unc = 0
            
            # Combine
            feature_unc[i] = 0.3 * out_of_range + 0.3 * deviation + 0.4 * min(1.0, recon_unc)
        
        return feature_unc
    
    def _generate_summary(
        self,
        total: float,
        epistemic: float,
        aleatoric: float,
        temporal: float,
        reliability: str,
        anomaly_score: float
    ) -> str:
        """Generate human-readable summary"""
        
        # Determine dominant source
        sources = {'epistemic': epistemic, 'aleatoric': aleatoric, 'temporal': temporal}
        dominant = max(sources.items(), key=lambda x: x[1])
        
        if total < 0.3:
            uncertainty_level = "LOW"
            confidence_statement = "The model is confident in this prediction."
        elif total < 0.6:
            uncertainty_level = "MODERATE"
            confidence_statement = "The model has moderate confidence in this prediction."
        else:
            uncertainty_level = "HIGH"
            confidence_statement = "The model is uncertain about this prediction."
        
        summary = f"UNCERTAINTY ANALYSIS\n"
        summary += f"{'-'*50}\n\n"
        summary += f"Overall Uncertainty: {uncertainty_level} ({total:.2f})\n"
        summary += f"Prediction Reliability: {reliability.upper()}\n\n"
        summary += f"{confidence_statement}\n\n"
        
        summary += f"Uncertainty Breakdown:\n"
        summary += f"  • Epistemic (lack of similar data):  {epistemic:.2f}\n"
        summary += f"  • Aleatoric (inherent data noise):   {aleatoric:.2f}\n"
        summary += f"  • Temporal (unusual time patterns):  {temporal:.2f}\n\n"
        
        if dominant[1] > 0.4:
            summary += f"Primary uncertainty source: {dominant[0].upper()}\n"
            
            if dominant[0] == 'epistemic':
                summary += "→ This pattern differs from training data\n"
            elif dominant[0] == 'aleatoric':
                summary += "→ This region has inherent variability\n"
            else:
                summary += "→ Unusual temporal dynamics detected\n"
        
        return summary
    
    def get_explanation_dict(self, explanation: UncertaintyExplanation) -> Dict:
        """Convert UncertaintyExplanation to dictionary for JSON serialization"""
        return {
            'total_uncertainty': explanation.total_uncertainty,
            'epistemic_uncertainty': explanation.epistemic_uncertainty,
            'aleatoric_uncertainty': explanation.aleatoric_uncertainty,
            'temporal_uncertainty': explanation.temporal_uncertainty,
            'uncertainty_sources': explanation.uncertainty_sources,
            'feature_uncertainties': explanation.feature_uncertainties.tolist(),
            'top_uncertain_features': explanation.top_uncertain_features,
            'summary': explanation.summary,
            'detailed_reasons': explanation.detailed_reasons,
            'recommendations': explanation.recommendations,
            'prediction_reliability': explanation.prediction_reliability,
            'should_flag_for_review': explanation.should_flag_for_review
        }


class UncertaintyVisualizer:
    """Visualization tools for uncertainty explanations"""
    
    @staticmethod
    def plot_uncertainty_breakdown(
        explanation: UncertaintyExplanation,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Create visual breakdown of uncertainty components.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Uncertainty pie chart
        ax1 = axes[0, 0]
        sizes = [
            explanation.epistemic_uncertainty,
            explanation.aleatoric_uncertainty,
            explanation.temporal_uncertainty
        ]
        labels = ['Epistemic\n(Model knowledge)', 'Aleatoric\n(Data noise)', 'Temporal\n(Time patterns)']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        # Only show non-zero components
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0.05]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Uncertainty Composition', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Low Uncertainty', ha='center', va='center', fontsize=14)
            ax1.axis('off')
        
        # Plot 2: Bar chart of components
        ax2 = axes[0, 1]
        components = ['Epistemic', 'Aleatoric', 'Temporal', 'Total']
        values = [
            explanation.epistemic_uncertainty,
            explanation.aleatoric_uncertainty,
            explanation.temporal_uncertainty,
            explanation.total_uncertainty
        ]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        bars = ax2.barh(components, values, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Uncertainty Score')
        ax2.set_title('Uncertainty by Component', fontsize=12, fontweight='bold')
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{val:.2f}', va='center', fontsize=10)
        
        # Plot 3: Feature uncertainty
        ax3 = axes[1, 0]
        n_features = len(explanation.feature_uncertainties)
        top_n = min(10, n_features)
        top_indices = np.argsort(explanation.feature_uncertainties)[::-1][:top_n]
        
        feature_names = [f'Feature {i}' for i in top_indices]
        feature_values = explanation.feature_uncertainties[top_indices]
        
        ax3.barh(range(top_n), feature_values, color='#f39c12')
        ax3.set_yticks(range(top_n))
        ax3.set_yticklabels(feature_names)
        ax3.set_xlabel('Feature Uncertainty')
        ax3.set_title('Top Uncertain Features', fontsize=12, fontweight='bold')
        ax3.invert_yaxis()
        
        # Plot 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary box
        reliability_color = {
            'high': 'green',
            'medium': 'orange', 
            'low': 'red'
        }[explanation.prediction_reliability]
        
        summary_text = f"""
PREDICTION RELIABILITY: {explanation.prediction_reliability.upper()}

{'FLAG FOR REVIEW' if explanation.should_flag_for_review else 'OK TO PROCEED'}

Top Reasons for Uncertainty:
"""
        for i, reason in enumerate(explanation.detailed_reasons[:3], 1):
            summary_text += f"\n{i}. {reason[:60]}..."
        
        summary_text += f"\n\nRecommendations:"
        for rec in explanation.recommendations[:2]:
            summary_text += f"\n• {rec[:50]}..."
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
