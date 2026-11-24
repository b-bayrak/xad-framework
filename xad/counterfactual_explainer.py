import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Union, Optional, List
from deap import base, creator, tools
from dataclasses import dataclass


@dataclass
class CounterfactualResult:
    """Container for counterfactual explanation results."""
    counterfactual: np.ndarray
    changes: np.ndarray
    distance: float
    prediction: str
    confidence: float
    validity: bool
    sparsity: float
    feature_changes: Dict[int, float]


class CounterfactualExplainer:
    """
    GA-CE: Genetic Algorithm-based Counterfactual Explanations.

    Implements the objective and procedure described in Section 3.5.1 /
    Algorithm 1 of the paper:

        L_C(x')     = lambda1 * L_val + lambda2 * L_prox + lambda3 * L_spar     (Eq. 17)
        L_val_cont  = 0 if s(x') < q_hat_alpha else (s(x')-q_hat_alpha)/q_hat_alpha  (Eq. 18)
        x'_0        = alpha*x + (1-alpha)*x_nn, alpha ~ U(0.1, 0.7)              (Eq. 19)
        L_smooth    = sum_t || x'_{t+1}-x'_t - (x_{t+1}-x_t) ||_2^2              (Eq. 22)
        L_total     = L_C + lambda_smooth * L_smooth                            (Eq. 23)

    `use_proximity`, `use_sparsity`, and `use_smooth` let a caller ablate
    individual loss terms (dropping lambda2, lambda3, lambda_smooth to 0
    respectively) while keeping every other part of the pipeline identical -
    the GA-CE loss-term ablation (Experiment 5) uses this, so its numbers
    are directly comparable to the main counterfactual run.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        conformal_predictor,
        n_neighbors: int = 10,
        n_generations: int = 100,
        population_size: int = 50,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.25,
        elitism: int = 5,
        tournament_size: int = 3,
        selection_fraction: float = 0.3,
        lambda_val: float = 1.0,
        lambda_prox: float = 1.0,
        lambda_spar: float = 1.0,
        lambda_smooth: float = 0.1,
        sparsity_threshold_ratio: float = 0.01,
        mutation_scale: float = 0.25,
        patience: int = 30,
        early_stop_fitness: float = 0.5,
        use_proximity: bool = True,
        use_sparsity: bool = True,
        use_smooth: bool = True,
    ):
        self.model = model
        self.conformal_predictor = conformal_predictor
        self.n_neighbors = n_neighbors
        self.n_generations = n_generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.selection_fraction = selection_fraction

        self.lambda_val = lambda_val
        self.lambda_prox = lambda_prox if use_proximity else 0.0
        self.lambda_spar = lambda_spar if use_sparsity else 0.0
        self.lambda_smooth = lambda_smooth if use_smooth else 0.0
        self.sparsity_threshold_ratio = sparsity_threshold_ratio
        self.mutation_scale = mutation_scale
        self.patience = patience
        self.early_stop_fitness = early_stop_fitness

        self.use_proximity = use_proximity
        self.use_sparsity = use_sparsity
        self.use_smooth = use_smooth

        self.normal_data: Optional[np.ndarray] = None
        self.normal_data_flat: Optional[np.ndarray] = None
        self.nn_model: Optional[NearestNeighbors] = None

    def fit(self, X_normal: Union[np.ndarray, torch.Tensor]) -> 'CounterfactualExplainer':
        """Fit the explainer with normal data used for k-NN initialization (Eq. 19)."""
        if isinstance(X_normal, torch.Tensor):
            self.normal_data = X_normal.detach().cpu().numpy()
        else:
            self.normal_data = np.array(X_normal)

        self.normal_data_flat = self.normal_data.reshape(len(self.normal_data), -1)

        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.normal_data))
        )
        self.nn_model.fit(self.normal_data_flat)

        return self

    def _setup_ga(self):
        """Set up DEAP genetic algorithm components."""
        if "FitnessMin" in creator.__dict__:
            del creator.FitnessMin
        if "Individual" in creator.__dict__:
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    def _anomaly_score(self, individual: np.ndarray) -> float:
        # update_state=False: candidate solutions explored during GA search
        # are synthetic, not genuine chronological observations, and must
        # never be folded into the conformal predictor's rolling calibration
        # window or overwrite its live threshold (see
        # TemporalConformalPredictor.predict_with_confidence docstring).
        result = self.conformal_predictor.predict_with_confidence(
            individual, self.model, update_state=False
        )
        return result['score']

    def _validity_loss(self, x_prime: np.ndarray) -> tuple:
        """Continuous validity loss (Eq. 18). Returns (loss, is_valid, score)
        so callers don't need a second detector forward pass to know validity."""
        s = self._anomaly_score(x_prime)
        q_hat = getattr(self.conformal_predictor, 'threshold', None)
        if q_hat is None or q_hat == 0:
            q_hat = 1e-6
        is_valid = s < q_hat
        loss = 0.0 if is_valid else (s - q_hat) / max(q_hat, 1e-6)
        return loss, is_valid, s

    def _smoothness_loss(self, x_prime: np.ndarray, x_anomaly: np.ndarray) -> float:
        """L_smooth (Eq. 22): squared temporal-dynamics mismatch."""
        if x_prime.shape[0] < 2:
            return 0.0
        true_diff = x_anomaly[1:] - x_anomaly[:-1]
        cf_diff = x_prime[1:] - x_prime[:-1]
        return float(np.sum((cf_diff - true_diff) ** 2))

    def _fitness(self, x_prime: np.ndarray, x_anomaly: np.ndarray) -> tuple:
        """L_total = L_C + lambda_smooth * L_smooth (Eq. 17 + Eq. 23).
        Returns (fitness, is_valid) - is_valid is a byproduct of the single
        validity-loss scoring call, so the caller never needs a second
        forward pass just to check validity."""
        l_val, is_valid, _ = self._validity_loss(x_prime)
        l_prox = float(np.linalg.norm((x_prime - x_anomaly).reshape(-1), ord=2)) if self.use_proximity else 0.0

        if self.use_sparsity:
            threshold = self.sparsity_threshold_ratio * np.std(x_anomaly)
            l_spar = float(np.mean(np.abs(x_prime - x_anomaly) > threshold))
        else:
            l_spar = 0.0

        l_smooth = self._smoothness_loss(x_prime, x_anomaly) if self.use_smooth else 0.0

        l_c = self.lambda_val * l_val + self.lambda_prox * l_prox + self.lambda_spar * l_spar
        return l_c + self.lambda_smooth * l_smooth, is_valid

    def generate_counterfactual(
        self,
        x_anomaly: Union[np.ndarray, torch.Tensor],
        verbose: bool = False
    ) -> Dict:
        """
        Run GA-CE (Algorithm 1) to find a counterfactual x' for an anomaly x.
        """
        if self.normal_data is None:
            raise ValueError("Must call fit() first with normal data")

        if isinstance(x_anomaly, torch.Tensor):
            x_anomaly = x_anomaly.detach().cpu().numpy()
        else:
            x_anomaly = np.array(x_anomaly)

        if x_anomaly.ndim == 3:
            x_anomaly = x_anomaly.squeeze(0)

        original_shape = x_anomaly.shape
        x_flat = x_anomaly.reshape(-1)

        self._setup_ga()
        toolbox = base.Toolbox()

        distances, indices = self.nn_model.kneighbors(x_flat.reshape(1, -1))
        neighbor_pool = indices[0]

        def create_individual():
            neighbor_idx = neighbor_pool[np.random.randint(0, len(neighbor_pool))]
            neighbor = self.normal_data_flat[neighbor_idx].reshape(original_shape)
            alpha = np.random.uniform(0.1, 0.7)  # Eq. 19
            individual = alpha * x_anomaly + (1 - alpha) * neighbor
            return creator.Individual(individual)

        def evaluate(individual):
            fitness, is_valid = self._fitness(np.asarray(individual), x_anomaly)
            individual.is_valid = is_valid  # cached so the generation loop doesn't re-score
            return (fitness,)

        def mutate(individual):
            noise_scale = self.mutation_scale * np.std(x_anomaly)
            noise = np.random.normal(0, noise_scale, individual.shape)
            individual += noise
            return (individual,)

        def crossover(ind1, ind2):
            beta = np.random.uniform(0.3, 0.7)
            new_ind1 = beta * ind1 + (1 - beta) * ind2
            new_ind2 = beta * ind2 + (1 - beta) * ind1
            ind1[:] = new_ind1
            ind2[:] = new_ind2
            return ind1, ind2

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        population = toolbox.population(n=self.population_size)
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        best_fitness_history = [min(ind.fitness.values[0] for ind in population)]
        gens_without_improvement = 0
        pool_size = max(2, int(round(self.selection_fraction * self.population_size)))
        n_offspring_needed = max(0, self.population_size - self.elitism)

        for gen in range(self.n_generations):
            best_fitness = min(ind.fitness.values[0] for ind in population)
            best_valid = min(
                (ind.fitness.values[0] for ind in population if getattr(ind, 'is_valid', False)),
                default=None
            )

            if verbose and (gen + 1) % 20 == 0:
                print(f"  Generation {gen + 1}/{self.n_generations}, Best fitness: {best_fitness:.6f}")

            if gen > 0:
                if best_fitness < best_fitness_history[-1] - 1e-9:
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1
            best_fitness_history.append(best_fitness)

            if gens_without_improvement >= self.patience or (best_valid is not None and best_valid < self.early_stop_fitness):
                break

            # Elitism: preserve top 5 (Algorithm 1, line 14)
            elites = [toolbox.clone(ind) for ind in tools.selBest(population, self.elitism)]

            # Tournament-select the top ~30% mating pool (Algorithm 1, line 15)
            mating_pool = [toolbox.clone(ind) for ind in toolbox.select(population, pool_size)]

            offspring = []
            while len(offspring) < n_offspring_needed:
                a, b = mating_pool[np.random.randint(len(mating_pool))], mating_pool[np.random.randint(len(mating_pool))]
                child1, child2 = toolbox.clone(a), toolbox.clone(b)
                if np.random.random() < self.crossover_rate:
                    toolbox.mate(child1, child2)
                if np.random.random() < self.mutation_rate:
                    toolbox.mutate(child1)
                if np.random.random() < self.mutation_rate:
                    toolbox.mutate(child2)
                offspring.extend([child1, child2])

            offspring = offspring[:max(n_offspring_needed, 0)] if n_offspring_needed > 0 else []
            for ind in offspring:
                ind.fitness.values = toolbox.evaluate(ind)

            offspring_kept = tools.selBest(offspring, n_offspring_needed) if offspring else []
            population = elites + offspring_kept

        best_ind = tools.selBest(population, 1)[0]

        # Final report on the chosen counterfactual - still read-only: this
        # is the returned explanation, not a real chronological observation.
        result = self.conformal_predictor.predict_with_confidence(
            best_ind, self.model, update_state=False
        )

        changes = np.asarray(best_ind) - x_anomaly
        change_magnitudes = np.abs(changes)

        feature_changes = {}
        mean_changes = np.mean(change_magnitudes, axis=0)
        threshold = self.sparsity_threshold_ratio * np.std(x_anomaly)
        for i, change in enumerate(mean_changes):
            if change > threshold:
                feature_changes[i] = float(change)

        sparsity = float(np.mean(change_magnitudes <= threshold))
        proximity = float(np.linalg.norm(changes.reshape(-1), ord=2))

        return {
            'counterfactual': np.asarray(best_ind),
            'changes': changes,
            'distance': proximity,
            'proximity': proximity,
            'l2_distance': proximity,
            'l1_distance': float(np.mean(np.abs(changes))),
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'validity': result['prediction'] == 'normal',
            'is_valid': result['prediction'] == 'normal',
            'sparsity': sparsity,
            'feature_changes': feature_changes,
            'top_changed_features': sorted(
                feature_changes.keys(),
                key=lambda k: feature_changes[k],
                reverse=True
            )[:5] if feature_changes else [],
            'fitness_history': best_fitness_history,
            'final_fitness': float(best_ind.fitness.values[0])
        }

    def generate_multiple_counterfactuals(
        self,
        x_anomaly: Union[np.ndarray, torch.Tensor],
        n_counterfactuals: int = 5,
    ) -> List[Dict]:
        """Generate multiple counterfactuals by re-running GA-CE with small input perturbations for diversity."""
        counterfactuals = []

        for i in range(n_counterfactuals):
            if isinstance(x_anomaly, torch.Tensor):
                x = x_anomaly.detach().cpu().numpy()
            else:
                x = np.array(x_anomaly)

            if i > 0:
                perturbation = np.random.normal(0, 0.01 * np.std(x), x.shape)
                x = x + perturbation

            cf = self.generate_counterfactual(x)
            counterfactuals.append(cf)

        return counterfactuals

    def explain_changes(
        self,
        x_anomaly: Union[np.ndarray, torch.Tensor],
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Human-readable explanation of required changes."""
        cf_result = self.generate_counterfactual(x_anomaly)

        if isinstance(x_anomaly, torch.Tensor):
            x_anomaly = x_anomaly.detach().cpu().numpy()
        if x_anomaly.ndim == 3:
            x_anomaly = x_anomaly.squeeze(0)

        explanations = []
        changes = cf_result['changes']

        mean_changes = np.mean(changes, axis=0)

        for i, change in enumerate(mean_changes):
            if abs(change) > 0.01 * np.std(x_anomaly):
                feature_name = feature_names[i] if feature_names else f"Feature {i}"
                direction = "increase" if change > 0 else "decrease"
                magnitude = abs(change)

                explanations.append({
                    'feature': feature_name,
                    'feature_index': i,
                    'direction': direction,
                    'magnitude': float(magnitude),
                    'original_mean': float(np.mean(x_anomaly[:, i])),
                    'suggested_mean': float(np.mean(cf_result['counterfactual'][:, i]))
                })

        explanations.sort(key=lambda x: x['magnitude'], reverse=True)

        return {
            'counterfactual_result': cf_result,
            'explanations': explanations,
            'summary': self._generate_summary(explanations),
            'n_features_changed': len(explanations)
        }

    def _generate_summary(self, explanations: List[Dict]) -> str:
        """Generate human-readable summary of changes."""
        if not explanations:
            return "No significant changes required."

        summary_parts = []
        for exp in explanations[:3]:
            summary_parts.append(
                f"{exp['direction']} {exp['feature']} by {exp['magnitude']:.4f}"
            )

        summary = "To transform this anomaly to normal: " + "; ".join(summary_parts)

        if len(explanations) > 3:
            summary += f" (and {len(explanations) - 3} more changes)"

        return summary
