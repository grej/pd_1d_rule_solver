"""
pd_1d_rule_solver/variable_selector.py

This module implements efficient information-theoretic approaches for identifying
promising variables to extend existing rules, with adaptive sampling for large datasets.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
from dataclasses import dataclass
from numba import njit


@dataclass
class VariableScore:
    """Container for variable scoring information"""
    name: str
    mi_score: float  # Mutual information score
    cmi_score: float  # Conditional mutual information given existing rule
    improvement_potential: float  # Estimated improvement potential
    boundary_sensitivity: float  # Sensitivity near rule boundaries
    sample_size: int  # Number of samples used in computation
    confidence: float  # Confidence in the score (based on sample stability)


@njit
def _fast_histogram2d(x: np.ndarray, y: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficient 2D histogram computation using Numba.
    Returns counts and bin edges for both dimensions.
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Add small epsilon to avoid edge cases
    x_max = x_max + (x_max - x_min) * 1e-10
    y_max = y_max + (y_max - y_min) * 1e-10

    counts = np.zeros((bins, bins), dtype=np.int64)
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)

    x_idx = np.floor((x - x_min) / (x_max - x_min) * bins).astype(np.int64)
    y_idx = np.floor((y - y_min) / (y_max - y_min) * bins).astype(np.int64)

    # Ensure indices are within bounds
    x_idx = np.clip(x_idx, 0, bins - 1)
    y_idx = np.clip(y_idx, 0, bins - 1)

    # Count occurrences
    for i in range(len(x)):
        counts[x_idx[i], y_idx[i]] += 1

    return counts, x_edges, y_edges


@njit
def _mutual_information_2d(joint_hist: np.ndarray) -> float:
    """Compute mutual information from a 2D histogram using Numba."""
    total = joint_hist.sum()
    if total == 0:
        return 0.0

    joint_prob = joint_hist / total
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)

    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))

    return max(0.0, mi)


class AdaptiveSampler:
    """Handles adaptive sampling for large datasets."""

    def __init__(self, df: pd.DataFrame, sample_threshold: int = 1000):
        """
        Initialize adaptive sampler.

        Args:
            df: Input DataFrame
            sample_threshold: Minimum number of rows before sampling is applied
        """
        self.df = df
        self.sample_threshold = sample_threshold
        self.current_sample = None
        self.sample_indices = None

    def get_sample(self,
                   target: str,
                   rule_mask: Optional[np.ndarray] = None,
                   initial_size: int = 1000,
                   max_size: int = 10000,
                   convergence_threshold: float = 0.01) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Get adaptive sample of the dataset.

        Args:
            target: Target variable name
            rule_mask: Boolean mask for current rule
            initial_size: Initial sample size
            max_size: Maximum sample size
            convergence_threshold: Threshold for sample convergence

        Returns:
            Tuple of (sampled DataFrame, adjusted rule mask)
        """
        if len(self.df) <= self.sample_threshold:
            return self.df, rule_mask

        if self.current_sample is not None:
            return self.current_sample, rule_mask[self.sample_indices] if rule_mask is not None else None

        # Initial random sample
        sample_indices = np.random.choice(len(self.df), size=initial_size, replace=False)
        current_sample = self.df.iloc[sample_indices].copy()

        if rule_mask is not None:
            current_mask = rule_mask[sample_indices]
        else:
            current_mask = None

        # Adaptive sampling
        while len(current_sample) < max_size:
            # Find regions of interest
            boundary_scores = self._compute_boundary_scores(
                current_sample, target, current_mask)

            # Check convergence
            if self._check_convergence(boundary_scores, convergence_threshold):
                break

            # Add samples in high-interest regions
            new_indices = self._sample_interesting_regions(
                boundary_scores,
                sample_indices,
                min(1000, max_size - len(current_sample))
            )

            if len(new_indices) == 0:
                break

            # Update sample
            sample_indices = np.concatenate([sample_indices, new_indices])
            current_sample = self.df.iloc[sample_indices].copy()
            if current_mask is not None:
                current_mask = rule_mask[sample_indices]

        self.current_sample = current_sample
        self.sample_indices = sample_indices

        return current_sample, current_mask

    def _compute_boundary_scores(self,
                               sample: pd.DataFrame,
                               target: str,
                               rule_mask: Optional[np.ndarray]) -> np.ndarray:
        """Compute scores indicating regions of interest for sampling."""
        # Get target gradients
        target_vals = sample[target].values
        gradients = np.abs(np.gradient(target_vals))

        # Add rule boundary information if available
        if rule_mask is not None:
            # Find points near rule boundaries
            rule_changes = np.abs(np.diff(rule_mask.astype(int)))
            rule_boundary = np.concatenate([[0], rule_changes])
            gradients += rule_boundary * np.std(gradients)

        return gradients

    def _check_convergence(self,
                          boundary_scores: np.ndarray,
                          threshold: float) -> bool:
        """Check if sampling has converged based on boundary scores."""
        # Simple convergence check based on ratio of high-score regions
        high_score_ratio = (boundary_scores > np.mean(boundary_scores)).mean()
        return high_score_ratio < threshold

    def _sample_interesting_regions(self,
                                  boundary_scores: np.ndarray,
                                  current_indices: np.ndarray,
                                  n_samples: int) -> np.ndarray:
        """Sample new points from interesting regions."""
        # Convert boundary scores to sampling probabilities
        unused_indices = np.setdiff1d(np.arange(len(self.df)), current_indices)
        if len(unused_indices) == 0:
            return np.array([], dtype=int)

        # Sample based on proximity to high-score regions
        probs = np.zeros(len(self.df))
        probs[unused_indices] = 1.0  # Base probability

        # Increase probability near interesting regions
        for idx in np.where(boundary_scores > np.mean(boundary_scores))[0]:
            current_idx = current_indices[idx]
            # Increase probability for nearby points
            vicinity = np.abs(unused_indices - current_idx) < 100
            probs[unused_indices[vicinity]] *= 2.0

        # Normalize probabilities
        probs /= probs.sum()

        # Sample new points
        return np.random.choice(
            len(self.df),
            size=min(n_samples, len(unused_indices)),
            p=probs,
            replace=False
        )


class VariableSelector:
    """Class for selecting promising variables to extend rules using information theory."""

    def __init__(self,
                 df: pd.DataFrame,
                 n_bins: int = 20,
                 boundary_width: float = 0.1,
                 sample_threshold: int = 1000):
        """
        Initialize the variable selector.

        Args:
            df: Input DataFrame
            n_bins: Number of bins for discretization
            boundary_width: Width of boundary region for sensitivity analysis
            sample_threshold: Minimum number of rows before sampling is applied
        """
        self.df = df
        self.n_bins = n_bins
        self.boundary_width = boundary_width
        self.sampler = AdaptiveSampler(df, sample_threshold)

    def score_variables(self,
                    target: str,
                    candidate_vars: List[str],
                    current_rule: Optional[Dict] = None) -> List[VariableScore]:
        """
        Score variables based on their potential for improving the current rule.
        """
        # Get rule mask if rule exists
        if current_rule:
            rule_mask = self._evaluate_rule(current_rule)
        else:
            rule_mask = np.ones(len(self.df), dtype=bool)

        # Get adaptive sample if needed
        sample_df, sample_mask = self.sampler.get_sample(target, rule_mask)

        # Convert target to numeric if categorical
        if isinstance(sample_df[target].dtype, pd.CategoricalDtype):
            target_vals = sample_df[target].cat.codes.values  # Convert to numpy array
        elif not np.issubdtype(sample_df[target].dtype, np.number):
            target_vals = pd.Categorical(sample_df[target]).codes
        else:
            target_vals = sample_df[target].values

        target_vals = target_vals.astype(np.float64)  # Ensure float64 type for numba

        scores = []
        for var in candidate_vars:
            if var in (current_rule or {}):
                continue

            # Convert variable to numeric
            if isinstance(sample_df[var].dtype, pd.CategoricalDtype):
                var_vals = sample_df[var].cat.codes.values
            elif not np.issubdtype(sample_df[var].dtype, np.number):
                var_vals = pd.Categorical(sample_df[var]).codes
            else:
                var_vals = sample_df[var].values

            var_vals = var_vals.astype(np.float64)  # Ensure float64 type for numba

            # Compute information metrics
            mi_score = self._compute_mi(var_vals, target_vals)
            cmi_score = self._compute_conditional_mi(var_vals, target_vals, sample_mask)

            # Compute boundary sensitivity
            boundary_sens = self._compute_boundary_sensitivity(
                var_vals, target_vals, sample_mask)

            # Estimate improvement potential
            improvement = self._estimate_improvement_potential(
                var_vals, target_vals, sample_mask)

            # Compute confidence based on sample size
            confidence = 1.0 - (1.0 / np.sqrt(len(sample_df)))

            scores.append(VariableScore(
                name=var,
                mi_score=mi_score,
                cmi_score=cmi_score,
                improvement_potential=improvement,
                boundary_sensitivity=boundary_sens,
                sample_size=len(sample_df),
                confidence=confidence
            ))

        return sorted(scores, key=lambda x: x.improvement_potential, reverse=True)

    def _compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two variables."""
        counts, _, _ = _fast_histogram2d(x, y, self.n_bins)
        return _mutual_information_2d(counts)

    def _compute_conditional_mi(self, x: np.ndarray, y: np.ndarray, condition: np.ndarray) -> float:
        """Compute mutual information between x and y conditional on existing rule."""
        if condition is None:
            return self._compute_mi(x, y)

        match_mi = self._compute_mi(x[condition], y[condition])
        non_match_mi = self._compute_mi(x[~condition], y[~condition])

        n_match = condition.sum()
        n_total = len(condition)
        if n_total == 0:
            return 0.0

        return (n_match * match_mi + (n_total - n_match) * non_match_mi) / n_total

    def _compute_boundary_sensitivity(self, x: np.ndarray, y: np.ndarray, rule_mask: np.ndarray) -> float:
        """Compute sensitivity of target to variable near rule boundaries."""
        if rule_mask is None or not rule_mask.any() or not (~rule_mask).any():
            return 0.0

        x_range = x.max() - x.min()
        if x_range == 0:  # Handle constant values
            return 0.0

        boundary_dist = x_range * self.boundary_width
        match_median = np.median(x[rule_mask])
        boundary_mask = np.abs(x - match_median) <= boundary_dist

        if not boundary_mask.any():
            return 0.0

        # Handle potential constant values
        x_boundary = x[boundary_mask]
        y_boundary = y[boundary_mask]

        if len(x_boundary) < 2 or len(np.unique(x_boundary)) < 2 or len(np.unique(y_boundary)) < 2:
            return 0.0

        try:
            boundary_corr = np.corrcoef(x_boundary, y_boundary)[0, 1]
            if np.isnan(boundary_corr):
                return 0.0
            return abs(boundary_corr)
        except:
            return 0.0


    def _estimate_improvement_potential(self, x: np.ndarray, y: np.ndarray, rule_mask: np.ndarray) -> float:
        """Estimate potential for improvement by combining multiple metrics."""
        try:
            mi_contrib = self._compute_mi(x, y)
            cmi_contrib = self._compute_conditional_mi(x, y, rule_mask)
            bound_contrib = self._compute_boundary_sensitivity(x, y, rule_mask)

            # Handle potential nan values
            mi_contrib = 0.0 if np.isnan(mi_contrib) else mi_contrib
            cmi_contrib = 0.0 if np.isnan(cmi_contrib) else cmi_contrib
            bound_contrib = 0.0 if np.isnan(bound_contrib) else bound_contrib

            return 0.4 * mi_contrib + 0.4 * cmi_contrib + 0.2 * bound_contrib
        except:
            return 0.0


    def _evaluate_rule(self, rule: Dict) -> np.ndarray:
        """Evaluate rule conditions on the dataset."""
        mask = np.ones(len(self.df), dtype=bool)
        for var, condition in rule.items():
            if isinstance(condition, tuple):
                mask &= (self.df[var] >= condition[0]) & (self.df[var] <= condition[1])
            else:
                mask &= (self.df[var] == condition)
        return mask


def suggest_next_variable(df: pd.DataFrame,
                         target: str,
                         current_rule: Optional[Dict] = None,
                         candidate_vars: Optional[List[str]] = None,
                         n_suggestions: int = 3,
                         sample_threshold: int = 1000) -> List[VariableScore]:
    """
    Convenience function to suggest next variables to consider for rule extension.

    Args:
        df: Input DataFrame
        target: Target variable name
        current_rule: Current rule conditions (optional)
        candidate_vars: List of variables to consider (if None, uses all numeric columns)
        n_suggestions: Number of variables to suggest
        sample_threshold: Minimum number of rows before sampling is applied

    Returns:
        List of top n_suggestions VariableScore objects
    """
    if candidate_vars is None:
        candidate_vars = [col for col in df.columns
                        if col != target and np.issubdtype(df[col].dtype, np.number)]

    selector = VariableSelector(df, sample_threshold=sample_threshold)
    scores = selector.score_variables(target, candidate_vars, current_rule)

    return scores[:n_suggestions]


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Find initial rule
    from pd_1d_rule_solver import RuleFinder
    result = df.findrule(
        target='sepal length (cm)',
        direction='maximize',
        variables=['petal length (cm)']
    )

    # Get suggestions for next variable
    suggestions = suggest_next_variable(
        df,
        target='sepal length (cm)',
        current_rule=result['rule'],
        n_suggestions=3
    )

    # Print suggestions with scores
    print("\nTop variable suggestions:")
    for var in suggestions:
        print(f"\nVariable: {var.name}")
        print(f"Improvement potential: {var.improvement_potential:.3f}")
        print(f"Mutual information: {var.mi_score:.3f}")
        print(f"Conditional MI: {var.cmi_score:.3f}")
        print(f"Boundary sensitivity: {var.boundary_sensitivity:.3f}")
        print(f"Sample size: {var.sample_size}")
        print(f"Confidence: {var.confidence:.3f}")
