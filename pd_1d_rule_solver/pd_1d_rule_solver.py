import pandas as pd
import numpy as np
import pandas.api.types
from typing import List, Union, Dict, Tuple, Optional
import warnings
from numba import njit  # Import Numba's JIT decorator

@njit
def find_best_interval_kadane(outcomes: np.ndarray) -> Tuple[float, int, int]:
    """
    Find the interval with the maximum sum using Kadane's algorithm.

    Args:
        outcomes: 1D numpy array of numerical outcomes.

    Returns:
        A tuple containing the maximum sum, start index, and end index of the best interval.
    """
    max_sum = -np.inf
    current_sum = 0.0
    start = 0
    best_start = 0
    best_end = 0

    for i in range(len(outcomes)):
        if current_sum <= 0.0:
            start = i
            current_sum = outcomes[i]
        else:
            current_sum += outcomes[i]

        if current_sum > max_sum:
            max_sum = current_sum
            best_start = start
            best_end = i

    return max_sum, best_start, best_end

@pd.api.extensions.register_dataframe_accessor("findrule")
class RuleFinder:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        # Add a dictionary to cache sorted DataFrames for each numeric feature
        self._sorted_dfs = {}


    def __call__(self, target: str, direction: str, variables: Optional[List[str]] = None,
                 bins: Optional[int] = None, visualize: bool = False, depth: int = 1,
                 min_improvement: float = 0.05, reopt_iterations: int = 2) -> Dict:
        """
        Find a rule that shifts the distribution of the target variable in the desired direction.

        Args:
            target: Target variable name
            direction: Either 'maximize'/'minimize' for numeric targets, or category name for categorical
            variables: List of feature names to consider for the rule. If None, uses all numeric columns
            bins: Number of bins for histogram visualization (default: 12 or number of unique values if less)
            visualize: Whether to include visualization in output (default: False)
            depth: Maximum number of conditions to include in rule (default: 1)
            min_improvement: Minimum relative improvement required to add a condition (default: 0.05)
            reopt_iterations: Number of re-optimization passes through all variables (default: 2)

        Returns:
            Dictionary containing:
                - rule: The best overall rule found
                - metrics: Metrics for the best rule
                - evolution: List of (rule, metrics) showing how rule was built
                - onedim_rules: Dict mapping each variable to its best 1D rule
        """

        # BEGIN Input validation
        if len(self._obj) == 0:
            raise ValueError("Cannot find rules on empty DataFrame")

        if target not in self._obj.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")

        if variables is None:
            variables = [col for col in self._obj.columns
                        if col != target and pandas.api.types.is_numeric_dtype(self._obj[col])]

        if not variables:
            raise ValueError("No numeric variables available for rule finding")

        if not all(var in self._obj.columns for var in variables):
            missing = [var for var in variables if var not in self._obj.columns]
            raise ValueError(f"Variables not found in DataFrame: {missing}")

        # Check if we have any data in target column after removing NAs
        if self._obj[target].dropna().empty:
            raise ValueError("No valid data in target column after removing NAs")

        # For categorical targets, check if direction is valid
        if isinstance(self._obj[target].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(self._obj[target]):
            unique_values = self._obj[target].unique()
            if direction not in unique_values:
                raise ValueError(f"Direction '{direction}' not found in target categories: {unique_values}")
        # END INPUT VALIDATION

        # First pass: find all 1D rules and the best single variable
        onedim_rules = {}
        best_rule = {}
        best_score = float('-inf')
        rule_metrics = None

        for var in variables:
            if not np.issubdtype(self._obj[var].dtype, np.number):
                continue

            rule_dict = self._find_numeric_rule(var, target, direction)
            onedim_rules[var] = {'rule': {var: rule_dict['interval']}, 'metrics': rule_dict}

            if rule_dict['score'] > best_score:
                best_score = rule_dict['score']
                best_rule = {var: rule_dict['interval']}
                rule_metrics = rule_dict

        # If we didn't find any valid rules, create a default metrics dict
        if rule_metrics is None:
            rule_metrics = {
                'score': float('-inf'),
                'matching_samples': 0,
                'total_samples': len(self._obj),
                'coverage': 0.0
            }

        # Initialize evolution tracking
        evolution = [(best_rule, rule_metrics)]
        current_rule = best_rule
        current_metrics = rule_metrics
        used_vars = list(current_rule.keys())

        # Add more conditions if depth > 1
        if depth > 1:
            for _ in range(depth - 1):
                # Get remaining variables
                remaining_vars = [col for col in variables if col not in used_vars]

                if not remaining_vars:
                    break

                # Create mask for current rule
                def matches_current_rule(row):
                    for feature, interval in current_rule.items():
                        if not (interval[0] <= row[feature] <= interval[1]):
                            return False
                    return True

                current_mask = self._obj.apply(matches_current_rule, axis=1)
                matching_df = self._obj[current_mask]

                # Try each remaining variable
                best_new_rule = None
                best_new_metrics = None

                for var in remaining_vars:
                    if not np.issubdtype(self._obj[var].dtype, np.number):
                        continue

                    result = matching_df.findrule(target=target, direction=direction, variables=[var])

                    if result['metrics']['score'] > current_metrics['score'] * (1 + min_improvement):
                        new_rule = current_rule.copy()
                        new_rule.update(result['rule'])

                        # Evaluate complete rule
                        test_mask = self._obj.apply(
                            lambda row: all(interval[0] <= row[var] <= interval[1]
                                          for var, interval in new_rule.items()),
                            axis=1
                        )
                        test_metrics = self._calculate_metrics(test_mask, target, direction)

                        if (best_new_metrics is None or
                            test_metrics['score'] > best_new_metrics['score']):
                            best_new_rule = new_rule
                            best_new_metrics = test_metrics

                # If we found an improvement, update current rule
                if best_new_rule is not None:
                    current_rule = best_new_rule
                    current_metrics = best_new_metrics
                    used_vars = list(current_rule.keys())
                    evolution.append((current_rule, current_metrics))

                    # Re-optimize intervals
                    for _ in range(reopt_iterations):
                        for var_to_reopt in used_vars:
                            # Create temporary rule without the variable we're re-optimizing
                            temp_rule = {k: v for k, v in current_rule.items()
                                       if k != var_to_reopt}

                            # Get data matching all other conditions
                            temp_mask = self._obj.apply(
                                lambda row: all(interval[0] <= row[var] <= interval[1]
                                              for var, interval in temp_rule.items()),
                                axis=1
                            )
                            temp_matching_df = self._obj[temp_mask]

                            # Find optimal interval for this variable
                            result = temp_matching_df.findrule(
                                target=target,
                                direction=direction,
                                variables=[var_to_reopt]
                            )

                            # Update interval if it improves overall rule
                            test_rule = temp_rule.copy()
                            test_rule.update(result['rule'])

                            test_mask = self._obj.apply(
                                lambda row: all(interval[0] <= row[var] <= interval[1]
                                              for var, interval in test_rule.items()),
                                axis=1
                            )
                            test_metrics = self._calculate_metrics(test_mask, target, direction)

                            if test_metrics['score'] > current_metrics['score'] * (1 + min_improvement):
                                current_rule = test_rule
                                current_metrics = test_metrics
                                evolution.append((current_rule, current_metrics))
                else:
                    break

        # Prepare return dictionary
        result = {
            'rule': current_rule,
            'metrics': current_metrics,
            'evolution': evolution,
            'onedim_rules': onedim_rules
        }

        if visualize:
            result['visualization'] = self._create_rule_visualization(
                current_rule, target, direction, current_metrics, bins=bins
            )

        return result

    def _find_numeric_rule(self, feature: str, target: str, direction: str) -> Dict:
        """Find optimal interval for numeric feature using modified Kadane's algorithm."""
        # Instead of sorting every time, check our cache:
        if feature not in self._sorted_dfs:
            # Cache a sorted DataFrame containing only this feature and the target
            self._sorted_dfs[feature] = self._obj[[feature, target]].sort_values(feature).reset_index(drop=True)

        # Retrieve the cached DataFrame
        sorted_df = self._sorted_dfs[feature]

        # Create target values array with proper standardization
        if pandas.api.types.is_numeric_dtype(self._obj[target]):
            values = sorted_df[target].values
            # Standardize numeric targets
            std_dev = np.std(values)
            if std_dev == 0:  # Handle single value case
                outcomes = np.zeros_like(values, dtype=float)
            else:
                outcomes = (values - np.mean(values)) / std_dev
            if direction == 'minimize':
                outcomes = -outcomes
        else:
            # For categorical targets, create binary outcomes and standardize
            outcomes = (sorted_df[target] == direction).astype(float)
            std_dev = np.std(outcomes)
            if std_dev == 0:  # Handle single value case
                outcomes = np.zeros_like(outcomes, dtype=float)
            else:
                outcomes = (outcomes - np.mean(outcomes)) / std_dev

        # Group by unique feature values and aggregate scores
        grouped = pd.DataFrame({
            'feature_val': sorted_df[feature],
            'outcome': outcomes
        }).groupby('feature_val', observed=True)['outcome'].sum().reset_index()

        # Convert to numpy arrays for Kadane's
        feature_values = grouped['feature_val'].values
        agg_scores = grouped['outcome'].values.astype(np.float64)  # Ensure float64 for numba

        # Find optimal interval using Kadane's
        max_sum, best_start, best_end = find_best_interval_kadane(agg_scores)

        # Get interval bounds from feature values
        interval = (feature_values[best_start], feature_values[best_end])

        # Calculate metrics
        rule_mask = (self._obj[feature] >= interval[0]) & (self._obj[feature] <= interval[1])
        metrics = self._calculate_metrics(rule_mask, target, direction)
        metrics['interval'] = interval

        return metrics

    def _find_categorical_rule(self, feature: str, target: str, direction: str) -> Dict:
        """Find best category for categorical feature."""
        best_value = None
        best_score = float('-inf')
        best_metrics = None

        for value in self._obj[feature].unique():
            rule_mask = self._obj[feature] == value
            metrics = self._calculate_metrics(rule_mask, target, direction)

            if metrics['score'] > best_score:
                best_score = metrics['score']
                best_value = value
                best_metrics = metrics

        best_metrics['value'] = best_value
        return best_metrics

    def _calculate_metrics(self, rule_mask: pd.Series, target: str, direction: str) -> Dict:
        """Calculate performance metrics for a rule."""
        matching_data = self._obj[rule_mask][target]
        non_matching_data = self._obj[~rule_mask][target]

        if len(matching_data) == 0 or len(non_matching_data) == 0:
            return {
                'score': float('-inf'),
                'matching_samples': len(matching_data),
                'total_samples': len(self._obj),
                'coverage': len(matching_data) / len(self._obj)
            }

        if pd.api.types.is_numeric_dtype(self._obj[target]):
            # For numeric targets
            matching_median = matching_data.median()
            non_matching_median = non_matching_data.median()
            matching_mean = matching_data.mean()
            non_matching_mean = non_matching_data.mean()

            # Handle single value case - if both medians are equal, no meaningful split found
            if matching_median == non_matching_median:
                return {
                    'score': float('-inf'),
                    'matching_median': matching_median,
                    'non_matching_median': non_matching_median,
                    'matching_mean': matching_median,
                    'non_matching_mean': non_matching_median,
                    'matching_std': 0,
                    'non_matching_std': 0,
                    'matching_samples': len(matching_data),
                    'total_samples': len(self._obj),
                    'coverage': len(matching_data) / len(self._obj)
                }

            if direction == 'maximize':
                score = (matching_median - non_matching_median) / non_matching_median
            else:  # minimize
                score = (non_matching_median - matching_median) / non_matching_median

            # Additional metrics for continuous outcomes
            metrics = {
                'score': score,
                'matching_median': matching_median,
                'non_matching_median': non_matching_median,
                'matching_mean': matching_mean,
                'non_matching_mean': non_matching_mean,
                'matching_std': matching_data.std(),
                'non_matching_std': non_matching_data.std(),
                'matching_samples': len(matching_data),
                'total_samples': len(self._obj),
                'coverage': len(matching_data) / len(self._obj),

                # Effect size metrics
                'cohens_d': (matching_mean - non_matching_mean) /
                        np.sqrt((matching_data.var() + non_matching_data.var()) / 2),

                # Percentile-based metrics
                'matching_quartiles': matching_data.quantile([0.25, 0.5, 0.75]).to_dict(),
                'non_matching_quartiles': non_matching_data.quantile([0.25, 0.5, 0.75]).to_dict(),

                # Distribution metrics
                'matching_skew': matching_data.skew(),
                'matching_kurtosis': matching_data.kurtosis(),

                # Range metrics
                'matching_range': matching_data.max() - matching_data.min(),
                'matching_iqr': matching_data.quantile(0.75) - matching_data.quantile(0.25)
            }

            # Add robustness metrics
            bootstrap_scores = []
            for _ in range(100):  # 100 bootstrap samples
                sample_idx = np.random.choice(len(self._obj), size=len(self._obj), replace=True)
                sample_mask = rule_mask.iloc[sample_idx]
                sample_matching = self._obj.iloc[sample_idx][sample_mask][target]
                sample_non_matching = self._obj.iloc[sample_idx][~sample_mask][target]

                if len(sample_matching) > 0 and len(sample_non_matching) > 0:
                    sample_score = ((sample_matching.median() - sample_non_matching.median()) /
                                sample_non_matching.median())
                    bootstrap_scores.append(sample_score)

            metrics['score_std'] = np.std(bootstrap_scores) if bootstrap_scores else np.nan
            metrics['score_95ci'] = (np.percentile(bootstrap_scores, [2.5, 97.5])
                                if bootstrap_scores else (np.nan, np.nan))

            return metrics

        else:
            # Original categorical metrics code remains the same
            matching_rate = (matching_data == direction).mean()
            non_matching_rate = (non_matching_data == direction).mean()
            improvement = matching_rate - non_matching_rate

            true_positives = (matching_data == direction).sum()
            false_positives = (matching_data != direction).sum()
            false_negatives = (non_matching_data == direction).sum()

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'score': improvement,
                'matching_rate': matching_rate,
                'non_matching_rate': non_matching_rate,
                'matching_samples': len(matching_data),
                'total_samples': len(self._obj),
                'coverage': len(matching_data) / len(self._obj),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }

    # pd_1d_rule_solver.py: class RuleFinder: def _create_rule_visualization
    def _create_rule_visualization(self, rule: Dict, target: str, direction: str, metrics: Dict, bins: Optional[int] = None) -> str:
        """
        Create text visualization of rule's impact.

        Args:
            rule: Dictionary containing the rule conditions
            target: Target variable name
            direction: Direction of optimization or target category
            metrics: Dictionary containing the rule's metrics
            bins: Number of bins for histogram (optional)
        """
        # Create rule mask
        def matches_rule(row):
            for feature, value in rule.items():
                if isinstance(value, tuple):
                    if not (value[0] <= row[feature] <= value[1]):
                        return False
                else:
                    if row[feature] != value:
                        return False
            return True

        rule_mask = self._obj.apply(matches_rule, axis=1)
        matching_df = self._obj[rule_mask]
        non_matching_df = self._obj[~rule_mask]

        # Build visualization
        lines = [
            "Rule Impact Analysis",
            "=" * 50,
            f"\nRule Conditions:"
        ]

        # Format rule conditions
        for feature, value in rule.items():
            if isinstance(value, tuple):
                lines.append(f"  {feature}: {value[0]:.3f} to {value[1]:.3f}")
            else:
                lines.append(f"  {feature}: {value}")

        # Add matching samples info
        total_matching = len(matching_df)
        lines.append(f"\nMatching Samples: {total_matching} "
                    f"({total_matching/len(self._obj):.1%} of data)")

        # Show target distribution
        if pandas.api.types.is_numeric_dtype(self._obj[target]):
            matching_median = matching_df[target].median()
            non_matching_median = non_matching_df[target].median()
            all_median = self._obj[target].median()
            all_values = self._obj[target].values
            matching_values = matching_df[target].values

            # Determine number of bins if not specified
            if bins is None:
                n_unique = len(np.unique(all_values))
                bins = min(12, n_unique)

            lines.extend([
                f"\n{target} Distribution:",
                f"Original median: {all_median:.2f}",
                f"Rule median: {matching_median:.2f}",
                ""
            ])

            # Add distribution shift histogram
            histogram = self._create_distribution_shift_histogram(
                all_values,
                matching_values,
                bins=bins,
                width=60
            )
            lines.extend(histogram)

            improvement = ((matching_median - non_matching_median) / non_matching_median * 100
                        if direction == 'maximize' else
                        (non_matching_median - matching_median) / non_matching_median * 100)
            lines.append(f"\nMedian improvement: {improvement:+.1f}%")

        else:
            # For categorical target
            matching_counts = matching_df[target].value_counts(normalize=True)
            non_matching_counts = non_matching_df[target].value_counts(normalize=True)

            lines.extend([
                f"\n{target} Distribution:",
                "\nMatching samples:"
            ])

            for cat, pct in matching_counts.items():
                lines.append(f"{cat:12} | {'█' * int(pct * 20):<20} | {pct:.1%}")

            lines.append("\nNon-matching samples:")
            for cat, pct in non_matching_counts.items():
                lines.append(f"{cat:12} | {'█' * int(pct * 20):<20} | {pct:.1%}")

            target_improvement = (matching_counts.get(direction, 0) -
                                non_matching_counts.get(direction, 0)) * 100

            # Add F1 score metrics using passed metrics dictionary
            lines.extend([
                f"\nTarget class improvement: {target_improvement:+.1f}%",
                f"F1 Score: {metrics['f1_score']:.3f}",
                f"Precision: {metrics['precision']:.3f}",
                f"Recall: {metrics['recall']:.3f}"
            ])

        # Add rule details at the end
        lines.extend([
            "",
            "Rule details:",
            str(rule)
        ])

        return '\n'.join(lines)

    # pd_1d_rule_solver.py: class RuleFinder: def _create_distribution_shift_histogram
    def _create_distribution_shift_histogram(self,
                                        all_values: np.ndarray,
                                        matching_values: Optional[np.ndarray] = None,
                                        bins: Optional[int] = None,
                                        width: int = 60) -> List[str]:
        """
        Create a histogram showing distribution shift with hatched patterns.
        Shows different patterns for rule-only, original-only, and overlapping regions.

        Args:
            all_values: Array of all values (original distribution)
            matching_values: Array of values matching the rule
            bins: Number of bins (default: 12 or number of unique values if less)
            width: Width of the visualization in characters

        Returns:
            List of strings representing the histogram
        """
        # Determine number of bins
        if bins is None:
            n_unique = len(np.unique(all_values))
            bins = min(12, n_unique)

        # Calculate histograms with same bins for both distributions
        hist_orig, bin_edges = np.histogram(all_values, bins=bins, density=True)
        hist_rule, _ = np.histogram(matching_values, bins=bin_edges, density=True)

        # Find the maximum height for scaling
        max_height = max(max(hist_orig), max(hist_rule))

        # Scale to desired width (each value will become N characters wide)
        scale = width / max_height if max_height > 0 else 1

        # Build the visualization
        lines = []
        max_label_width = 10  # Width for the frequency labels

        # Add top label line
        # lines.append(" " * max_label_width + "┬" + "─" * width)

        # Create each row of the histogram
        for freq_orig, freq_rule, edge in zip(hist_orig, hist_rule, bin_edges[:-1]):
            # Calculate bar lengths
            rule_len = int(freq_rule * scale)
            orig_len = int(freq_orig * scale)

            # Create the base bar
            bar = ""

            # Fill the bar with proper characters
            for i in range(width):
                if i < rule_len and i < orig_len:
                    # Both distributions overlap here
                    bar += "▓"  # Heavy shade for overlap
                elif i < rule_len:
                    # Only rule distribution
                    bar += "█"  # Full block for rule-only
                elif i < orig_len:
                    # Only original distribution
                    bar += "░"  # Light shade for original-only
                else:
                    bar += " "  # Empty space

            # Add label (right-aligned with 1 decimal place) and bar
            label = f"{edge:8.1f} │"
            lines.append(f"{label}{bar}")

        # Add bottom border
        # lines.append(" " * max_label_width + "┴" + "─" * width)

        # Add legend
        lines.extend([
            "",
            "Legend: █ Rule-only distribution",
            "       ░ Original-only distribution",
            "       ▓ Overlapping distributions"
        ])

        return lines


# Example usage:
if __name__ == "__main__":
    # Load sample data
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Example 1: Find rule for maximizing sepal length with default bins
    print("\nExample 1: Maximizing sepal length (default bins)")
    result = df.findrule(
        target='sepal length (cm)',
        direction='maximize',
        variables=['petal length (cm)', 'petal width (cm)']
    )
    print(result['visualization'])

    # Example 2: Find rule for maximizing sepal length with custom bins
    print("\nExample 2: Maximizing sepal length (20 bins)")
    result = df.findrule(
        target='sepal length (cm)',
        direction='maximize',
        variables=['petal length (cm)', 'petal width (cm)'],
        bins=20
    )
    print(result['visualization'])

    # Example 3: Find rule for predicting setosa species
    print("\nExample 3: Predicting setosa species")
    result = df.findrule(
        target='species',
        direction='setosa',
        variables=['sepal length (cm)', 'sepal width (cm)']
    )
    print(result['visualization'])

    # Example 4: Find rule for minimizing petal length
    print("\nExample 4: Minimizing petal length")
    result = df.findrule(
        target='petal length (cm)',
        direction='minimize',
        variables=['sepal length (cm)', 'sepal width (cm)'],
        bins=15
    )
    print(result['visualization'])
