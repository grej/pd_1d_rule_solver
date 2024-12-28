import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from typing import List, Union, Dict, Tuple, Optional
import warnings

@pd.api.extensions.register_dataframe_accessor("findrule")
class RuleFinder:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, target: str, direction: str, variables: List[str], bins: Optional[int] = None) -> Dict:
        """
        Find a rule that shifts the distribution of the target variable in the desired direction.

        Args:
            target: Target variable name
            direction: Either 'maximize'/'minimize' for numeric targets, or a category name for categorical
            variables: List of feature names to consider for the rule
            bins: Number of bins for histogram visualization (default: 12 or number of unique values if less)

        Returns:
            Dictionary containing the rule and its impact metrics
        """
        # [Previous validation code remains the same]

        # Pass bins parameter through to visualization
        best_rule = {}
        best_score = float('-inf')
        rule_metrics = None

        # Process each variable
        for var in variables:
            if is_numeric_dtype(self._obj[var]):
                # For numeric variables, find optimal interval
                rule_dict = self._find_numeric_rule(var, target, direction)
                if rule_dict['score'] > best_score:
                    best_score = rule_dict['score']
                    best_rule = {var: rule_dict['interval']}
                    rule_metrics = rule_dict
            else:
                # For categorical variables, find best category
                rule_dict = self._find_categorical_rule(var, target, direction)
                if rule_dict['score'] > best_score:
                    best_score = rule_dict['score']
                    best_rule = {var: rule_dict['value']}
                    rule_metrics = rule_dict

        # Create visualization with bin parameter
        viz = self._create_rule_visualization(best_rule, target, direction, rule_metrics, bins=bins)

        return {
            'rule': best_rule,
            'metrics': rule_metrics,
            'visualization': viz
        }

    def _find_numeric_rule(self, feature: str, target: str, direction: str) -> Dict:
        """Find optimal interval for numeric feature using modified Kadane's algorithm."""
        # Sort data by feature
        sorted_df = self._obj.sort_values(feature).reset_index(drop=True)

        # Prepare target values
        if is_numeric_dtype(self._obj[target]):
            # For numeric targets, standardize values
            values = sorted_df[target].values
            outcomes = (values - np.mean(values)) / np.std(values)
            if direction == 'minimize':
                outcomes = -outcomes
        else:
            # For categorical targets, create binary outcomes
            outcomes = (sorted_df[target] == direction).astype(float)
            outcomes = (outcomes - np.mean(outcomes)) / np.std(outcomes)

        # Run Kadane's algorithm
        max_sum = float('-inf')
        current_sum = 0
        start = 0
        best_start = 0
        best_end = 0

        for i in range(len(outcomes)):
            if current_sum <= 0:
                start = i
                current_sum = outcomes[i]
            else:
                current_sum += outcomes[i]

            if current_sum > max_sum:
                max_sum = current_sum
                best_start = start
                best_end = i

        # Get interval bounds
        interval = (
            sorted_df[feature].iloc[best_start],
            sorted_df[feature].iloc[best_end]
        )

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
            return {'score': float('-inf')}

        if is_numeric_dtype(self._obj[target]):
            # For numeric targets
            matching_median = matching_data.median()
            non_matching_median = non_matching_data.median()

            if direction == 'maximize':
                score = (matching_median - non_matching_median) / non_matching_median
            else:  # minimize
                score = (non_matching_median - matching_median) / non_matching_median

            return {
                'score': score,
                'matching_median': matching_median,
                'non_matching_median': non_matching_median,
                'matching_samples': len(matching_data),
                'total_samples': len(self._obj),
                'coverage': len(matching_data) / len(self._obj)
            }
        else:
            # For categorical targets
            matching_rate = (matching_data == direction).mean()
            non_matching_rate = (non_matching_data == direction).mean()
            improvement = matching_rate - non_matching_rate

            # Calculate precision, recall, and F1 score
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
        if is_numeric_dtype(self._obj[target]):
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
