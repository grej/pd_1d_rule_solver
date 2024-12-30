"""
pd_1d_rule_solver/rule_extension.py

This module provides functionality to extend existing rules by adding new conditions
based on information-theoretic variable selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from .variable_selection.variable_selection import suggest_next_variable, VariableScore
from .pd_1d_rule_solver import RuleFinder

class RuleExtender:
    """Class for extending existing rules with additional conditions."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize rule extender.

        Args:
            df: Input DataFrame
        """
        self.df = df

    def extend_rule(self,
                current_rule: Dict,
                target: str,
                direction: str,
                candidate_vars: Optional[List[str]] = None,
                min_improvement: float = 0.05,
                max_conditions: int = 3) -> Tuple[Dict, Dict]:
        """Extend an existing rule by adding conditions that improve its performance."""
        if len(current_rule) >= max_conditions:
            return current_rule, self._evaluate_rule(current_rule, target, direction)

        # Get original rule performance
        original_metrics = self._evaluate_rule(current_rule, target, direction)
        best_score = original_metrics['score']
        best_rule = current_rule.copy()
        best_metrics = original_metrics

        # If no candidate vars provided, use all numeric columns except those already in rule
        if candidate_vars is None:
            candidate_vars = [col for col in self.df.columns
                            if col != target
                            and col not in current_rule
                            and np.issubdtype(self.df[col].dtype, np.number)]

        # Get variable suggestions
        suggestions = suggest_next_variable(
            self.df,
            target=target,
            current_rule=current_rule,
            candidate_vars=candidate_vars,
            n_suggestions=3
        )

        # Try each suggested variable
        for suggestion in suggestions:
            # Skip if variable is already in rule
            if suggestion.name in current_rule:
                continue

            # Find optimal condition for this variable
            extended_rule, metrics = self._find_optimal_extension(
                current_rule,
                suggestion.name,
                target,
                direction
            )

            # Check if improvement is sufficient and rule actually changed
            if (metrics['score'] > best_score * (1 + min_improvement) and
                len(extended_rule) > len(current_rule)):
                best_score = metrics['score']
                best_rule = extended_rule
                best_metrics = metrics

        return best_rule, best_metrics

    def extend_rule_iteratively(self,
                              initial_rule: Dict,
                              target: str,
                              direction: str,
                              candidate_vars: Optional[List[str]] = None,
                              min_improvement: float = 0.05,
                              max_conditions: int = 3) -> List[Tuple[Dict, Dict]]:
        """
        Iteratively extend a rule until no more improvements can be found.

        Returns list of (rule, metrics) tuples showing the evolution of the rule.
        """
        current_rule = initial_rule.copy()
        evolution = [(current_rule, self._evaluate_rule(current_rule, target, direction))]

        while len(current_rule) < max_conditions:
            new_rule, new_metrics = self.extend_rule(
                current_rule,
                target,
                direction,
                candidate_vars,
                min_improvement,
                max_conditions
            )

            # Check if we got an improvement
            if new_rule == current_rule:
                break

            current_rule = new_rule
            evolution.append((current_rule, new_metrics))

        return evolution

    def _evaluate_rule(self, rule: Dict, target: str, direction: str) -> Dict:
        """Evaluate a rule using the RuleFinder metrics."""
        # Create rule mask
        mask = np.ones(len(self.df), dtype=bool)
        for var, condition in rule.items():
            if isinstance(condition, tuple):
                mask &= (self.df[var] >= condition[0]) & (self.df[var] <= condition[1])
            else:
                mask &= (self.df[var] == condition)

        # Use RuleFinder's metric calculation
        finder = RuleFinder(self.df)
        metrics = finder._calculate_metrics(mask, target, direction)
        metrics['rule'] = rule
        return metrics

    def _find_optimal_extension(self,
                              current_rule: Dict,
                              new_var: str,
                              target: str,
                              direction: str) -> Tuple[Dict, Dict]:
        """Find the optimal condition on new_var to add to current_rule."""
        # Create mask for current rule
        base_mask = np.ones(len(self.df), dtype=bool)
        for var, condition in current_rule.items():
            if isinstance(condition, tuple):
                base_mask &= (self.df[var] >= condition[0]) & (self.df[var] <= condition[1])
            else:
                base_mask &= (self.df[var] == condition)

        # Create subset of data matching current rule
        matching_df = self.df[base_mask].copy()

        # Find optimal condition on new variable using RuleFinder
        finder = RuleFinder(matching_df)
        result = matching_df.findrule(
            target=target,
            direction=direction,
            variables=[new_var]
        )

        # Combine conditions
        extended_rule = current_rule.copy()
        extended_rule.update(result['rule'])

        # Evaluate complete rule
        metrics = self._evaluate_rule(extended_rule, target, direction)

        return extended_rule, metrics


def suggest_rule_extension(df: pd.DataFrame,
                         current_rule: Dict,
                         target: str,
                         direction: str,
                         candidate_vars: Optional[List[str]] = None,
                         min_improvement: float = 0.05) -> Tuple[Dict, Dict, List[VariableScore]]:
    """
    Convenience function to suggest how to extend a rule.

    Returns:
        Tuple of (extended rule, metrics, variable suggestions)
    """
    extender = RuleExtender(df)
    extended_rule, metrics = extender.extend_rule(
        current_rule, target, direction, candidate_vars, min_improvement
    )

    # Get variable suggestions for further extensions
    suggestions = suggest_next_variable(
        df,
        target=target,
        current_rule=extended_rule,
        candidate_vars=candidate_vars,
        n_suggestions=3
    )

    return extended_rule, metrics, suggestions
