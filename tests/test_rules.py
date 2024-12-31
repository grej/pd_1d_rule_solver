# tests/test_rules.py

import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import warnings
import sys
from pathlib import Path

# Add parent directory to path so we can import the package
module_path = str(Path('.').absolute().parent)
if module_path not in sys.path:
    sys.path.append(module_path)

# Import and register the RuleFinder extension
from pd_1d_rule_solver.pd_1d_rule_solver import RuleFinder  # This registers the extension automatically

class TestRuleFinder(unittest.TestCase):
    def setUp(self):
        """Create test datasets for continuous and categorical targets"""
        np.random.seed(42)

        # Create continuous target dataset with embedded rule
        self.n_samples = 1000
        self.n_features = 5
        X = np.random.normal(0, 1, (self.n_samples, self.n_features))
        self.df_continuous = pd.DataFrame(X, columns=[f'x{i}' for i in range(self.n_features)])
        self.df_continuous['target'] = np.random.normal(10, 2, self.n_samples)

        # Embed a known rule: if x0 > 1 and x1 < -1 and x2 > 0, multiply target by 2
        mask = (self.df_continuous['x0'] > 1) & \
               (self.df_continuous['x1'] < -1) & \
               (self.df_continuous['x2'] > 0)
        self.df_continuous.loc[mask, 'target'] *= 2

        # Create categorical target dataset
        self.df_categorical = self.df_continuous.copy()
        # Create categories based on target value
        self.df_categorical['target'] = pd.cut(
            self.df_categorical['target'],
            bins=3,
            labels=['low', 'medium', 'high']
        )

    def test_basic_continuous_rule(self):
        """Test finding a single condition rule for continuous target"""
        result = self.df_continuous.findrule(
            target='target',
            direction='maximize',
            depth=1
        )

        self.assertIsInstance(result, dict)
        self.assertIn('rule', result)
        self.assertIn('metrics', result)
        self.assertIn('evolution', result)
        self.assertIn('onedim_rules', result)

        # Check that we found exactly one condition
        self.assertEqual(len(result['rule']), 1)

        # Check that evolution has exactly one step
        self.assertEqual(len(result['evolution']), 1)

        # Check that we have 1D rules for all features
        self.assertEqual(
            len(result['onedim_rules']),
            len([col for col in self.df_continuous.columns if col != 'target'])
        )

    def test_depth_continuous_rule(self):
        """Test finding multiple conditions for continuous target"""
        max_depth = 3
        result = self.df_continuous.findrule(
            target='target',
            direction='maximize',
            depth=max_depth
        )

        # Check that we found at most max_depth conditions
        self.assertLessEqual(len(result['rule']), max_depth)

        # Check that each step in evolution improves the score
        for i in range(1, len(result['evolution'])):
            self.assertGreater(
                result['evolution'][i][1]['score'],
                result['evolution'][i-1][1]['score']
            )

    def test_categorical_rule(self):
        """Test finding rules for categorical target"""
        result = self.df_categorical.findrule(
            target='target',
            direction='high',
            depth=2
        )

        self.assertIsInstance(result, dict)
        self.assertIn('rule', result)
        self.assertIn('metrics', result)

        # Verify metrics contains categorical-specific metrics
        self.assertIn('precision', result['metrics'])
        self.assertIn('recall', result['metrics'])
        self.assertIn('f1_score', result['metrics'])

    def test_min_improvement_threshold(self):
        """Test that conditions are only added if they meet minimum improvement"""
        min_improvement = 0.5  # Set high threshold
        result = self.df_continuous.findrule(
            target='target',
            direction='maximize',
            depth=3,
            min_improvement=min_improvement
        )

        # Check each evolution step meets minimum improvement
        for i in range(1, len(result['evolution'])):
            improvement = (
                result['evolution'][i][1]['score'] -
                result['evolution'][i-1][1]['score']
            ) / abs(result['evolution'][i-1][1]['score'])
            self.assertGreaterEqual(improvement, min_improvement)

    def test_reoptimization(self):
        """Test that reoptimization can improve rule quality"""
        # First find rule without reoptimization
        result_no_reopt = self.df_continuous.findrule(
            target='target',
            direction='maximize',
            depth=2,
            reopt_iterations=0
        )

        # Then with reoptimization
        result_with_reopt = self.df_continuous.findrule(
            target='target',
            direction='maximize',
            depth=2,
            reopt_iterations=2
        )

        # Store final scores for comparison
        score_no_reopt = result_no_reopt['metrics']['score']
        score_with_reopt = result_with_reopt['metrics']['score']

        # Print scores for debugging
        print(f"Score without reopt: {score_no_reopt:.3f}")
        print(f"Score with reopt: {score_with_reopt:.3f}")

    def test_visualization(self):
        """Test that visualization is generated when requested"""
        result = self.df_continuous.findrule(
            target='target',
            direction='maximize',
            visualize=True
        )

        self.assertIn('visualization', result)
        self.assertIsInstance(result['visualization'], str)

    def test_variable_selection(self):
        """Test that rules only use specified variables"""
        variables = ['x0', 'x1']
        result = self.df_continuous.findrule(
            target='target',
            direction='maximize',
            variables=variables,
            depth=3
        )

        # Check that all conditions use only specified variables
        for var in result['rule'].keys():
            self.assertIn(var, variables)

        # Check that onedim_rules only contains specified variables
        self.assertEqual(set(result['onedim_rules'].keys()), set(variables))

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty DataFrame
        df_empty = pd.DataFrame(columns=['x0', 'target'])
        with self.assertRaises(ValueError):
            df_empty.findrule(target='target', direction='maximize')

        # Single value in target
        df_single = self.df_continuous.copy()
        df_single['target'] = 1.0
        result = df_single.findrule(target='target', direction='maximize')
        self.assertEqual(result['metrics']['score'], float('-inf'))

        # No numeric features
        df_no_numeric = pd.DataFrame({
            'cat1': ['A', 'B'] * 50,
            'cat2': ['X', 'Y'] * 50,
            'target': range(100)
        })
        with self.assertRaises(ValueError):
            df_no_numeric.findrule(target='target', direction='maximize')

if __name__ == '__main__':
    unittest.main(verbosity=2)
