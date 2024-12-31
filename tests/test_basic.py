import sys
from pathlib import Path

# Add parent directory to path so we can import the package
module_path = str(Path('.').absolute().parent)
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
from pd_1d_rule_solver.pd_1d_rule_solver import RuleFinder  # This registers the extension automatically

# Load sample iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Look at the data
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Without visualization
# Example 1: Find rule for maximizing sepal length with default bins (12)
print("\nExample 1: Maximizing sepal length (default bins)")
result = df.findrule(
    target='sepal length (cm)',
    direction='maximize',
    variables=['petal length (cm)', 'petal width (cm)']
)
print(result)

# Example 2: Same analysis with more bins
print("\nExample 2: Same analysis with 20 bins")
result = df.findrule(
    target='sepal length (cm)',
    direction='maximize',
    variables=['petal length (cm)', 'petal width (cm)'],
    bins=20
)
print(result)

# Example 3: Find rule for predicting setosa species
print("\nExample 3: Predicting setosa species")
result = df.findrule(
    target='species',
    direction='setosa',
    variables=['sepal length (cm)', 'sepal width (cm)']
)
print(result)

# Access the metrics for the best rule
print("\nDetailed metrics for the best rule:")
print(f"Score: {result['metrics']['score']:.3f}")
print(f"Coverage: {result['metrics']['coverage']:.1%}")
print(f"Matching samples: {result['metrics']['matching_samples']}")

# Example 1: Find rule for maximizing sepal length with default bins (12)
print("\nExample 1: Maximizing sepal length (default bins)")
result = df.findrule(
    target='sepal length (cm)',
    direction='maximize',
    variables=['petal length (cm)', 'petal width (cm)'],
    visualize=True
)
print(result['visualization'])

# Example 2: Same analysis with more bins
print("\nExample 2: Same analysis with 20 bins")
result = df.findrule(
    target='sepal length (cm)',
    direction='maximize',
    variables=['petal length (cm)', 'petal width (cm)'],
    bins=20,
    visualize=True
)
print(result['visualization'])

# Example 3: Find rule for predicting setosa species
print("\nExample 3: Predicting setosa species")
result = df.findrule(
    target='species',
    direction='setosa',
    variables=['sepal length (cm)', 'sepal width (cm)'],
    visualize=True
)
print(result['visualization'])

# Access the metrics for the best rule
print("\nDetailed metrics for the best rule:")
print(f"Score: {result['metrics']['score']:.3f}")
print(f"Coverage: {result['metrics']['coverage']:.1%}")
print(f"Matching samples: {result['metrics']['matching_samples']}")

result = df.findrule(
    target='species',
    direction='setosa',
    variables=['sepal length (cm)', 'sepal width (cm)'],
    visualize=True
)

print(result['visualization'])


# Example 4: Extending Rules with Additional Conditions
print("\nExample 4: Extending Rules")

# First, let's find an initial rule for setosa
initial_result = df.findrule(
    target='species',
    direction='setosa',
    variables=['sepal length (cm)'],
    visualize=True
)
print("\nInitial Rule:")
print(initial_result['visualization'])

# Now let's extend the rule
from pd_1d_rule_solver.rule_extension import suggest_rule_extension

extended_rule, metrics, suggestions = suggest_rule_extension(
    df,
    current_rule=initial_result['rule'],
    target='species',
    direction='setosa'
)

# Show the extended rule
result_with_extension = df.findrule(
    target='species',
    direction='setosa',
    variables=list(extended_rule.keys()),
    visualize=True
)
print("\nExtended Rule:")
print(result_with_extension['visualization'])

# Print improvement metrics
print("\nRule Evolution:")
print(f"Initial F1 Score: {initial_result['metrics']['f1_score']:.3f}")
print(f"Extended F1 Score: {metrics['f1_score']:.3f}")
print(f"Improvement: {(metrics['f1_score'] - initial_result['metrics']['f1_score'])/initial_result['metrics']['f1_score']:.1%}")

# Show suggestions for further extensions
print("\nSuggested Variables for Further Extension:")
for suggestion in suggestions:
    print(f"\nVariable: {suggestion.name}")
    print(f"Improvement potential: {suggestion.improvement_potential:.3f}")
    print(f"Mutual information: {suggestion.mi_score:.3f}")
    print(f"Boundary sensitivity: {suggestion.boundary_sensitivity:.3f}")


# Example 5: Find rule for predicting setosa species
print("\nExample 5: Predicting setosa species")
result = df.findrule(
    target='species',
    direction='setosa',
    variables=['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)'],
    visualize=True
)
print(result['visualization'])

# Access the metrics for the best rule
print("\nDetailed metrics for the best rule:")
print(f"Score: {result['metrics']['score']:.3f}")
print(f"Coverage: {result['metrics']['coverage']:.1%}")
print(f"Matching samples: {result['metrics']['matching_samples']}")

result = df.findrule(
    target='species',
    direction='setosa',
    variables=['sepal length (cm)', 'sepal width (cm)'],
    visualize=True
)

print(result['visualization'])
