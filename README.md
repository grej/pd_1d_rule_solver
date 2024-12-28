# Simple Pandas Rule Finder

A simple pandas extension for finding rules in your data that maximize or minimize target variables, or predict categorical outcomes.

## Quick Start

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pandas-rule-finder
```

2. Copy `rule_finder.py` to your working directory or notebook location

3. Import in your notebook:
```python
import pandas as pd
import rule_finder  # This registers the extension automatically

# Load your data
df = pd.DataFrame(...)

# Find a rule to maximize a numeric variable
result = df.findrule(
    target='price',
    direction='maximize',
    variables=['square_feet', 'bedrooms', 'location']
)

# Print the visualization
print(result['visualization'])
```

## Examples

Check out `example.ipynb` for detailed usage examples.

## Requirements

- Python ≥ 3.7
- pandas ≥ 1.0.0
- numpy ≥ 1.18.0
