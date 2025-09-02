# Development Sandbox

## Purpose

This directory contains experimental code, prototypes, and test implementations that are not yet ready for production. Use this space to:

- Test new features and approaches
- Compare different implementations
- Benchmark performance optimizations
- Prototype visualizations
- Experiment with algorithm variations

## Structure

Organize experiments by the module they relate to:

```bash
/dev_sandbox
  /spacetime          # Experiments related to spacetime module
  /performance        # Performance benchmarks and tests
  /visualizations     # Visualization prototypes
```

## Import Pattern

Since sandbox files are outside the main package structure, add the project root to the Python path:

```python
import sys
from pathlib import Path
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import normally
from openwave.spacetime import quantum_space
from openwave.core import constants
```

## Naming Conventions

Use descriptive names that indicate the experiment's purpose:

- `*_test.py` - Testing specific functionality
- `*_benchmark.py` - Performance benchmarking
- `*_compare.py` - Comparing implementations
- `*_proto.py` - Prototypes
- `wip_*.py` - Work in progress

## Git Strategy

### Option 1: Track experiments (Recommended for small teams)

Commit experiments with clear commit messages:

```bash
git add dev_sandbox/quantum_space/new_experiment.py
git commit -m "dev_sandbox: testing wave interference patterns"
```

### Option 2: Ignore experiments (For personal testing)

Add to `.gitignore`:

```bash
/dev_sandbox/*
!/dev_sandbox/README.md
```

### Option 3: Selective tracking

Track only certain experiments:

```bash
/dev_sandbox/wip_*
/dev_sandbox/*/wip_*
```

## Best Practices

1. **Document experiments**: Add docstrings explaining what you're testing
2. **Clean up regularly**: Remove failed experiments or move successful ones to production
3. **Use meaningful variable names**: Even in test code, maintain readability
4. **Add results comments**: Document findings directly in the code
5. **Date your experiments**: Include creation date in file docstring

## Example Experiment File

```python
"""
Experiment: Quantum Space 2D Slider Visualization
Created: 2024-12-XX
Purpose: Test interactive parameter adjustment for wave visualization
Results: [Document findings here]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openwave.spacetime import quantum_space

# Experiment code here...
```

## Promotion Path

Successful experiments should follow this path:

1. **Sandbox**: Initial experimentation
2. **Review**: Code review and cleanup
3. **Tests**: Add formal tests in `/tests`
4. **Integration**: Merge into main codebase
5. **Documentation**: Update relevant docs

## Note

This directory is for development only.

Do not import sandbox code into production modules.
