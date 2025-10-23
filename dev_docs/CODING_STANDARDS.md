# Coding Standards

## Overview

This document outlines the coding standards and conventions for the OpenWave project.
Python style guide, naming conventions, documentation standards, and code review checklist.

## Python Style Guide

### General Principles

- Follow PEP 8 for Python code style
- Use descriptive variable and function names
- Write self-documenting code with clear intent
- Keep functions small and focused on a single responsibility

### Naming Conventions

- **Classes**: PascalCase (e.g., `WaveFunction`, `ParticleSimulator`)
- **Functions/Methods**: snake_case (e.g., `calculate_energy`, `update_position`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `PLANCK_CONSTANT`, `SPEED_OF_LIGHT`)
- **Private Methods**: Leading underscore (e.g., `_internal_calculation`)

### Code Organization

- Import statements grouped in order: standard library, third-party, local imports
- One class per file for major components
- Related utility functions grouped in modules

### Documentation

- Use docstrings for all public classes, methods, and functions
- Follow NumPy documentation style for scientific functions
- Include type hints for function parameters and return values

### Example

```python
def calculate_wave_amplitude(
    frequency: float,
    wavelength: float,
    time: float
) -> float:
    """
    Calculate the amplitude of an energy wave at a given time.
    
    Parameters
    ----------
    frequency : float
        Wave frequency in Hz
    wavelength : float
        Wavelength in meters
    time : float
        Time in seconds
        
    Returns
    -------
    float
        Wave amplitude
    """
    # Implementation here
    pass
```

## Taichi-Specific Guidelines

### Kernel Functions

- Keep kernel functions pure and side-effect free when possible
- Minimize branching in performance-critical kernels
- Use Taichi's native types for better performance

### Memory Management

- Pre-allocate Taichi fields when sizes are known
- Use appropriate data layouts (SoA vs AoS) based on access patterns
- Clear explanation of memory layout choices in comments

## Testing Standards

### Test Coverage

- Aim for minimum 80% code coverage
- All physics calculations must have validation tests
- Performance-critical code should include benchmark tests

### Test Organization

- Mirror source code structure in test directory
- Use descriptive test names that explain what is being tested
- Include both unit tests and integration tests

## Version Control

### Commit Messages

- Use clear, descriptive commit messages
- Follow conventional commits format when applicable
- Reference issue numbers when fixing bugs

### Branch Naming

- `feature/description` for new features
- `fix/issue-description` for bug fixes
- `perf/optimization-description` for performance improvements

## Code Review Checklist

Before submitting a pull request:

- Code follows style guidelines
- All tests pass
- Documentation is updated
- Performance implications considered
- No hardcoded values (use constants)
- Error handling is appropriate
