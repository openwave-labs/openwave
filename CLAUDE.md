# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenWave is an open-source quantum physics simulator implementing Energy Wave Theory (EWT) to model the formation of matter and energy from quantum wave interactions. The project simulates phenomena from spacetime emergence through particle formation to complex matter behavior.

### Project Goals

To develop the OpenWave an open-source computer simulator with objectives described in the text below, based on the papers attached as 9 files, with special attention to the file `Relationship of the Speed of Light to Aether Density` where there is a Planck mass correction from previous papers (affecting granule mass), using the `a6. Constants and Equations - Waves.pdf` as constants reference, built in phases. Simulation physics, constants, and equations will be drawn from the attached EWT research papers. For performance on the granular physics simulations we'll be using the Taichi Lang python library.

### What is OpenWave?

OpenWave is an open-source application designed to simulate the formation and behavior of matter and other identities of energy — from the emergence of spacetime and quantum waves, through the creation of subatomic particles, to the development of matter, motion, light, and heat — based on the Energy Wave Theory (EWT) model.Core ScopeOpenWave provides computational and visualization tools to explore, demonstrate, and validate EWT predictions through three main functions:

#### Visual Demonstration

Illustrates complex, often invisible phenomena for better comprehension.
Represents graphically wave equations and analyses.
Automates animation export for online video publishing.

#### Numerical Validation

Runs simulations derived directly from equations.
Validates outcomes by comparing them against observed reality.
Generates numerical analysis reports for scientific publications.

#### --PLANNED-- Experiments Simulation (#energy_hacking)

Models experimental conditions to explore new tech derived from subatomic-scale energy exchange simulations.
Generates schematics to serve as baseline for patent applications.

### Known Challenges & Limitations

#### Blender Limitation Lessons

Previous QSCOPE experiments showed that Blender’s physics engine was not suitable for Planck-scale simulation (wave modifier limits, animation-focused, partial physics, limited parallel processing).
This project will use a dedicated physics computational backend, independent of 3D modeling software.

#### Granularity vs. Performance

Full Planck-scale fidelity may be computationally prohibitive; require user-tunable resolution.

## Development Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/openwaveHQ/openwave.git

# Create conda environment
conda create -n openwave312 python=3.12 -y
conda activate openwave312

# Install dependencies from project directory
cd openwave
pip install -e .

# Install LaTeX and FFmpeg (macOS)
brew install --cask mactex-no-gui ffmpeg
echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
exec zsh -l
```

### Verification

```bash
# Verify LaTeX installation
which latex && latex --version
which dvisvgm && dvisvgm --version
which gs && gs --version

# Test equations module
python -c "from openwave.equations import energy_wave_equation; print('Equations OK')"
```

## Project Architecture

### Core Module Structure

- **constants.py**: EWT wave constants, particle constants, and classical physics constants with unit conversion utilities
- **config.py**: Configuration management using INI files
- **quantum_space.py**: Spacetime simulation module (entry point incomplete)
- **quantum_wave.py**: Wave propagation and interaction modeling (entry point incomplete)
- **forces.py**: Physics forces simulation module (under development)
- **heat.py**: Thermal energy simulation module (under development)
- **matter.py**: Matter formation and behavior simulation module (under development)
- **motion.py**: Motion and velocity simulation module (under development)
- **photon.py**: Light and photon simulation module (under development)

### Configuration System

Configuration is managed through `config.ini` with sections for:

- `[universe]`: Simulation parameters (size, time_step)
- `[screen]`: Display resolution settings
- `[color]`: Color scheme for different physics entities (space, quantum_waves, matter, antimatter, motion, photons, energy, heat)

### Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing and analysis
- **matplotlib**: 2D plotting and visualization
- **taichi**: High-performance parallel computing for simulations

## CLI Usage (Work in Progress)

The project provides command-line interfaces for quantum simulations:

### Quantum Space Simulation

```bash
space -h  # or python openwave/quantum_space.py -h
# Options: -R (use-ray), -t (test), -p PREFIX, -asis, -dnr, -env, -run
```

### Quantum Wave Simulation

```bash
wave -h  # or python openwave/quantum_wave.py -h
# Same options as space command
```

Note: CLI implementations are currently incomplete but follow the pattern above.

## Code Conventions

### Constants Usage

- Import from `openwave.constants` for all physics constants
- Use EWT-specific constants (QWAVE_LENGTH, QWAVE_AMPLITUDE, etc.) for wave modeling
- Classical constants are provided for compatibility and validation
- All constants use SI units (kg, m, s)

### Configuration Access

- Load configuration via `openwave.config` module
- Access screen dimensions: `config.screen_width`, `config.screen_height`
- Configuration file: `openwave/config.ini`

### Current Issues

- CLI entry points are not fully implemented
- Testing infrastructure is minimal

## Scientific Documentation & Requirements

### Project Requirements

The `requirements/requirements_source/` directory contains simulation specification documents:

1. Simulating a Fundamental Particle - EWT.pdf
2. Simulating Standalone Particles - EWT.pdf  
3. Simulating Composite Particles - EWT.pdf
4. Simulating Atoms - EWT.pdf
5. Simulating Molecules - EWT.pdf

### Scientific Source Materials

The `requirements/scientific_source/` directory contains foundational EWT research papers:

- a1. The Geometry of Spacetime and the Unification of Forces v2.3.pdf
- a2. The Geometry of Particles and the Explanation of Their Creation and Decay v2.pdf
- a3. The Physics of SubAtomic Particles.pdf
- a4. Relationship of the Speed of Light to Aether Density.docx (contains Planck mass correction)
- a5. The Relationship of Planck Constants and Wave Constants v2.pdf
- a6. Constants and Equations - Waves.pdf (primary constants reference)
- a7. Constants and Equations - Classical.pdf
- a8. Geometry - EWT.pdf
- a9. Mechanics - EWT.pdf

**Key Reference**: File `a6. Constants and Equations - Waves.pdf` serves as the primary constants reference for the simulation, and `a4. Relationship of the Speed of Light to Aether Density` contains important Planck mass corrections that affect granule mass calculations.

## Physics Context

This project implements Energy Wave Theory concepts:

- Quantum waves as fundamental building blocks
- Wave interactions forming particles and matter
- Simulation from Planck scale to macroscopic phenomena
- Validation against experimental observations
