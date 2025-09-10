# OpenWave

`>simulate(the_universe)`

[![License](https://img.shields.io/badge/license-MIT-orange.svg?style=for-the-badge)](LICENSE)
[![openwave](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/openwave-labs/openwave)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![openwave](https://img.shields.io/badge/Reddit-%23FF4500.svg?style=for-the-badge&logo=Reddit&logoColor=white)](https://www.reddit.com/r/openwave/)
[![openwavelabs](https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white)](https://x.com/openwavelabs/)
[![openwave-labs](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://youtube.com/@openwave-labs/)

[![OpenWave Logo](images/openwave-white-small.png)](https://openwavelabs.com/)

## What is OpenWave?

OpenWave is an open-source application designed to simulate the formation and behavior of matter and other identities of energy — from the emergence of spacetime and quantum waves, through the creation of subatomic particles, to the development of matter, motion, light, and heat — based on the [Energy Wave Theory (EWT)](https://energywavetheory.com "Energy Wave Theory") model.

## Core Scope

OpenWave provides computational and visualization tools to explore, demonstrate, and validate EWT predictions through three main functions:

### Numerical Validation

- Runs simulations derived directly from equations.
- Validates outcomes by comparing them against observed reality.
- Generates numerical analysis reports for scientific publications.

### Visual Demonstration

- Illustrates complex, often invisible phenomena for better comprehension.
- Represents graphically wave equations and analyses.
- [PLANNED] Automates animation export for online video publishing.

### Experiments Simulation (#energy_hacking)

- [PLANNED] Models experimental conditions to explore new tech derived from subatomic-scale energy exchange simulations.
- [PLANNED] Generates schematics to serve as baseline for patent applications.

## System Architecture

### Modular Design

This diagram illustrates the architecture of the OpenWave system, broken down into the following system modules:

```mermaid
classDiagram
  direction LR
  
class `SPACETIME MODULES
  (ENERGY SOURCE)`{
    quantum_space.py
    quantum_wave.py
  }
class `FORCE
  MODULES`{
    electric.py
    magnetic.py
    gravitational.py
    strong.py
    orbital.py
  }
class `MATTER MODULES
  (PARTICLE ENERGY)`{
    fundamental_particle.py
    standalone_particle.py
    composite_particle.py
    atom.py
    molecule.py
  }
class `MOTION MODULES
  (KINETIC ENERGY)`{    
    --TBD*
  }
class `PHOTON MODULES
  (PHOTON ENERGY)`{
    --TBD*
  }
class `HEAT MODULES
  (THERMAL ENERGY)`{
    --TBD*
  }
class `VALIDATION
  MODULES`{
    derivations.py
    --TBD*
  }
class `XPERIMENT MODULES
  (DIY PHYSICS)`{
    render_granule.py
    --TBD*
  }
class `CORE
  MODULES`{
    config.py
    constants.py
    equations.py
    --TBD*
  }
class `I/O
  MODULES`{
    viz
    cli
    file
    --TBD*
  }

`SPACETIME MODULES
  (ENERGY SOURCE)` --> `MATTER MODULES
  (PARTICLE ENERGY)`
  
`SPACETIME MODULES
  (ENERGY SOURCE)` --> `MOTION MODULES
  (KINETIC ENERGY)`

`SPACETIME MODULES
  (ENERGY SOURCE)` --> `PHOTON MODULES
  (PHOTON ENERGY)`

`SPACETIME MODULES
  (ENERGY SOURCE)` --> `HEAT MODULES
  (THERMAL ENERGY)`
  
`MATTER MODULES
  (PARTICLE ENERGY)` <--> `FORCE
  MODULES`
  
`MOTION MODULES
  (KINETIC ENERGY)` <-- `FORCE
  MODULES`
  
`MATTER MODULES
  (PARTICLE ENERGY)` --> `XPERIMENT MODULES
  (DIY PHYSICS)`
  
`MOTION MODULES
  (KINETIC ENERGY)` --> `XPERIMENT MODULES
  (DIY PHYSICS)`
  
`PHOTON MODULES
  (PHOTON ENERGY)` --> `XPERIMENT MODULES
  (DIY PHYSICS)`
  
`HEAT MODULES
  (THERMAL ENERGY)` --> `XPERIMENT MODULES
  (DIY PHYSICS)`
  
`CORE
  MODULES` <--> `VALIDATION
  MODULES`

`VALIDATION
  MODULES` <--> `I/O
  MODULES`
```

### Scalability & Performance

- Support increasing simulation resolution to handle extreme granularity of Planck-scale interactions
- Efficient handling of large particle counts and ultra-small wavelength resolution
- GPU optimized parallel processing for computational performance

### Tech Stack

- **Primary Language**: Python (>=3.12)
- **Parallel Processing**:
  - Taichi Python Acceleration: GPU optimization for computationally intensive wave simulations
- **Math/Physics Libraries**:
  - NumPy, SciPy
- **Visualization**:
  - Taichi: 3D rendering
  - Matplotlib: numerical analysis plots and cross-sectional graphs
  - Export of 3D images and GIFs for visual inspection
- **Data Output**:
  - Numerical datasets, graphs, and analysis reports in open formats (CSV, JSON, PNG, STL)

## Installation

### On Linux / macOS (conda)

```bash
# Clone the repository
git clone https://github.com/openwave-labs/openwave.git

# Create virtual environment
conda create -n openwave312 python=3.12 -y
conda activate openwave312

# Install dependencies
cd openwave
pip install .  # reads dependencies from pyproject.toml

# IF issues, remove virtual environment and start over again
conda env remove -n openwave312
```

### On Windows *--WORK IN PROGRESS--*

```bash
TBD
```

### Optional: LaTex & FFmpeg (video generation)

```bash
# Install LaTeX and FFmpeg (macOS)
brew install --cask mactex-no-gui ffmpeg
echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
exec zsh -l

# Verify LaTeX installation
which latex && latex --version
which dvisvgm && dvisvgm --version
which gs && gs --version
```

## CLI Usage *--WORK IN PROGRESS--*

Note: CLI implementations are currently incomplete but will follow the pattern below.

### Quantum Space

- Use `space -h` to get the following help or `python openwave/space.py -h`

```bash
usage: space [-h] [-R] [-t]
             [-p PREFIX]
             [-asis] [-dnr] [-env] [-run]

optional arguments:
      -h, --help                    show this help message and exit
      -R, --use-ray
      -t, --test
      -p PREFIX, --prefix PREFIX    modifies name, useful for report folder customization
      -asis, --execute-as-is
      -dnr, --dont-refresh          refresh
      -env, --print-env-var         prints all environmental variables
      -run, --run-analysis          runs at selected locations
```

### Quantum Wave

- Use `wave -h` to get the following help or `python openwave/wave.py -h`

```bash
usage: wave [-h] [-R] [-t]
             [-p PREFIX]
             [-asis] [-dnr] [-env] [-run]

optional arguments:
      -h, --help                    show this help message and exit
      -R, --use-ray
      -t, --test
      -p PREFIX, --prefix PREFIX    modifies name, useful for report folder customization
      -asis, --execute-as-is
      -dnr, --dont-refresh          refresh
      -env, --print-env-var         prints all environmental variables
      -run, --run-analysis          runs at selected locations
```

## Todo

- [ ] Implement CLI entry points
- [ ] Develop documentation
- [ ] Define pre-commit hooks and style enforcement tools to ensure consistent formatting
- [ ] Introduce automated testing and continuous integration to validate code changes

## Contributing to this Project

Please refer to the [Contribution Guide](CONTRIBUTING.md)

See `/dev_docs` for coding standards and development guidelines

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

Real human power comes from collaboration.
