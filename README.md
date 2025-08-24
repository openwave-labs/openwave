# OpenWave

`>simulate(the_universe)`

[![License](https://img.shields.io/badge/license-MIT-orange.svg?style=for-the-badge)](LICENSE)
[![openwaveHQ](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/openwaveHQ/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![openwave](https://img.shields.io/badge/Reddit-%23FF4500.svg?style=for-the-badge&logo=Reddit&logoColor=white)](https://www.reddit.com/r/openwave/)
[![openwave_HQ](https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white)](https://x.com/openwave_HQ/)
[![openwaveHQ](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://youtube.com/@openwaveHQ/)

[![OpenWave Logo](images/openwave-white-small.png)](https://openwavehq.com/)

## What is OpenWave?

OpenWave is an open-source application designed to simulate the formation and behavior of matter and other identities of energy — from the emergence of spacetime and quantum waves, through the creation of subatomic particles, to the development of matter, motion, light, and heat — based on the [Energy Wave Theory (EWT)](https://energywavetheory.com "Energy Wave Theory") model.

## Core Scope

OpenWave provides computational and visualization tools to explore, demonstrate, and validate EWT predictions through three main functions:

### Visual Demonstration

- Illustrates complex, often invisible phenomena for better comprehension.  
- Represents graphically wave equations and analyses.  
- Automates animation export for online video publishing.

### Numerical Validation

- Runs simulations derived directly from equations.  
- Validates outcomes by comparing them against observed reality.  
- Generates numerical analysis reports for scientific publications.

### *--PLANNED--* Experiments Simulation (*#energy_hacking*)

- Models experimental conditions to explore new tech derived from subatomic-scale energy exchange simulations.  
- Generates schematics to serve as baseline for patent applications.

## Installation *--WORK IN PROGRESS--*

### On macOS / conda

#### Clone the repository

```bash
git clone https://github.com/openwaveHQ/openwave.git
cd openwave
```

#### Install [HOMEBREW](https://brew.sh/)

```bash
/bin/bash -c `$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)`
```

#### Config ENV

Dependencies are defined in `pyproject.toml`.

```bash
# Create virtual environment
conda create -n openwave312 python=3.12 -y
conda activate openwave312

# Install dependencies
pip install .  # reads dependencies from pyproject.toml
brew install --cask mactex-no-gui ffmpeg

# IF issues, remove virtual environment and start over again
conda env remove -n openwave312
```

#### LaTEX path config & sanity check

```bash
echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
exec zsh -l
which latex && latex --version
which dvisvgm && dvisvgm --version
which gs && gs --version
```

### On Linux

```bash
TBD
```

### On Windows

```bash
TBD
```

## CLI Usage *--WORK IN PROGRESS--*

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

## System Architecture

### Modular Design

- Core physics engine (wave propagation, granule interaction, particle formation)
- Visualization module (3D, cross-sections, mesh generator)
- Data analysis module (plots, numerical analysis, reports)
- Configuration layer (user parameters, simulation scale, constants)

### System Modules

This diagram illustrates the architecture of the OpenWave system, broken down into the following categories:

```mermaid
classDiagram
  direction LR
  class `BASE
  PACKAGES`{
    numpy
    scipy
    matplotlib
    taichi
  }
  class `CORE
  ENGINES`{
    Wave Engine
    Elastic Engine
  }
  class `SPACETIME
  MODULES`{
    Quantum Space
    Quantum Wave
  }
  class `ENERGY
  MODULES`{
    Matter
    Motion
    Photon
    Heat
  }
  class `I/O`{
    Configuration Layer
    Numerical Analysis
    Data Visualization
    File Export
  }
  `BASE
  PACKAGES` --> `CORE
  ENGINES`
  `CORE
  ENGINES` --> `SPACETIME
  MODULES`
  `SPACETIME
  MODULES` --> `ENERGY
  MODULES`
  `ENERGY
  MODULES` --> `I/O`
```

### Scalability

- Support increasing simulation resolution to handle extreme granularity of Planck-scale interactions
- Efficient handling of large particle counts and ultra-small wavelength resolution
- Distribute computation across clusters if needed

### Tech Stack

- **Primary Language**: Python (>=3.12)
- **Parallel Processing**:
  - Multi-CPU/GPU utilization for computationally intensive wave simulations
  - Ray.io python package for distributed task management
- **Math/Physics Libraries**: NumPy, SciPy, SymPy
- **Visualization**:
  - 3D rendering with Taichi, PyVista, VTK, or OpenGL-based solutions
  - Matplotlib/Plotly for plots and cross-sectional graphs
  - Export of 3D meshes and GIFs for visual inspection
- **Data Output**:
  - Numerical datasets, graphs, and analysis reports in open formats (CSV, JSON, PNG, STL)

## Todo

- [ ] Develop Front-end UX & GUI
- [ ] Develop fuller documentation
- [ ] Define pre-commit hooks and style enforcement tools to ensure consistent formatting
- [ ] Introduce automated testing and continuous integration to validate code changes

## Contributing to this Project

Please refer to the [Contribution Guide](CONTRIBUTING.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Credits Timeline

Real human power comes from collaboration.

| Public Legacy             | Credit                                            | Work |
| :---                      | :---                                              | :--- |
| Wave Structure of Matter  | Dr. Milo Wolff, Gabriel LaFreniere                | Pioneer models (books, webpages) |
| Energy Wave Theory        | Jeff Yee                                          | Physics modeling, numerical validation and initial simulator design (research papers, books, webpages and videos) |
| OpenWave Simulator        | Rodrigo Griesi & Jeff Yee                         | Initial work on a programmatic validation, visualization and experimentation tool for the Energy Wave Theory (open-source computer based simulator) |
| OpenWave Simulator        | development community (coming soon)               | Continuous development                        |
| Open-Source Libraries     | numpy, scipy, matplotlib, taichi, ffmpeg          | Development of open-source software packages by multiple communities (open-source libraries) |
| Others to come...         |                                                   |  |
