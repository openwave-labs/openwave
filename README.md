# OpenWave

`>simulate(the_universe)`

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![openwave](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/openwave-labs/openwave)
[![openwave](https://img.shields.io/badge/Reddit-%23FF4500.svg?style=for-the-badge&logo=Reddit&logoColor=white)](https://www.reddit.com/r/openwave/)
[![openwavelabs](https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white)](https://x.com/openwavelabs/)
[![openwave-labs](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://youtube.com/@openwave-labs/)

[![OpenWave Logo](images/openwave-white-small.png)](https://openwavelabs.com/)

## What is OpenWave?

OpenWave is an open-source application designed to simulate the formation and behavior of matter and other identities of energy — from the emergence of spacetime and quantum waves, through the creation of subatomic particles, to the development of matter, motion, light, and heat — based on the deterministic quantum mechanics model: [Energy Wave Theory (EWT)](https://energywavetheory.com "Energy Wave Theory").

## Core Scope

OpenWave provides computational and visualization tools to explore, demonstrate, and validate EWT predictions through three main functions:

### Numerical Validation

- Runs simulations derived directly from equations.
- Validates outcomes by comparing them against observed reality.
- [PLANNED] Generates numerical analysis reports for scientific publications.

### Visual Demonstration

- Illustrates complex, often invisible phenomena for better comprehension.
- Represents graphically wave equations and analyses.
- [PLANNED] Automates animation export for online video publishing.

### Experiments Simulation (#energy_hacking)

- Models experimental conditions to explore new tech derived from subatomic-scale energy exchange simulations.
- [PLANNED] Generates baseline knowledge for your patent applications.

## Render SPACETIME in 3D

![demo clip](images/demo2.gif)

## Scientific Source

OpenWave is a programmatic computing and rendering package based on the [Energy Wave Theory (EWT)](https://energywavetheory.com "Energy Wave Theory") model.

Prior to using and contributing to OpenWave, it is recommended to study and familiarize yourself with this interpretation of quantum mechanics from the following resources:

- Website & Videos: [Energy Wave Theory (EWT)](https://energywavetheory.com "Energy Wave Theory")
- Scientific Papers: [Core Concepts](https://github.com/openwave-labs/openwave/tree/main/research_requirements/scientific_source "Energy Wave Theory")
- Original Requirements: [Requirements](https://github.com/openwave-labs/openwave/tree/main/research_requirements/original_requirements "Energy Wave Theory")

### Origins

The [Energy Wave Theory (EWT)](https://energywavetheory.com "Energy Wave Theory") is a deterministic quantum mechanics model designed by [Jeff Yee](https://www.youtube.com/@EnergyWaveTheory) that builds upon the work of pioneers like:

- [Albert Einstein](https://en.wikipedia.org/wiki/Einstein%E2%80%93Podolsky%E2%80%93Rosen_paradox)
- [Louis de Broglie](https://en.wikipedia.org/wiki/Pilot_wave_theory)
- [Dr. Milo Wolff](https://www.amazon.com/dp/0962778710)
- [Gabriel LaFreniere](http://www.rhythmodynamics.com/Gabriel_LaFreniere/matter.htm)
- among others.

>*"Quantum mechanics is very worthy of respect. But an inner voice tells me this is not the genuine article after all. The theory delivers much but it hardly brings us closer to the Old One's secret. In any event, I am convinced that He is not playing dice."*
>>Albert Einstein (December 4, 1926), challenging the adoption of a probabilistic interpretation to quantum mechanics, arguing that the description of physical reality provided was incomplete.

## System Architecture

### Modular Design

This diagram illustrates the architecture of the OpenWave system, broken down into the following system modules:

- ✓ = module already released

```mermaid
classDiagram
  direction LR
  
class `SOURCE MODULE
  (ENERGY SOURCE)`{
    spacetime.py ✓
    quantum_wave.py}
`SOURCE MODULE
  (ENERGY SOURCE)` --> `MATTER MODULE
  (PARTICLE ENERGY)`
`SOURCE MODULE
  (ENERGY SOURCE)` --> `MOTION MODULE
  (KINETIC ENERGY)`
`SOURCE MODULE
  (ENERGY SOURCE)` --> `PHOTON MODULE
  (PHOTON ENERGY)`
`SOURCE MODULE
  (ENERGY SOURCE)` --> `HEAT MODULE
  (THERMAL ENERGY)`


class `MATTER MODULE
  (PARTICLE ENERGY)`{
    fundamental_particle.py
    standalone_particle.py
    composite_particle.py
    atom.py
    molecule.py}
`MATTER MODULE
  (PARTICLE ENERGY)` <--> `FORCE
  MODULE`
`MATTER MODULE
  (PARTICLE ENERGY)` --> `XPERIMENTS MODULE
  (VIRTUAL BENCH)`


class `FORCE
  MODULE`{
    electric.py
    magnetic.py
    gravitational.py
    strong.py
    orbital.py}


class `MOTION MODULE
  (KINETIC ENERGY)`{    
    --TBD*}
`MOTION MODULE
  (KINETIC ENERGY)` <-- `FORCE
  MODULE`
`MOTION MODULE
  (KINETIC ENERGY)` --> `XPERIMENTS MODULE
  (VIRTUAL BENCH)`


class `PHOTON MODULE
  (PHOTON ENERGY)`{
    --TBD*}
`PHOTON MODULE
  (PHOTON ENERGY)` --> `XPERIMENTS MODULE
  (VIRTUAL BENCH)`


class `HEAT MODULE
  (THERMAL ENERGY)`{
    --TBD*}
`HEAT MODULE
  (THERMAL ENERGY)` --> `XPERIMENTS MODULE
  (VIRTUAL BENCH)`


class `XPERIMENTS MODULE
  (VIRTUAL BENCH)`{
    qwave_render.py ✓
    --TBD*}


class `COMMON
  MODULE`{
    config.py ✓
    constants.py ✓
    equations.py ✓
    render.py ✓
    --TBD*}
`COMMON
  MODULE` <--> `VALIDATIONS
  MODULE`


class `VALIDATIONS
  MODULE`{
    derivations.py ✓
    --TBD*}
`VALIDATIONS
  MODULE` <--> `I/O
  MODULE`


class `I/O
  MODULE`{
    CLI
    file_export
    video_manager
    --TBD*}
```

### DEVELOPMENT ROADMAP

```mermaid
kanban
  [BACKLOG]
    [**MATTER MODULE**
      - stdalone_particle.py
      - comp_particle.py
      - atom.py
      - molecule.py]
    
    [**FORCE MODULE**
      - electric.py
      - magnetic.py
      - gravitational.py
      - strong.py
      - orbital.py]
    
    [**MOTION MODULE**
      - motion.py]
    
    [**PHOTON MODULE**
      - light.py]
    
    [**HEAT MODULE**
      - heat.py]

  [NEXT]
    [**MATTER MODULE**
      - fundam_particle.py]@{ priority: 'High', assigned: 'xrodz' }
    
  [IN PROGRESS]
    [**SOURCE MODULE**
      - quantum_wave.py]@{ priority: 'Very High', assigned: 'xrodz' }
    [**COMMON MODULE**
      - equations.py]@{ priority: 'Very Low', assigned: 'xrodz' }
    
  [RELEASED]
    [**SOURCE MODULE**
      - spacetime.py]
    [**XPERIMENTS MODULE**
      - qwave_render.py]
    [**VALIDATIONS MODULE**
      - derivations.py]
    [**COMMON MODULE**
      - config.py
      - constants.py
      - render.py]
```

### Scalability & Performance

- Support increasing simulation resolution to handle extreme granularity of Planck-scale interactions
- Efficient handling of large particle counts and ultra-small wavelength resolution
- GPU optimized parallel processing for computational performance

### Tech Stack

- **Primary Language**:
  - Python (>=3.12)
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

### Todo

- [ ] Implement CLI entry points
- [ ] Develop documentation
- [ ] Define pre-commit hooks and style enforcement tools to ensure consistent formatting
- [ ] Introduce automated testing and continuous integration to validate code changes

## Installation

For development installation refer to [Contribution Guide](CONTRIBUTING.md)

```bash
# Clone the OpenWave repository
  git clone https://github.com/openwave-labs/openwave.git
  cd openwave # point to local directory where OpenWave was installed

# Create virtual environment (via Venv)
  python -m venv openwave
  source openwave/bin/activate  # On Windows: openwave\Scripts\activate
   
# Or Create virtual environment (via Conda)
  conda create -n openwave python=3.12 -y
  conda activate openwave

# Install OpenWave package & dependencies
  pip install .  # reads dependencies from pyproject.toml
```

## Usage

### Play with the /xperiments module

Xperiments are virtual bench scripts where you can experiment with quantum objects and simulate desired outcomes.

```bash
# Run your first OpenWave xperiment
  python openwave/xperiments/qwave_render.py

# Run sample xperiments shipped with the OpenWave package, tweak them, or create your own
```

Note: CLI entry points are in the development roadmap, if you want to contribute, please check how at the [Contribution Guide](CONTRIBUTING.md)

## Wanna Contribute to this Project?

- Please read the [Contribution Guide](CONTRIBUTING.md)
- See `/dev_docs` for coding standards and development guidelines
  - [Coding Standards](dev_docs/CODING_STANDARDS.md)
  - [Performance Guidelines](dev_docs/PERFORMANCE_GUIDELINES.md)
  - [Loop Optimization Patterns](dev_docs/LOOP_OPTIMIZATION.md)
  - [Markdown Style Guide](dev_docs/MARKDOWN_STYLE_GUIDE.md)  
- **This is the Way!** ... Real human power comes from collaboration.
