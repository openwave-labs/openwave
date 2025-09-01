# MODULAR STRUCTURE & OBJECTS MAP

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

`SPACETIME MODULES
  (ENERGY SOURCE)` --> `FORCE
  MODULES`
  
`MATTER MODULES
  (PARTICLE ENERGY)` <--> `FORCE
  MODULES`
  
`MOTION MODULES
  (KINETIC ENERGY)` <-- `FORCE
  MODULES`
  
`CORE
  MODULES` <--> `I/O
  MODULES`
```

## Configuration System

Configuration is managed through `config.py` with sections for:

- `universe`: Simulation parameters (size, time_step)
- `screen`: Display resolution settings
- `color`: Color scheme for different physics entities (space, quantum_waves, matter, antimatter, motion, photons, energy, heat)

### Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing and analysis
- **matplotlib**: 2D plotting and visualization
- **taichi**: High-performance parallel computing for simulations

### Constants Usage

- Import from `openwave.constants` for all physics constants
- Use EWT-specific constants (QWAVE_LENGTH, QWAVE_AMPLITUDE, etc.) for wave modeling
- Classical constants are provided for compatibility and validation
- All constants use SI units (kg, m, s)

### Configuration Access

- Load configuration via `openwave.config` module
- Access screen dimensions: `config.screen_width`, `config.screen_height`
- Configuration file: `openwave/config.ini`
