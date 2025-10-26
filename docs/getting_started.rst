Getting Started
===============

Welcome to OpenWave! This guide will help you get up and running quickly.

Prerequisites
-------------

Before installing OpenWave, ensure you have:

- **Python 3.12 or higher**
- **Git** (for cloning the repository)
- **Conda** (recommended for environment management)
- **GPU with CUDA support** (optional, but recommended for performance)

Installation
------------

Step 1: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/openwave-labs/openwave.git
   cd openwave

Step 2: Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Conda (Recommended):

.. code-block:: bash

   conda create -n openwave python=3.12
   conda activate openwave

Using venv:

.. code-block:: bash

   python -m venv openwave-env
   source openwave-env/bin/activate  # On Windows: openwave-env\Scripts\activate

Step 3: Install OpenWave
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install .

This will install OpenWave and all required dependencies from ``pyproject.toml``.

Verify Installation
-------------------

Test that OpenWave is installed correctly:

.. code-block:: bash

   openwave --help

You should see the OpenWave command-line interface help message.

Your First Xperiment
---------------------

Launch the xperiments selector to explore wave mechanics:

.. code-block:: bash

   openwave -x

This will open an interactive menu where you can select from various pre-built experiments:

- **Medium Vibration**: Observe fundamental medium oscillations
- **Spherical Wave**: Watch wave propagation in 3D
- **Standing Wave**: See wave interference patterns
- **The Pulse**: Explore energy transfer
- **X-Waves**: Investigate wave superposition

Example: Running Standing Wave
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openwave.spacetime.medium_level0 import BCCLattice
   from openwave.spacetime.energy_wave_level0 import StandingWave
   from openwave.common import constants
   import taichi as ti

   # Initialize Taichi
   ti.init(arch=ti.gpu)

   # Create simulation universe
   UNIVERSE_SIZE = [
       4 * constants.EWAVE_LENGTH,
       4 * constants.EWAVE_LENGTH,
       4 * constants.EWAVE_LENGTH
   ]

   # Initialize medium
   lattice = BCCLattice(UNIVERSE_SIZE, theme="OCEAN")

   # Create standing wave
   wave = StandingWave(lattice)

   # Run simulation (from render module)
   wave.simulate()

Next Steps
----------

- Read the :doc:`usage` guide for detailed examples
- Explore :doc:`xperiments` for available experiments
- Check the :doc:`api/openwave` for API reference
- Review :doc:`theory/energy_wave_theory` to understand the physics

Common Issues
-------------

GPU Not Found
~~~~~~~~~~~~~

If Taichi can't find your GPU, it will fall back to CPU mode:

.. code-block:: python

   import taichi as ti
   ti.init(arch=ti.cpu)  # Force CPU mode

Import Errors
~~~~~~~~~~~~~

If you encounter import errors, ensure you're in the correct environment:

.. code-block:: bash

   conda activate openwave
   pip list | grep OPENWAVE

Dependencies Issues
~~~~~~~~~~~~~~~~~~~

If dependency installation fails, try updating pip:

.. code-block:: bash

   pip install --upgrade pip
   pip install .

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/openwave-labs/openwave/issues>`_
2. Read the :doc:`contributing` guide
3. Join the discussion on `Reddit <https://www.reddit.com/r/openwave/>`_
4. Watch tutorials on `YouTube <https://youtube.com/@openwave-labs/>`_
