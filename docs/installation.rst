Installation Guide
==================

This guide covers various installation methods for OpenWave.

System Requirements
-------------------

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

**Minimum:**

- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8 GB
- Storage: 2 GB free space
- GPU: Not required but highly recommended

**Recommended:**

- CPU: 8+ core processor
- RAM: 16+ GB
- Storage: 5+ GB free space
- GPU: NVIDIA GPU with CUDA support (for GPU acceleration)

Software Requirements
~~~~~~~~~~~~~~~~~~~~~

- **Python:** 3.12 or higher
- **Operating Systems:** Linux, macOS, Windows
- **Optional:** Conda/Miniconda for environment management

Installation Methods
--------------------

Method 1: Standard Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended method for most users.

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/openwave-labs/openwave.git
   cd openwave

   # Create and activate conda environment
   conda create -n openwave python=3.12
   conda activate openwave

   # Install OpenWave
   pip install .

Method 2: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For developers who want to modify the source code:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/openwave-labs/openwave.git
   cd openwave

   # Create and activate conda environment
   conda create -n openwave python=3.12
   conda activate openwave

   # Install in editable mode
   pip install -e .

This allows you to modify the source code without reinstalling.

Method 3: Installation with Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install with development tools
   pip install -e ".[dev]"

   # Install with documentation tools
   pip install -e ".[docs]"

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

OpenWave requires the following packages (automatically installed):

- **numpy** (>=2.3, <3.0): Numerical computing
- **matplotlib** (>=3.8, <4.0): Plotting and visualization
- **taichi** (>=1.6, <2.0): GPU acceleration and rendering
- **pyautogui** (>=0.9, <1.0): GUI automation
- **simple-term-menu** (>=1.6.0, <2.0): CLI menu interface

Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For building documentation (see ``docs/requirements.txt``):

.. code-block:: bash

   pip install -r docs/requirements.txt

This includes:

- sphinx
- sphinx-rtd-theme
- sphinx-autodoc-typehints
- sphinx-copybutton
- myst-parser

Verifying Installation
----------------------

Check Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Verify OpenWave is installed
   openwave --help

   # Check Python version
   python --version  # Should be 3.12 or higher

   # Verify Taichi GPU support
   python -c "import taichi as ti; ti.init(arch=ti.gpu); print('GPU support: OK')"

Test Installation
~~~~~~~~~~~~~~~~~

Run a quick smoke test:

.. code-block:: python

   from openwave.spacetime.medium_level0 import BCCLattice
   from openwave.common import constants
   import taichi as ti

   ti.init(arch=ti.gpu)

   UNIVERSE_SIZE = [
       2 * constants.EWAVE_LENGTH,
       2 * constants.EWAVE_LENGTH,
       2 * constants.EWAVE_LENGTH
   ]

   lattice = BCCLattice(UNIVERSE_SIZE)
   print(f"Lattice created with {lattice.total_granules:,} granules")
   print("Installation successful!")

GPU Configuration
-----------------

NVIDIA CUDA
~~~~~~~~~~~

For NVIDIA GPUs, ensure CUDA is installed:

.. code-block:: bash

   # Check CUDA version
   nvidia-smi

   # Verify Taichi can use CUDA
   python -c "import taichi as ti; ti.init(arch=ti.cuda)"

AMD ROCm
~~~~~~~~

For AMD GPUs (experimental):

.. code-block:: bash

   # Taichi with Vulkan backend
   python -c "import taichi as ti; ti.init(arch=ti.vulkan)"

CPU Fallback
~~~~~~~~~~~~

If no GPU is available, Taichi will use CPU:

.. code-block:: python

   import taichi as ti
   ti.init(arch=ti.cpu)

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue:** ``pip install .`` fails with dependency conflicts

**Solution:** Update pip and try again:

.. code-block:: bash

   pip install --upgrade pip setuptools wheel
   pip install .

**Issue:** Taichi installation fails

**Solution:** Install Taichi separately first:

.. code-block:: bash

   pip install taichi==1.7.1
   pip install .

**Issue:** GPU not detected

**Solution:** Check CUDA installation and drivers:

.. code-block:: bash

   nvidia-smi  # Should show GPU info
   python -c "import taichi as ti; print(ti.gpu_available())"

Platform-Specific Notes
-----------------------

macOS
~~~~~

On macOS, you may need to install Xcode Command Line Tools:

.. code-block:: bash

   xcode-select --install

For Apple Silicon (M1/M2), use native Python:

.. code-block:: bash

   # Use conda with osx-arm64
   conda create -n openwave python=3.12
   conda activate openwave
   pip install .

Windows
~~~~~~~

On Windows, you may need Visual C++ Build Tools:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"

Linux
~~~~~

On Linux, you may need development packages:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-dev build-essential

   # Fedora/RHEL
   sudo dnf install python3-devel gcc

Uninstalling
------------

To uninstall OpenWave:

.. code-block:: bash

   pip uninstall OPENWAVE

To remove the conda environment:

.. code-block:: bash

   conda deactivate
   conda env remove -n openwave

Next Steps
----------

- Continue to :doc:`getting_started` for your first experiment
- Read :doc:`usage` for detailed examples
- Check :doc:`contributing` if you want to contribute
