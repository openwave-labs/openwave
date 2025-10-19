OpenWave Documentation
======================

**OpenWave** is an open-source quantum physics simulator implementing Energy Wave Theory (EWT) 
to model the formation of matter and energy from quantum wave interactions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/modules
   architecture
   physics_guide
   contributing

Features
--------

* **Quantum Wave Simulation**: Model wave interactions at the Planck scale
* **Particle Formation**: Simulate the emergence of subatomic particles
* **Matter Behavior**: Track complex matter interactions
* **Visual Demonstration**: Real-time visualization of quantum phenomena
* **Numerical Validation**: Compare simulations against experimental data

Quick Start
-----------

.. code-block:: bash

   # Clone the OpenWave repository
   git clone https://github.com/openwave-labs/openwave.git
   cd openwave

   # Create conda environment (recommended, Python >=3.12)
   conda create -n openwave python=3.12
   conda activate openwave

   # Install OpenWave package & dependencies
   pip install .

   # Launch xperiments using the CLI xperiment selector
   openwave -x

For detailed installation and usage instructions, see :doc:`getting_started`.

API Reference
-------------

.. autosummary::
   :toctree: api
   :recursive:

   openwave

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
