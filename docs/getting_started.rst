Getting Started
===============

This guide will help you install and start using OpenWave.

Installation
------------

Clone the Repository
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Clone the OpenWave repository
   git clone https://github.com/openwave-labs/openwave.git
   cd openwave

Set Up Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure you have Python >=3.12 installed. We recommend using Anaconda:

.. code-block:: bash

   # Create a conda environment (recommended)
   conda create -n openwave python=3.12
   conda activate openwave

Install OpenWave
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Install OpenWave package & dependencies
   pip install .

Quick Start
-----------

Play with Xperiments
^^^^^^^^^^^^^^^^^^^^

XPERIMENTS are virtual lab scripts where you can play with subatomic objects and simulate desired outcomes.

.. code-block:: bash

   # Launch xperiments using the CLI xperiment selector
   openwave -x

   # Run sample xperiments shipped with the OpenWave package, tweak them, or create your own

Available Xperiments
^^^^^^^^^^^^^^^^^^^^

The OpenWave package includes several sample experiments in the ``/xperiments`` module:

**Granule-Based Medium:**

- ``pulse.py`` - Pulse wave simulation
- ``radial_wave.py`` - Radial wave propagation
- ``spherical_wave.py`` - Spherical wave simulation
- ``spring_mass.py`` - Spring-mass system (demonstrates instability challenges)
- ``xwaves.py`` - X-waves simulation

**Field-Based Medium:**

- ``flow_wave.py`` - Flow wave simulation (WIP)

**Anti-Gravity:**

- ``proton_vibration.py`` - Proton vibration simulation (WIP)

**Heat Dynamics:**

- ``heat_waves.py`` - Heat wave simulation (WIP)

Next Steps
----------

For more information:

- See the :doc:`architecture` for system design details
- Check the :doc:`physics_guide` for physics concepts
- Read the :doc:`contributing` guide to contribute to the project
- Explore the :doc:`api/modules` for detailed API documentation
