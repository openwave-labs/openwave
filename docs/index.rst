OpenWave Documentation
======================

.. image:: https://img.shields.io/badge/License-AGPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/
   :alt: License

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/
   :alt: Python Version

Welcome to OpenWave's documentation!

**OpenWave** is an open-source computational physics toolkit for modeling matter and energy phenomena
using wave field dynamics. The project implements the mathematical framework of
`Energy Wave Theory (EWT) <https://energywavetheory.com>`_ through two complementary computational approaches:

- **Level-1:** Field-based method (similar to lattice gauge theory) for research simulations
- **Level-0:** Granule-based method for educational visualization

.. note::
   OpenWave is currently in active development (v0.3.0). The API may change in future releases.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/openwave-labs/openwave.git
   cd openwave

   # Create conda environment (recommended)
   conda create -n openwave python=3.12
   conda activate openwave

   # Install OpenWave
   pip install .

Basic Usage
~~~~~~~~~~~

Launch the xperiments module to explore wave mechanics:

.. code-block:: bash

   # Launch xperiments using the CLI selector
   openwave -x

Key Features
------------

✅ **Computational Approaches**
   - Field-based medium (PDE-based, scalable for research)
   - Granule-based medium (intuitive for education)

✅ **GPU-Accelerated**
   - Taichi-powered parallel processing
   - Efficient large-scale simulations

✅ **Interactive Visualization**
   - Real-time 3D rendering
   - Export animations and data

✅ **Modular Architecture**
   - Spacetime, Matter, Force, Motion modules
   - Extensible design for research

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   installation
   usage
   xperiments

.. toctree::
   :maxdepth: 2
   :caption: Theory & Background

   theory/energy_wave_theory
   theory/computational_approaches
   theory/relationship_to_physics

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/openwave
   api/spacetime
   api/common
   api/io
   api/validations

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   dev/coding_standards
   dev/performance
   dev/architecture

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   license
   changelog
   references

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Community & Support
===================

- **GitHub:** https://github.com/openwave-labs/openwave
- **Reddit:** https://www.reddit.com/r/openwave/
- **YouTube:** https://youtube.com/@openwave-labs/
- **Website:** https://openwavelabs.com/

License
-------

OpenWave is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

See the :doc:`license` page for details.
