OpenWave API Reference
======================

This section contains the complete API documentation for the OpenWave quantum physics simulator.

Core Modules
------------

.. autosummary::
   :toctree: generated
   :recursive:
   :template: custom-module-template.rst

   openwave.core.constants
   openwave.core.config
   openwave.core.equations

Spacetime Modules
-----------------

.. autosummary::
   :toctree: generated
   :recursive:
   :template: custom-module-template.rst

   openwave.spacetime.quantum_space
   openwave.spacetime.quantum_wave

Simulation Modules
------------------

.. autosummary::
   :toctree: generated
   :recursive:
   :template: custom-module-template.rst

   openwave.force
   openwave.heat
   openwave.matter
   openwave.motion
   openwave.photon

Module Dependency Graph
-----------------------

.. graphviz::

   digraph dependencies {
      rankdir=TB;
      node [shape=box, style="rounded,filled", fillcolor=lightblue];
      
      "openwave.core.constants" -> "openwave.core.equations";
      "openwave.core.config" -> "openwave.spacetime.quantum_space";
      "openwave.core.config" -> "openwave.spacetime.quantum_wave";
      "openwave.core.constants" -> "openwave.spacetime.quantum_space";
      "openwave.core.constants" -> "openwave.spacetime.quantum_wave";
      "openwave.core.equations" -> "openwave.spacetime.quantum_wave";
      "openwave.spacetime.quantum_wave" -> "openwave.force";
      "openwave.spacetime.quantum_wave" -> "openwave.matter";
      "openwave.spacetime.quantum_wave" -> "openwave.photon";
      "openwave.matter" -> "openwave.motion";
      "openwave.matter" -> "openwave.heat";
   }

