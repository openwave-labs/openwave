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
   openwave.core.quantum_space
   openwave.core.quantum_wave

Simulation Modules
------------------

.. autosummary::
   :toctree: generated
   :recursive:
   :template: custom-module-template.rst

   openwave.core.forces
   openwave.core.heat
   openwave.core.matter
   openwave.core.motion
   openwave.core.photon

Module Dependency Graph
-----------------------

.. graphviz::

   digraph dependencies {
      rankdir=TB;
      node [shape=box, style="rounded,filled", fillcolor=lightblue];
      
      "openwave.core.constants" -> "openwave.core.equations";
      "openwave.core.config" -> "openwave.core.quantum_space";
      "openwave.core.config" -> "openwave.core.quantum_wave";
      "openwave.core.constants" -> "openwave.core.quantum_space";
      "openwave.core.constants" -> "openwave.core.quantum_wave";
      "openwave.core.equations" -> "openwave.core.quantum_wave";
      "openwave.core.quantum_wave" -> "openwave.core.forces";
      "openwave.core.quantum_wave" -> "openwave.core.matter";
      "openwave.core.quantum_wave" -> "openwave.core.photon";
      "openwave.core.matter" -> "openwave.core.motion";
      "openwave.core.matter" -> "openwave.core.heat";
   }

Class Hierarchy
---------------

.. inheritance-diagram:: 
   openwave.core.constants.WaveConstants
   openwave.core.constants.ParticleConstants
   openwave.core.constants.ClassicalConstants
   :parts: 1