Coding Standards
================

.. note::
   For detailed coding standards, see ``dev_docs/CODING_STANDARDS.md`` in the repository.

This page summarizes key coding conventions for OpenWave.

Code Style
----------

PEP 8 with Modifications
~~~~~~~~~~~~~~~~~~~~~~~~~

We follow PEP 8 with these modifications:

- **Line Length:** 99 characters (Black default)
- **String Quotes:** Double quotes preferred
- **Imports:** Absolute imports preferred

Black Formatter
~~~~~~~~~~~~~~~

We use Black for consistent formatting:

.. code-block:: bash

   # Format code
   black openwave/

   # Check formatting
   black --check openwave/

Type Hints
~~~~~~~~~~

Use type hints for function signatures:

.. code-block:: python

   def create_lattice(
       universe_size: list[float],
       theme: str = "OCEAN"
   ) -> BCCLattice:
       """Create a BCC lattice."""
       pass

Docstrings
----------

Google Style
~~~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def calculate_energy(lattice: BCCLattice) -> float:
       """Calculate total energy in the lattice.

       Args:
           lattice: The BCC lattice to analyze.

       Returns:
           Total energy in Joules.

       Raises:
           ValueError: If lattice is invalid.

       Example:
           >>> lattice = BCCLattice([1e-17, 1e-17, 1e-17])
           >>> energy = calculate_energy(lattice)
       """
       pass

Module Docstrings
~~~~~~~~~~~~~~~~~

All modules should have docstrings:

.. code-block:: python

   """
   Module: openwave.spacetime.medium_level0

   Implements granule-motion wave medium using BCC and SC lattices.
   """

Naming Conventions
------------------

Variables and Functions
~~~~~~~~~~~~~~~~~~~~~~~

- **snake_case** for variables and functions
- Descriptive names (avoid single letters except loops)

.. code-block:: python

   # Good
   total_energy = calculate_total_energy(lattice)

   # Avoid
   e = calc(l)

Classes
~~~~~~~

- **PascalCase** for class names

.. code-block:: python

   class BCCLattice:
       pass

Constants
~~~~~~~~~

- **UPPER_SNAKE_CASE** for constants

.. code-block:: python

   PLANCK_LENGTH = 1.616255e-35  # meters

Private Members
~~~~~~~~~~~~~~~

- Prefix with single underscore

.. code-block:: python

   def _internal_helper():
       pass

Taichi Conventions
------------------

Kernels
~~~~~~~

Mark GPU functions with ``@ti.kernel``:

.. code-block:: python

   @ti.kernel
   def update_positions(dt: float):
       for i in range(granule_count):
           position[i] += velocity[i] * dt

Type Annotations
~~~~~~~~~~~~~~~~

Use Taichi type hints:

.. code-block:: python

   @ti.kernel
   def compute(
       amplitude: ti.f32,
       frequency: ti.f32
   ) -> ti.f32:
       pass

File Organization
-----------------

Module Structure
~~~~~~~~~~~~~~~~

Organize imports:

.. code-block:: python

   """Module docstring."""

   # Standard library
   import sys
   from pathlib import Path

   # Third-party
   import numpy as np
   import taichi as ti

   # Local
   from openwave.common import constants
   from openwave.spacetime import medium_level0

Comments
--------

When to Comment
~~~~~~~~~~~~~~~

Comment for:

- Complex algorithms
- Non-obvious optimizations
- Physics equations
- Workarounds

.. code-block:: python

   # Use attometer units for f32 precision
   # This scales 1e-17 m values to ~10 am
   position_am = position / constants.ATTOMETER

When Not to Comment
~~~~~~~~~~~~~~~~~~~

Don't comment obvious code:

.. code-block:: python

   # Bad: Obvious
   i += 1  # Increment i

   # Good: Self-documenting
   granule_count += 1

Error Handling
--------------

Exceptions
~~~~~~~~~~

Use specific exceptions:

.. code-block:: python

   if lattice is None:
       raise ValueError("Lattice cannot be None")

   if granule_count <= 0:
       raise ValueError(f"Invalid granule count: {granule_count}")

Testing
-------

Smoke Tests
~~~~~~~~~~~

Include smoke tests in ``if __name__ == "__main__"``:

.. code-block:: python

   if __name__ == "__main__":
       print("SMOKE TEST: Module Name")
       # Run basic tests
       print("SMOKE TEST PASSED")

Performance
-----------

See :doc:`performance` for detailed performance guidelines.

**Key Points:**

- Profile before optimizing
- Use GPU kernels for parallel work
- Minimize Python loops
- Use vectorized operations

Resources
---------

- **Black:** https://black.readthedocs.io/
- **PEP 8:** https://www.python.org/dev/peps/pep-0008/
- **Type Hints:** https://docs.python.org/3/library/typing.html
- **Google Style Guide:** https://google.github.io/styleguide/pyguide.html
