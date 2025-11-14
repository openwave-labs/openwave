Contributing to OpenWave
=========================

We welcome contributions from the community! This guide will help you get started.

Ways to Contribute
------------------

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Create new xperiments
- 🔧 Fix bugs and implement features
- 🎨 Improve visualizations
- 🚀 Optimize performance

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/openwave.git
   cd openwave

3. Create development environment:

.. code-block:: bash

   conda create -n openwave-dev python=3.12
   conda activate openwave-dev

4. Install in development mode:

.. code-block:: bash

   pip install -e .

5. Create a branch for your work:

.. code-block:: bash

   git checkout -b feature/my-new-feature

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

We follow PEP 8 with some modifications. See ``dev_docs/CODING_STANDARDS.md`` for details.

**Key Points:**

- Use Black formatter (line length: 99)
- Type hints for function signatures
- Docstrings for all public functions
- Meaningful variable names

Performance
~~~~~~~~~~~

See ``dev_docs/PERFORMANCE_GUIDELINES.md`` for optimization patterns.

**Key Principles:**

- Use ``@ti.kernel`` for GPU parallelization
- Single outermost loop for Taichi kernels
- Avoid nested loops in kernels
- Profile before optimizing

Documentation
~~~~~~~~~~~~~

- Update docstrings for new functions
- Add examples for new features
- Update relevant .rst files in ``docs/``
- Follow Google docstring style

Contribution Workflow
---------------------

1. Find or Create an Issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Check existing issues first
- Create new issue describing your contribution
- Discuss approach before major work

2. Write Code
~~~~~~~~~~~~~

- Follow coding standards
- Write clear commit messages
- Keep commits focused and atomic

3. Test Your Changes
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run existing tests
   pytest tests/

   # Test your specific feature
   python openwave/your_module.py

4. Update Documentation
~~~~~~~~~~~~~~~~~~~~~~~

- Add/update docstrings
- Update relevant docs pages
- Add examples if applicable

5. Submit Pull Request
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Push your branch
   git push origin feature/my-new-feature

Then create PR on GitHub:

- Describe changes clearly
- Reference related issues
- Include screenshots if visual changes
- Wait for review

Code Review Process
-------------------

What to Expect
~~~~~~~~~~~~~~

- Maintainers will review your PR
- May request changes or clarifications
- Be patient and responsive
- Iterate based on feedback

Review Criteria
~~~~~~~~~~~~~~~

✅ Follows coding standards
✅ Includes documentation
✅ Passes tests
✅ Performance considered
✅ Clear commit messages

Areas for Contribution
----------------------

High Priority
~~~~~~~~~~~~~

🚨 **Level-1 Field Implementation**

Help implement wave-field medium:

- PDE solvers
- Boundary conditions
- Force calculations

🚨 **Matter Module**

Particle formation simulations:

- Standing wave configurations
- Particle properties
- Multi-particle systems

🚨 **Force Module**

Implement fundamental forces:

- Electric field interactions
- Magnetic field dynamics
- Gravitational effects

Medium Priority
~~~~~~~~~~~~~~~

⚡ **Performance Optimization**

- GPU memory management
- Kernel optimization
- Profiling and benchmarking

⚡ **Visualization Improvements**

- Better rendering
- Animation export
- Interactive controls

⚡ **New Xperiments**

Create educational simulations:

- Wave phenomena
- Interference patterns
- Custom scenarios

Lower Priority
~~~~~~~~~~~~~~

📚 **Documentation**

- Tutorial improvements
- API documentation
- Example notebooks

🧪 **Testing**

- Unit tests
- Integration tests
- Performance tests

Creating New Xperiments
-----------------------

Template
~~~~~~~~

Use this template for new xperiments:

.. code-block:: python

   """
   Module: openwave.xperiments.my_xperiment
   Description: Brief description of what this demonstrates
   """

   from openwave.spacetime.medium_level0 import BCCLattice
   from openwave.common import constants
   import taichi as ti

   class MyXperiment:
       """
       Detailed description of the experiment.

       Args:
           lattice: The BCC lattice medium
           param1: Description of parameter 1
           param2: Description of parameter 2
       """

       def __init__(self, lattice: BCCLattice, param1: float, param2: int):
           self.lattice = lattice
           self.param1 = param1
           self.param2 = param2

       @ti.kernel
       def update(self, time: float):
           """Update simulation state."""
           for i in range(self.lattice.granule_count):
               # Update logic here
               pass

       def run(self):
           """Main simulation loop."""
           window = ti.ui.Window("My Xperiment", (1920, 1080))
           # Rendering loop
           pass

Checklist
~~~~~~~~~

- [ ] Follows template structure
- [ ] Includes clear docstrings
- [ ] Implements update() kernel
- [ ] Implements run() method
- [ ] Added to CLI menu
- [ ] Documented in xperiments.rst

Adding to CLI
~~~~~~~~~~~~~

Edit ``openwave/_io/cli.py`` to add your xperiment to the menu:

.. code-block:: python

   xperiments = [
       "My Xperiment",
       # ... existing xperiments
   ]

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

We follow the `Contributor Covenant Code of Conduct <CODE_OF_CONDUCT.md>`_.

**Key Points:**

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Report unacceptable behavior

Communication
~~~~~~~~~~~~~

- GitHub Issues: Bug reports, feature requests
- Pull Requests: Code contributions
- Discussions: General questions
- Reddit: Community discussion

Scientific Integrity
~~~~~~~~~~~~~~~~~~~~

When contributing:

- Be honest about limitations
- Follow scientific method
- Document assumptions
- Cite sources appropriately

Getting Help
------------

Stuck?
~~~~~~

- Read ``dev_docs/`` for detailed guidance
- Check existing issues and PRs
- Ask questions in discussions
- Reach out on Reddit community

Resources
~~~~~~~~~

- Coding Standards: ``dev_docs/CODING_STANDARDS.md``
- Performance Guide: ``dev_docs/PERFORMANCE_GUIDELINES.md``
- Loop Optimization: ``dev_docs/LOOP_OPTIMIZATION.md``
- Markdown Style: ``dev_docs/MARKDOWN_STYLE_GUIDE.md``

License
-------

By contributing, you agree that your contributions will be licensed under AGPL-3.0.

See `LICENSE <../LICENSE>`_ for details.

Recognition
-----------

Contributors will be:

- Listed in CONTRIBUTORS file
- Acknowledged in release notes
- Credited in relevant documentation

Thank You!
----------

Thank you for contributing to OpenWave! Your work helps advance computational physics
and makes wave mechanics more accessible to everyone.

.. note::

   Questions? Feel free to ask in GitHub Discussions or open an issue!
