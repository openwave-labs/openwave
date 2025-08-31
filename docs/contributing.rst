Contributing
============

Contributing to OpenWave
------------------------

We welcome contributions to the OpenWave project! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment::

    conda create -n openwave312 python=3.12 -y
    conda activate openwave312
    pip install -e .

4. Install development dependencies::

    pip install pytest black flake8 mypy

Code Standards
--------------

Python Style
~~~~~~~~~~~~

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document all public functions with docstrings
- Run black formatter before committing

Testing
~~~~~~~

- Write tests for new features
- Ensure all tests pass before submitting PR
- Target minimum 80% code coverage

Documentation
~~~~~~~~~~~~~

- Update docstrings for API changes
- Add examples for new features
- Update CLAUDE.md if adding new modules

Submission Process
------------------

1. Create a feature branch from main
2. Make your changes with clear commit messages
3. Run tests and linting
4. Push to your fork
5. Open a pull request with description of changes

Areas for Contribution
-----------------------

- Physics equation implementations
- Performance optimizations
- Visualization enhancements
- Documentation improvements
- Test coverage expansion
- CLI feature additions

Questions?
----------

Open an issue on GitHub for discussion or clarification.