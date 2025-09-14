# Contributing to OpenWave

Thank you for your interest in contributing!  
Whether you're fixing a typo, adding a feature, or reporting a bug, your help makes OpenWave better for everyone.

## How You Can Contribute

- **Report Issues:** If you find a bug, open an issue describing the problem and how to reproduce it.
- **Suggest Features:** Share ideas for new features or improvements through the issue tracker.
- **Improve Documentation:** Help us make guides, examples, and API references clearer.
- **Write Code:** Fix bugs, add features, or improve existing code.

## Development Documentation

Detailed development documentation available in `/dev_docs/`:

- [Coding Standards](dev_docs/CODING_STANDARDS.md)
- [Performance Guidelines](dev_docs/PERFORMANCE_GUIDELINES.md)
- [Loop Optimization Patterns](dev_docs/LOOP_OPTIMIZATION.md)
- [Markdown Style Guide](dev_docs/MARKDOWN_STYLE_GUIDE.md)

## Getting Started

- **Fork the Repository**  
  - Click “Fork” on GitHub to create your own copy.

- **Clone Your Fork**

```bash
      git clone https://github.com/YOUR-USERNAME/openwave.git
      cd openwave
   ```

- **Set Up the Environment & Install**

```bash
   # Create virtual environment (via Venv)
      python -m venv openwave
      source openwave/bin/activate  # On Windows: openwave\Scripts\activate
   
   # Or Create virtual environment (via Conda)
      conda create -n openwave python=3.12 -y
      conda activate openwave

   # Install OpenWave & Dependencies for Development (-e = edit mode)
      pip install -e .  # installs dependencies from pyproject.toml
   ```

- **Create a Branch to Develop Your Feature**

```bash
      git checkout -b your-feature-name
   ```

### Optional: LaTex & FFmpeg (video generation)

```bash
   # Install LaTeX and FFmpeg (macOS)
      brew install --cask mactex-no-gui ffmpeg
      echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
      exec zsh -l

   # Verify LaTeX installation
      which latex && latex --version
      which dvisvgm && dvisvgm --version
      which gs && gs --version
```

## Code Style & Quality

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use [Black](https://black.readthedocs.io/) and [isort](https://pycqa.github.io/isort/) for formatting.
- Run tests before committing:

```bash
  pytest
  ```

---

## Submitting Your Changes

- Commit with a clear, descriptive message.
- Push your branch to your fork:

```bash
   git push origin your-feature-name
   ```

- Open a Pull Request (PR) on GitHub.
- Be ready to discuss and revise your PR after review.

---

## Community Guidelines

- Be respectful and constructive.
- Follow the [OpenWave Code of Conduct](./CODE_OF_CONDUCT.md).
- Ask questions — we’re here to help.

---

## Need Help?

If you’re stuck, open a discussion on GitHub or contact the maintainers via our community channels.
