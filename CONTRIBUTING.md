# Contributing to OpenWave

Thank you for your interest in contributing!  
Whether you're fixing a typo, adding a feature, or reporting a bug, your help makes OpenWave better for everyone.

## How You Can Contribute

- **Report Issues:** If you find a bug, open an issue describing the problem and how to reproduce it.
- **Suggest Features:** Share ideas for new features or improvements through the issue tracker.
- **Improve Documentation:** Help us make guides, examples, and API references clearer.
- **Write Code:** Fix bugs, add features, or improve existing code.

## Practice the Community Code

- Be respectful and constructive.
- Follow the [OpenWave Code of Conduct](./CODE_OF_CONDUCT.md).
- Ask questions — we’re here to help each other.
- Read this Contribution Guide

See `/dev_docs` for coding standards and development guidelines

- [Coding Standards](dev_docs/CODING_STANDARDS.md)
- [Performance Guidelines](dev_docs/PERFORMANCE_GUIDELINES.md)
- [Loop Optimization Patterns](dev_docs/LOOP_OPTIMIZATION.md)
- [Markdown Style Guide](dev_docs/MARKDOWN_STYLE_GUIDE.md)  

*This is the Way!*

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
# Create virtual environment
  # Option 1: via Venv
    python -m venv openwave
    source openwave/bin/activate  # On Windows: openwave\Scripts\activate
   
  # Option 2: via Conda (recommended)
    conda create -n openwave python=3.12
    conda activate openwave

# Install OpenWave & Dependencies for Development (-e = edit mode)
   pip install -e .  # installs dependencies from pyproject.toml
   ```

- **Create a Branch to Develop Your Feature**

```bash
   git checkout -b your-feature-name
   ```

- Optional: LaTex & FFmpeg (video generation)

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

## Contributor License Agreement (CLA)

By contributing to OpenWave, you agree that:

1. **You own the rights** to your contribution, or have permission to contribute it.
1. **You grant OpenWave Labs** a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to use, modify, and distribute your contribution under the project's license.
1. **Your contribution is voluntary** and you receive no compensation.
1. **You understand** that OpenWave is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0), which is a strong copyleft license.
1. **You agree** that your contributions will be subject to the same license as the project.
1. **You represent** that your contribution is your original work and does not violate any third-party rights.

### Why We Need This

This agreement ensures:

- OpenWave Labs can maintain and distribute the project
- All contributions remain under the same license
- The project is legally protected
- Contributors retain credit for their work

### Attribution

All contributors will be credited in project documentation and/or a CONTRIBUTORS file.

## License Notice

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This means:

- ✅ **Open-Source:** Free to use, modify, and distribute
- ✅ **Commercial use allowed:** Businesses can use OpenWave
- ✅ **Strong copyleft:** If you distribute modified versions (including use over a network/SaaS), you MUST share your source code under GNU AGPL-3.0
- ✅ **Network protection:** Even cloud/web services must disclose source
- ⚠️ **No proprietary forks:** You cannot create closed-source versions (this PROTECTS against misuse while keeping the project truly open-source)

See the [LICENSE](LICENSE) file for full terms.

## Trademark Notice

"OpenWave" is a trademark of OpenWave Labs. See [TRADEMARK](TRADEMARK) for usage guidelines.

## Need Help?

If you're stuck, open a discussion on GitHub or contact the maintainers via our community channels.
