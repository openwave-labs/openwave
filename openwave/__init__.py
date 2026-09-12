"""Core package for the OpenWave project.

Open-source subatomic physics simulator using classical field methods with topology
to study particle and force emergence. GPU-accelerated.

"""

# Calendar versioning, YY.M.D: the date the release was cut. No zero padding, because
# PEP 440 strips it and "26.09.07" would install as "26.9.7", silently unmatching its tag.
# A second release on a day already used takes a fourth component, "26.9.7.1". NOT a letter
# suffix: PEP 440 reads "26.9.7b" as beta, which sorts BEFORE 26.9.7 and is skipped by a
# default pip install. Keep this in sync with pyproject.toml. See dev_docs/VERSION_MANAGEMENT.md.
__version__ = "26.9.7"
