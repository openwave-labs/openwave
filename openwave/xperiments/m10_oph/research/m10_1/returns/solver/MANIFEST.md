# MANIFEST

## Files read
- `TASK.md` (read with the Read tool).
- `py` (the interpreter wrapper script, read with `head` while checking how it runs).
- Directory listing of this directory (`ls -la`) showed `.room.sb`; I did not open it.
- `solve.py` and `results.json` were written by me and not re-read apart from the printed output of the run.

Nothing outside this directory was read or fetched by me. The `./py` wrapper sets `PYTHONPATH` to a site-packages directory outside the room, from which Python itself loaded `sympy` and `mpmath` (and their dependencies) at run time. I did not choose or open those files.

## Interpreter and libraries
- Python 3.12.14 (`./py`, run with `-S`)
- sympy 1.14.0
- mpmath 1.3.0

## Prior-knowledge statement
I recognise the object as the vertex set of a regular icosahedron (12 vertices). Its sum over directions of a cosine dispersion is a lattice-like toy with icosahedral symmetry. I knew in general terms that the icosahedron is a spherical 5-design and that its first non-isotropic harmonic is at l = 6. I did not recall any of the coefficients (1, −1/20, 1/840, 16/75, 2/7875, 132/25) from memory; all were computed by the script.
