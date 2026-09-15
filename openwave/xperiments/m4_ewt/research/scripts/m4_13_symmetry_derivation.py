#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""M4.13 - q = 1 from spherical symmetry of the EMC density profile.

OpenWave criterion:
    Gravity: local metric phenomena

This is a jupytext percent file. To run as a notebook:

    pip install jupytext
    python -m jupytext --to notebook m4_13_symmetry_derivation.py
    jupyter notebook m4_13_symmetry_derivation.ipynb

Or run as a plain script (Markdown cells are ignored):

    python m4_13_symmetry_derivation.py
"""

# %% [markdown]
# # M4.13 — q = 1 from spherical symmetry of the EMC density profile
#
# **Criterion:** `Gravity: local metric phenomena`
#
# This notebook is the executable derivation for M4.13. It addresses the
# reviewer's request to say what the M4.9 exponents still assume: the
# pair potential exponent `n = 1` and the axial compression factor `q = 1`.
#
# The `n = 1` assumption is the EMC pair potential (documented in the
# manuscript). What this notebook removes is the separate `q = 1`
# assumption, and it adds a consistency test showing that `n = 1` is the
# only value compatible with the rest of the model.
#
# Structure:
#
# 1. Symbolic derivation: equal-magnitude strains at `m = 1`.
# 2. Physical bridge: why equal magnitudes means 1D.
# 3. Consistency test: `n != 1` fails the strain and metric tests.
# 4. Numerical verification.
# 5. Anticipated reviewer questions.
# 6. What this establishes.

# %% [markdown]
# ## Setup
#
# We work in the far field of a spherically symmetric source. The EMC
# density profile has the form
#
#     eta(r) = 1 - delta_eta(r),    delta_eta(r) = A / r^m
#
# with `A > 0` and `m` the profile exponent. The model identifies the
# profile exponent with the pair potential exponent, `m = n`, where
# `V(r) ~ 1/r^n`. For the EMC model `n = 1`, and we will show that this
# is the only value consistent with the rest of the chain.

# %%
import sympy as sp

r, A, m, n = sp.symbols('r A m n', positive=True)
delta_eta = A / r**m
sp.Eq(sp.Symbol('delta_eta'), delta_eta)

# %% [markdown]
# ## Step 1 — displacement field
#
# In the Enhanced EWT model the radial displacement field is the
# cumulative structural response to the density perturbation (see the
# manuscript section "Bridging the Vector Displacement to the Scalar
# Refractive Index"). For a spherically symmetric profile it evaluates to
#
#     u_r(r) = |delta_eta(r)| = A / r^m
#
# by the same construction that gives `u_r = r_s / r` in the Newtonian
# case. This is a radial vector field: `u(r) = u_r(r) * r_hat`.

# %%
u_r = A / r**m
sp.Eq(sp.Symbol('u_r(r)'), u_r)

# %% [markdown]
# ## Step 2 — strain tensor components
#
# For a radial displacement field the strain tensor in spherical
# coordinates has two independent components:
#
#     eps_rr    = du_r / dr      (radial strain)
#     eps_theta = u_r / r        (tangential strain)

# %%
eps_rr = sp.diff(u_r, r)
eps_theta = u_r / r

print("eps_rr    =", sp.simplify(eps_rr))
print("eps_theta =", sp.simplify(eps_theta))

# %% [markdown]
# ## Step 3 — ratio of strain components
#
# The ratio `eps_rr / eps_theta` determines whether the deformation is
# directionally uniform.

# %%
ratio = sp.simplify(eps_rr / eps_theta)
print("eps_rr / eps_theta =", ratio)

# %% [markdown]
# The symbolic result is
#
#     eps_rr / eps_theta = -m
#
# so the magnitudes are equal only when `m = 1`.

# %% [markdown]
# ## Step 4 — Newtonian case (m = 1)
#
# At `m = 1` the strain components are

# %%
ratio_m1 = ratio.subs(m, 1)
eps_rr_m1 = sp.simplify(eps_rr.subs(m, 1))
eps_theta_m1 = sp.simplify(eps_theta.subs(m, 1))

print("ratio at m=1     =", ratio_m1)
print("eps_rr at m=1    =", eps_rr_m1)
print("eps_theta at m=1 =", eps_theta_m1)

# %% [markdown]
# At `m = 1`:
#
#     eps_rr    = -A / r^2
#     eps_theta = +A / r^2
#
# The absolute values are equal.

# %% [markdown]
# ## Step 5 — physical bridge: why equal magnitudes means 1D
#
# The equality `|eps_rr| = |eps_theta|` is not a numerical coincidence;
# it is the statement that the deformation has no distinguished shear
# direction on a shell. For a spherically symmetric source, the local
# lattice spacing along a tangent to the shell (controlled by
# `eps_theta`) is stretched or compressed by the same amount in every
# tangent direction, and the radial spacing (controlled by `eps_rr`) is
# stretched or compressed by the equal-magnitude amount in the opposite
# sense. There is no tangent axis preferred over another.
#
# Because light bending and gravitational redshift in the model are
# encoded by the local refractive index `n(r)`, and because the local
# lattice deformation at each radius `r` has no preferred direction on
# the shell, the refractive index depends only on `r`:
#
#     n(r) = (1 - delta_eta(r))^{-1/2}   is spherically symmetric.
#
# Once `n` is a function of `r` alone, the wave equation for a probe
# traversing the region reduces to an effective 1D problem in the radial
# variable. The standard ray integral (M4.3), the clock-speed reduction
# `v_clock(r) = sqrt(eta(r))` (M4.4), and the Shapiro-delay quadrature
# along the impact-parameter line (M4.5) are all 1D problems in the
# radial profile.
#
# In the notation of M4.9, that collapse is exactly `q = 1`: one
# effective compressing direction (radial), no tangential preference.

# %% [markdown]
# ## Step 6 — consistency test: n != 1 fails the rest of the model
#
# The model identifies the profile exponent `m` with the pair potential
# exponent `n`. To check that `n = 1` is not just a convenient choice,
# we test independent sectors against `n != 1`. Each sector fails for a
# different reason, and each failure is a strong signal that `n = 1` is
# selected, not assumed.

# %% [markdown]
# ### Test 6a — strain equality fails for n != 1
#
# The strain ratio is `-m`. For the deformation to be directionally
# uniform we need `|ratio| = 1`, so `m = 1`, hence `n = 1`.

# %%
for n_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
    ratio_val = ratio.subs(m, n_val)
    print(f"n = {n_val}:  eps_rr / eps_theta = {ratio_val},  "
          f"|ratio| = {abs(ratio_val)}")

# %% [markdown]
# Only `n = 1` gives `|ratio| = 1`. For every other value the
# deformation is directionally non-uniform, and the 1D reduction is
# not justified.

# %% [markdown]
# ### Test 6b — Newtonian force law: numerical sanity check at n = 1
#
# The M4.10/M4.12 force law comes from the overlap integral
#
#     I(R) = ∫ ∇eta_1 · ∇eta_2  dV
#
# with `∇eta_i = -m A_i / r_i^(m+1) · r_hat_i`. For a Newtonian profile
# (`m = 1`), the angular integral is the pre-computed `4π/r^2` for
# `r > R`, and `I(R) ∝ 1/R`, so the force is `∝ 1/R^2`. We verify this
# numerically at `m = 1` as a sanity check that the numerical integration
# reproduces the analytic result.
#
# We do **not** run the numerical integrator for `m != 1`. The integrand
# has a corner at `r = R, θ = 0` where the two source points coincide.
# The behaviour of the integral there depends on `m`:
#
# - `m = 1`: the integrand is finite at the corner, the integral
#   converges, and the numerical result reproduces the analytic
#   `4π A1 A2 / R`.
# - `m = 2`: the integrand diverges as `1/θ` (equivalently `1/(r-R)`),
#   so the integral is **logarithmically divergent**. Any finite
#   numerical answer is a cutoff artifact, not a physical force.
# - `m >= 3`: the divergence is stronger than logarithmic.
#
# Because `m != 1` is not integrable at the corner, the numerical
# route cannot adjudicate it. The exclusion of `m != 1` comes from the
# analytic strain test (6a) and the analytic metric test (6c), both of
# which are exact for any `m` and do not suffer from the corner problem.
#
# **Numerical note.** The dominant numerical error in the overlap
# integral is the truncation of the r-integration at finite `r_max`.
# For `m = 1` the angular integral is analytically `2/r^4` for every
# `r > R`, so the untruncated integral is `∫_R^∞ (2/r^4)·r² dr = 2/R`,
# and cutting at `r_max` loses `2/r_max` of the total. A cutoff at
# `30 R` loses ~3.3%. We therefore use `r_max = 100 R` and add the
# analytic tail `2/r_max` explicitly. The narrow corner peak at
# `r = R, θ = 0` is regular for `m = 1` and does not require a special
# grid.

# %%
import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:  # NumPy < 2.0
    _trapz = np.trapz


def overlap_integral_m1(R, A1=1.0, A2=1.0,
                        n_theta=4000, n_r=8000,
                        r_rel_min=1e-4):
    """Numerical I(R) for two Newtonian profiles delta_eta_i = A_i / r_i.

    The r-integration is carried out from r = R(1 + r_rel_min) to
    r_max = 100 R, and the analytic tail

        integral from r_max to infinity of 2/r^2 dr = 2/r_max

    is added explicitly. Without the tail correction, a 30 R cutoff
    loses ~3% of the integral; with r_max = 100 R plus the tail, the
    residual error is below 0.1%.
    """
    r_max = 100.0 * R
    r_arr = np.linspace(R * (1.0 + r_rel_min), r_max, n_r)
    dr = r_arr[1] - r_arr[0]

    # Uniform theta grid is fine here: for m = 1 the integrand is
    # finite at the corner and the trapezoid rule converges quickly.
    theta_arr = np.linspace(1e-6, np.pi - 1e-6, n_theta)
    sin_theta = np.sin(theta_arr)
    cos_theta = np.cos(theta_arr)

    I_total = 0.0
    for r in r_arr:
        r2_sq = r*r + R*R - 2.0*r*R*cos_theta
        r2_sq = np.maximum(r2_sq, 1e-30)
        integrand = ((r - R*cos_theta) * sin_theta
                     / (r * r * r2_sq**1.5))
        angular = _trapz(integrand, theta_arr)
        I_total += angular * r * r * dr

    # Analytic tail: for r > R, the angular integral is exactly 2/r^4,
    # so integral from r_max to infinity of angular(r) * r^2 dr = 2/r_max.
    I_total += 2.0 / r_max

    return 2.0 * np.pi * A1 * A2 * I_total


A1_num, A2_num = 1.0, 1.0
R_values = np.array([1.0, 2.0, 4.0, 8.0])

print(f"{'R':>6} {'I_num':>18} {'I_analytic':>18} {'rel. diff':>12}")
print("-" * 58)

I_numeric_vals = []
for R in R_values:
    I_num = overlap_integral_m1(R, A1_num, A2_num)
    I_analytic = 4.0 * np.pi * A1_num * A2_num / R
    rel_diff = abs(I_num - I_analytic) / I_analytic
    I_numeric_vals.append(I_num)
    print(f"{R:>6.1f} {I_num:>18.10e} {I_analytic:>18.10e} "
          f"{rel_diff:>12.4e}")

log_R = np.log(R_values)
log_I = np.log(np.array(I_numeric_vals))
slope, _ = np.polyfit(log_R, log_I, 1)
force_exp = slope - 1.0

print()
print(f"Fitted slope d ln I / d ln R = {slope:.6f}")
print(f"Force exponent                = {force_exp:.6f}")
print(f"(Newtonian target: -2)")

# %% [markdown]
# For `m = 1` the numerical overlap integral reproduces the analytic
# `4π A1 A2 / R` to well below 0.1%, and the extracted force exponent
# is `-2` to the printed precision. This confirms that the numerical
# machinery reproduces the Newtonian result at `m = 1`.
#
# The same numerical integrator **cannot be used for `m >= 2`**: the
# angular integral diverges at `r = R, θ = 0`, so any finite answer
# would be a numerical cutoff artifact, not a physical force exponent.
# Rather than run an ill-posed computation, the discriminator for
# `m != 1` is the analytic strain test 6a and the analytic metric test
# 6c.

# %% [markdown]
# ### Test 6c — metric profile form fails for n != 1
#
# The metric tests of M4.3–M4.5 use the Schwarzschild-like refractive
# index
#
#     n(r) = (1 - r_s / r)^{-1/2}
#
# which corresponds to the Newtonian profile `delta_eta = r_s / r`.
# For general exponent `m`, the profile becomes `delta_eta = A / r^m`
# and the refractive index takes the form `(1 - A / r^m)^{-1/2}`.
# Only `m = 1` reproduces the logarithmic Shapiro-delay formula used
# in M4.5; for `m != 1` the logarithmic structure is lost.

# %%
r_s, A_g = sp.symbols('r_s A_g', positive=True)
schwarzschild = (1 - r_s / r)**sp.Rational(-1, 2)
general = (1 - A_g / r**m)**sp.Rational(-1, 2)

print("Schwarzschild form :", schwarzschild)
print("General form       :", general)
print()
print("The two match only when m = 1 and A_g = r_s.")
print()

# Numerically check the Shapiro delay integrand structure.
# For m=1: n(r) - 1 ≈ (1/2) * r_s / r    -> integrates to a logarithm
# For m=2: n(r) - 1 ≈ (1/2) * A / r^2    -> integrates to 1/r
for m_val, coeff in [(1, 1.0), (2, 1.0)]:
    r_test = 10.0
    n_minus_1 = (1 - coeff / r_test**m_val)**(-0.5) - 1.0
    leading = 0.5 * coeff / r_test**m_val
    print(f"m = {m_val}:  n - 1 = {n_minus_1:.6e}  "
          f"(leading ~ {leading:.6e})")

# %% [markdown]
# For `m = 1` the leading term is `(1/2) r_s / r`, whose integral along
# a straight path gives the standard logarithmic Shapiro delay. For
# `m = 2` the leading term is `(1/2) A / r^2`, whose integral gives a
# `1/r` term, not a logarithm. The M4.5 test therefore requires
# `m = 1`, hence `n = 1`.

# %% [markdown]
# ### Summary of the consistency test
#
# Independent sectors constrain `n`:
#
# | Sector | Requires | Fails for | Test type |
# |---|---|---|---|
# | Strain equality (Step 4) | `m = 1` | every other `m` | analytic, exact |
# | Force law (M4.10/M4.12) | `m = 1` | every other `m` | numerical at `m = 1`; analytic divergence excludes `m >= 2` |
# | Metric profile (M4.3–M4.5) | `m = 1` | every other `m` | analytic, exact |
#
# The strain and metric tests are analytic and exact for any `m`. The
# force test confirms the numerical machinery at `m = 1` and analytically
# excludes `m != 1` by the divergence of the overlap integral at the
# corner `r = R, θ = 0`.
#
# All three require `n = 1`. This is a consistency argument, not a
# dynamical derivation: `n = 1` is the only value compatible with the
# rest of the model. Any other value would break at least one
# already-verified sector.

# %% [markdown]
# ## Step 7 — numerical verification of the strain result

# %%
A_num = 1.0
radii = [1.0, 2.0, 5.0, 10.0]

print(f"{'r':>6} {'|eps_rr|':>14} {'|eps_theta|':>14} "
      f"{'ratio':>10} {'n':>4}")
print("-" * 54)

for r_val in radii:
    for n_val in [1, 2]:
        eps_rr_num = -n_val * A_num / r_val**(n_val + 1)
        eps_theta_num = A_num / r_val**(n_val + 1)
        ratio_num = abs(eps_rr_num) / abs(eps_theta_num)
        print(f"{r_val:>6.1f} {abs(eps_rr_num):>14.6e} "
              f"{abs(eps_theta_num):>14.6e} {ratio_num:>10.4f} "
              f"{n_val:>4}")

# %% [markdown]
# At `n = 1` the ratio is 1 for every radius. At `n = 2` the ratio is 2
# for every radius. The result does not depend on `r`, only on `n`.

# %% [markdown]
# ## Anticipated reviewer questions
#
# **Q1. Where does `u_r(r) = A / r^m` come from? Is it derived here?**
#
# No. The displacement field is the model's far-field response to a
# spherically symmetric density perturbation, given in the manuscript
# section "Bridging the Vector Displacement to the Scalar Refractive
# Index". It is a model input in the same sense as the profile itself.
#
# **Q2. Does `|eps_rr| = |eps_theta|` alone imply the 1D reduction?**
#
# No. The 1D reduction follows from the refractive index being a
# function of `r` alone, which is a consequence of spherical symmetry
# of the source. The equal-magnitude condition is what makes the
# deformation *compatible* with that symmetry.
#
# **Q3. Is this a derivation of `q = 1` or a restatement of spherical
# symmetry?**
#
# It is stronger than a restatement. It identifies what would break
# the 1D reduction — an `m != 1` profile with unequal strain
# magnitudes — and shows that the model, through `n = 1`, is not in
# that regime. In the notation of the M4.9 backlog row, `q = 1` moves
# from an independent postulate to a consequence.
#
# **Q4. Is `n = 1` derived or assumed?**
#
# Neither, strictly. It is the EMC pair potential exponent documented
# in the manuscript, and the consistency test in Step 6 shows that
# every other value breaks at least one already-verified sector. So
# `n = 1` is a model input that is uniquely compatible with the rest
# of the chain. That is a strong consistency statement, not a
# derivation from dynamics.
#
# **Q5. Why is the numerical force test run only at `m = 1`?**
#
# The overlap integrand has a corner at `r = R, θ = 0`. For `m = 1`
# the integrand is finite there and the integral converges. For `m = 2`
# the integrand diverges as `1/θ`, so the integral is logarithmically
# divergent and any finite numerical answer would be a cutoff artifact.
# For `m >= 3` the divergence is stronger. Rather than run an ill-posed
# computation, the numerical integrator is used only at `m = 1` as a
# sanity check. The exclusion of `m != 1` comes from the analytic
# strain and metric tests, which are exact for any `m`.
#
# **Q6. Does the strain argument constrain light propagation directly?**
#
# No. It constrains the refractive index to be spherically symmetric,
# and the three metric tests of M4.3–M4.5 operate on that index. The
# M4.9 exponents `beta = n / (2q)` come from the 1D chain model, which
# this notebook justifies as the correct reduction of the 3D problem.

# %% [markdown]
# ## Step 8 — what this establishes and what it does not
#
# **Establishes.**
#
# 1. For a Newtonian density profile (`m = 1`), the strain tensor from
#    a spherically symmetric source has equal-magnitude radial and
#    tangential components, and the local refractive index depends only
#    on `r`. The effective propagation problem reduces to 1D, giving
#    `q = 1`.
#
# 2. `q = 1` is not an independent postulate. It follows from `m = 1`
#    under spherical symmetry.
#
# 3. `n = 1` is the only pair potential exponent compatible with the
#    force sector (M4.10/M4.12), the metric sector (M4.3–M4.5), and
#    the strain equality derived here. Every other value breaks at
#    least one of the three.
#
# **Does not establish.**
#
# 1. A dynamical derivation of `n = 1`. The consistency argument is
#    strong but is not a derivation from the soliton equations.
#
# 2. That the M4.9 exponents `beta = n / (2q)` apply to the full
#    profile. M4.9's model is a 1D chain with a uniform density
#    perturbation; this notebook addresses the reduction from 3D
#    to that 1D model, not the propagation exponents themselves.
#
# **Net effect for the row.** The M4.9 assumption list shrinks from
# `{n = 1, q = 1}` to `{n = 1}` alone, and `n = 1` itself is now
# supported by an independent-sector consistency test rather than
# being a standalone postulate.

# %% [markdown]
# ## Reference
#
# Enhanced EWT manuscript, version 5.0.0 — DOI:
# [10.5281/zenodo.22540635](https://doi.org/10.5281/zenodo.22540635)
#
# Relevant sections:
#
# - "Bridging the Vector Displacement to the Scalar Refractive Index"
# - "Asymptotic Continuous Limit and Schwarzschild Equivalence"
# - "Newtonian Force from Interacting EMC Deficits"