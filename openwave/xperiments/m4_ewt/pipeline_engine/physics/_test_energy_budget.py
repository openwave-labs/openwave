"""
Tests for EnergyBudget and EnergyBudgetUpdate (plan item 1.8).

The energy kernel uses the forward-difference edge sum of plan item
1.23, with the mean of the two leapfrog levels in time, over interior
nodes and edges that touch interior. Tests here pin the shape, the
conservation (spread over a full period and O(dt^2) convergence), the
dE_dt reporting against a finite difference of the recorded E_total,
and conservation under all three BoundaryProcessor kinds (dirichlet,
reflecting, periodic).

The boundary weight is read from the BoundaryCondition feature, the
same feature BoundaryProcessor reads, so the two processors cannot
disagree on the kind through the feature alone. A pipeline without
the feature scores at weight 1.0 (dirichlet/reflecting value).

The negative control (gradient-only) shows why E_kin is required.

Run from the project root:
    python -m openwave.xperiments.m4_ewt.pipeline_engine.physics._test_energy_budget

Note: no `from __future__ import annotations` here. Taichi kernel
definitions need live type objects.
"""

import sys
import traceback
from dataclasses import dataclass

import numpy as np
import taichi as ti

from .features import (
    BoundaryCondition,
    EMCDensityField,
    EnergyBudget,
    PsiLongField,
    WaveGrid,
)
from ..pipeline import (
    BaseProcessor,
    ErrorPolicy,
    Pipeline,
    PipelineError,
    Stage,
)
from .allocator import AllocateWaveField, AllocateWaveSpeed
from .boundary import BoundaryProcessor
from .emc import UpdateEMCDensityProcessor, UpdateWaveSpeedProcessor
from .energy import EnergyBudgetUpdate
from .evolution import (
    ClearAccelerationProcessor,
    LaplacianProcessor,
    LeapfrogProcessor,
)
from .units import UnitSystem

_TI_INITIALIZED = False


def _ti_init():
    global _TI_INITIALIZED
    if not _TI_INITIALIZED:
        ti.init(arch=ti.cpu, log_level=ti.ERROR)
        _TI_INITIALIZED = True


# ======================================================================
# Fixtures
# ======================================================================


@dataclass(frozen=True)
class _UnitsWithC(UnitSystem):
    """
    Local unit system with a chosen c. The rest is fixed so a test can
    set c != 1 without touching the shipped implementations.
    """

    c_value: float = 1.0

    @property
    def c(self):
        return self.c_value

    @property
    def wavelength(self):
        return 1.0

    @property
    def dx(self):
        return 0.05

    @property
    def dt(self):
        return 0.01

    @property
    def rho_0(self):
        return 1.0

    def to_physical_length(self, x):
        return x

    def to_physical_time(self, t):
        return t

    def to_physical_energy(self, E):
        return E

    def to_physical_density(self, r):
        return r


class _SeedStandingWave(BaseProcessor):
    """
    Seed PsiLong with a 3D standing wave, rest start (psi_prev = psi).
    Modes are mode_k * pi / (n - 1) per axis, zero on the outer shell.

    mode is a (mx, my, mz) tuple. mode=(1, 1, 1) is the mirror-
    symmetric case, whose periodic wrap is zero. mode=(2, 2, 2) is
    the asymmetric case with non-zero wrap, used in the periodic
    boundary test.
    """

    name = "_SeedStandingWave"
    stage = Stage.PRE_UPDATE
    order = 5
    requires = (WaveGrid, PsiLongField)

    def __init__(self, amp=1.0, mode=(1, 1, 1)):
        self.amp = float(amp)
        self.mode = tuple(int(m) for m in mode)

    def process(self, ctx):
        if ctx.sim.step > 0:
            return
        grid = ctx.data.require(WaveGrid)
        field = ctx.data.require(PsiLongField)
        _seed_standing_wave(
            field.psi,
            field.psi_prev,
            grid.nx,
            grid.ny,
            grid.nz,
            self.amp,
            self.mode[0],
            self.mode[1],
            self.mode[2],
        )


@ti.kernel
def _seed_standing_wave(
    psi: ti.template(),
    prev: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    amp: ti.f32,
    mx: ti.i32,
    my: ti.i32,
    mz: ti.i32,
):
    kx = ti.cast(mx, ti.f32) * ti.math.pi / ti.cast(nx - 1, ti.f32)
    ky = ti.cast(my, ti.f32) * ti.math.pi / ti.cast(ny - 1, ti.f32)
    kz = ti.cast(mz, ti.f32) * ti.math.pi / ti.cast(nz - 1, ti.f32)
    for i, j, k in ti.ndrange(nx, ny, nz):
        s = (
            ti.sin(kx * ti.cast(i, ti.f32))
            * ti.sin(ky * ti.cast(j, ti.f32))
            * ti.sin(kz * ti.cast(k, ti.f32))
        )
        v = ti.Vector([amp * s, 0.0, 0.0])
        psi[i, j, k] = v
        prev[i, j, k] = v


class _SeedRhoUniform(BaseProcessor):
    """Set rho = 1.0 everywhere at step 0."""

    name = "_SeedRhoUniform"
    stage = Stage.PRE_UPDATE
    order = 6
    requires = (EMCDensityField,)

    def process(self, ctx):
        if ctx.sim.step > 0:
            return
        ctx.data.require(EMCDensityField).rho.fill(1.0)


class _SeedRhoDeficit(BaseProcessor):
    """Set rho = 1 - 0.5 * exp(-r^2/8) at step 0. Sharp deficit at centre."""

    name = "_SeedRhoDeficit"
    stage = Stage.PRE_UPDATE
    order = 6
    requires = (EMCDensityField, WaveGrid)

    def process(self, ctx):
        if ctx.sim.step > 0:
            return
        grid = ctx.data.require(WaveGrid)
        ctx.data.require(EMCDensityField).rho.from_numpy(
            _rho_deficit_array(grid.nx, grid.ny, grid.nz)
        )


def _rho_deficit_array(nx, ny, nz):
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    r2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
    return (1.0 - 0.5 * np.exp(-r2 / 8.0)).astype(np.float32)


class _RunResult:
    """
    Lightweight container for a run and its recorded histories.
    """

    __slots__ = ("ctx", "E_total", "E_grad", "dE_dt")

    def __init__(self, ctx, E_total, E_grad, dE_dt):
        self.ctx = ctx
        self.E_total = E_total
        self.E_grad = E_grad
        self.dE_dt = dE_dt


def _run(
    grid_n=16,
    dx=1.0,
    dt=0.05,
    max_steps=100,
    amp=1.0,
    mode=(1, 1, 1),
    rho_def=False,
    kappa=1.0,
    use_variable_c2=False,
    c_value=1.0,
    boundary_kind=None,
):
    """
    Minimal energy pipeline.

    boundary_kind None -> no BoundaryProcessor, no BoundaryCondition
    feature. EnergyBudgetUpdate scores at weight 1.0.

    boundary_kind set -> BoundaryProcessor with that kind, and
    BoundaryCondition provided through initial_features. EnergyBudgetUpdate
    reads the feature and uses the matching weight.

    Returns a _RunResult with the ctx and the three histories.
    """
    E_total_hist = []
    E_grad_hist = []
    dE_dt_hist = []

    class _Record(BaseProcessor):
        name = "_Record"
        stage = Stage.MEASURE
        order = 6
        requires = (EnergyBudget,)

        def process(self, ctx):
            b = ctx.data.require(EnergyBudget)
            E_total_hist.append(b.E_total)
            E_grad_hist.append(b.E_grad)
            dE_dt_hist.append(b.dE_dt)

    class P(Pipeline):
        def __init__(self):
            external = [UnitSystem]
            if boundary_kind is not None:
                external.append(BoundaryCondition)
            super().__init__(
                error_policy=ErrorPolicy.FAIL_FAST,
                external_provides=tuple(external),
            )
            self.add(AllocateWaveField(nx=grid_n, ny=grid_n, nz=grid_n, dx=dx))
            if use_variable_c2:
                self.add(AllocateWaveSpeed())
            self.add(_SeedStandingWave(amp=amp, mode=mode))
            if use_variable_c2:
                self.add(UpdateEMCDensityProcessor(field_type=PsiLongField, beta_rho=0.1))
                self.add(UpdateWaveSpeedProcessor())
            elif rho_def:
                self.add(_SeedRhoDeficit())
            else:
                self.add(_SeedRhoUniform())
            self.add(ClearAccelerationProcessor(field_type=PsiLongField))
            self.add(LaplacianProcessor(field_type=PsiLongField))
            self.add(LeapfrogProcessor(field_type=PsiLongField))
            self.add(
                EnergyBudgetUpdate(
                    field_type=PsiLongField,
                    use_variable_c2=use_variable_c2,
                    kappa=kappa,
                )
            )
            if boundary_kind is not None:
                self.add(BoundaryProcessor(field_type=PsiLongField))
            self.add(_Record())

    from ..runner import Runner
    from ..sinks import InMemorySink

    initial = [_UnitsWithC(c_value)]
    if boundary_kind is not None:
        initial.append(BoundaryCondition(boundary_kind))

    ctx = Runner({"session": InMemorySink()}).run(
        P(),
        name="energy_budget_test",
        params={},
        dt=dt,
        max_steps=max_steps,
        initial_features=initial,
    )
    return _RunResult(ctx, E_total_hist, E_grad_hist, dE_dt_hist)


# ======================================================================
# Structural tests
# ======================================================================


def test_uniform_rho_gives_zero_deform():
    """
    rho = 1.0 everywhere -> (rho - 1.0) = 0 everywhere -> E_deform = 0.

    Mutation caught by this test: the deformation term reading |rho|
    instead of (rho - 1).
    """
    _ti_init()
    r = _run(grid_n=12, dt=0.05, max_steps=3)
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    b = r.ctx.data.require(EnergyBudget)
    assert b.E_deform == 0.0, b.E_deform
    assert b.E_kin > 0.0, b.E_kin
    assert b.E_grad > 0.0, b.E_grad


def test_deficit_contributes_to_deform():
    """
    rho < 1 in the centre -> E_deform > 0, and its value matches an
    independent numpy integral over the interior nodes.

    Mutation caught: E_deform reading a constant instead of the field,
    or the deformation sum including the shell.
    """
    _ti_init()
    n = 12
    dx = 1.0
    kappa = 1.0
    r = _run(grid_n=n, dx=dx, dt=0.05, max_steps=2, rho_def=True, kappa=kappa)
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    b = r.ctx.data.require(EnergyBudget)
    rho_arr = r.ctx.data.require(EMCDensityField).rho.to_numpy()
    interior = (slice(1, -1),) * 3
    drho = rho_arr[interior] - 1.0
    dv = dx**3
    expected = 0.5 * kappa * (drho * drho).sum() * dv
    assert abs(b.E_deform - expected) < 1e-6, (b.E_deform, expected)


def test_components_against_numpy_at_nonunit_scales():
    """
    dx = 0.5, kappa = 2, c_value = 2, mode-1 standing wave with a
    deficit rho, no BoundaryProcessor. All three components match a
    numpy reference on the same arena, so the dv exponent, the kappa
    factor and c against c^2 are all checked absolutely.

    Mutation caught: dv = dx^2 or dv = 1, kappa dropped or squared,
    c instead of c^2.
    """
    _ti_init()
    n = 12
    dx = 0.5
    dt = 0.05
    kappa = 2.0
    c_value = 2.0
    r = _run(
        grid_n=n,
        dx=dx,
        dt=dt,
        max_steps=1,
        rho_def=True,
        kappa=kappa,
        c_value=c_value,
    )
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    psi = r.ctx.data.require(PsiLongField).psi.to_numpy().astype(np.float64)
    psi_prev = r.ctx.data.require(PsiLongField).psi_prev.to_numpy().astype(np.float64)
    rho = r.ctx.data.require(EMCDensityField).rho.to_numpy().astype(np.float64)

    e_kin_exp, e_grad_exp, e_deform_exp = _numpy_energy_components(
        psi, psi_prev, rho, dx, dt, kappa, c_value**2, weight_shell=1.0
    )

    b = r.ctx.data.require(EnergyBudget)
    assert abs(b.E_kin - e_kin_exp) < 1e-6 * max(1.0, abs(e_kin_exp)), (b.E_kin, e_kin_exp)
    assert abs(b.E_grad - e_grad_exp) < 1e-6 * max(1.0, abs(e_grad_exp)), (
        b.E_grad,
        e_grad_exp,
    )
    assert abs(b.E_deform - e_deform_exp) < 1e-6 * max(1.0, abs(e_deform_exp)), (
        b.E_deform,
        e_deform_exp,
    )


def _numpy_energy_components(psi, psi_prev, rho, dx, dt, kappa, c2, weight_shell):
    """
    Numpy reference for the three components: interior nodes, interior-
    interior edges, shell-interior edges weighted by weight_shell.
    """
    interior = (slice(1, -1),) * 3
    dv = dx**3
    inv_dx = 1.0 / dx
    inv_dt = 1.0 / dt

    v = (psi - psi_prev) * inv_dt
    e_kin = 0.5 * (v[interior] ** 2).sum() * dv

    drho = rho[interior] - 1.0
    e_deform = 0.5 * kappa * (drho**2).sum() * dv

    e_grad = 0.0
    # x-axis interior-interior
    dn = (psi[2:-1, 1:-1, 1:-1] - psi[1:-2, 1:-1, 1:-1]) * inv_dx
    dp = (psi_prev[2:-1, 1:-1, 1:-1] - psi_prev[1:-2, 1:-1, 1:-1]) * inv_dx
    e_grad += 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv
    # x-axis shell-interior (lower and upper)
    dn = (psi[1, 1:-1, 1:-1] - psi[0, 1:-1, 1:-1]) * inv_dx
    dp = (psi_prev[1, 1:-1, 1:-1] - psi_prev[0, 1:-1, 1:-1]) * inv_dx
    e_grad += weight_shell * 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv
    dn = (psi[-1, 1:-1, 1:-1] - psi[-2, 1:-1, 1:-1]) * inv_dx
    dp = (psi_prev[-1, 1:-1, 1:-1] - psi_prev[-2, 1:-1, 1:-1]) * inv_dx
    e_grad += weight_shell * 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv
    # y-axis interior-interior
    dn = (psi[1:-1, 2:-1, 1:-1] - psi[1:-1, 1:-2, 1:-1]) * inv_dx
    dp = (psi_prev[1:-1, 2:-1, 1:-1] - psi_prev[1:-1, 1:-2, 1:-1]) * inv_dx
    e_grad += 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv
    # y-axis shell-interior
    dn = (psi[1:-1, 1, 1:-1] - psi[1:-1, 0, 1:-1]) * inv_dx
    dp = (psi_prev[1:-1, 1, 1:-1] - psi_prev[1:-1, 0, 1:-1]) * inv_dx
    e_grad += weight_shell * 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv
    dn = (psi[1:-1, -1, 1:-1] - psi[1:-1, -2, 1:-1]) * inv_dx
    dp = (psi_prev[1:-1, -1, 1:-1] - psi_prev[1:-1, -2, 1:-1]) * inv_dx
    e_grad += weight_shell * 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv
    # z-axis interior-interior
    dn = (psi[1:-1, 1:-1, 2:-1] - psi[1:-1, 1:-1, 1:-2]) * inv_dx
    dp = (psi_prev[1:-1, 1:-1, 2:-1] - psi_prev[1:-1, 1:-1, 1:-2]) * inv_dx
    e_grad += 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv
    # z-axis shell-interior
    dn = (psi[1:-1, 1:-1, 1] - psi[1:-1, 1:-1, 0]) * inv_dx
    dp = (psi_prev[1:-1, 1:-1, 1] - psi_prev[1:-1, 1:-1, 0]) * inv_dx
    e_grad += weight_shell * 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv
    dn = (psi[1:-1, 1:-1, -1] - psi[1:-1, 1:-1, -2]) * inv_dx
    dp = (psi_prev[1:-1, 1:-1, -1] - psi_prev[1:-1, 1:-1, -2]) * inv_dx
    e_grad += weight_shell * 0.5 * c2 * 0.5 * ((dn**2).sum() + (dp**2).sum()) * dv

    return e_kin, e_grad, e_deform


def test_energy_total_includes_all_three_components():
    """
    Structural check: E_total is the sum of its three components.

    Mutation caught: E_total defined as E_grad + E_deform (kinetic
    term dropped).
    """
    _ti_init()
    r = _run(grid_n=12, dt=0.05, max_steps=5)
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    b = r.ctx.data.require(EnergyBudget)
    assert b.E_kin > 0.0, b.E_kin
    assert b.E_grad > 0.0, b.E_grad

    assert b.E_total > b.E_grad + b.E_deform, (
        f"E_total {b.E_total} not greater than E_grad + E_deform " f"{b.E_grad + b.E_deform}"
    )
    assert (
        abs(b.E_total - (b.E_kin + b.E_grad + b.E_deform)) < 1e-9
    ), f"E_total {b.E_total} vs sum {b.E_kin + b.E_grad + b.E_deform}"


def test_dE_dt_zero_on_static_state():
    """
    A field with psi == psi_prev and no gradient has E_total = 0 on
    both steps, so dE_dt = 0 after the first step.
    """
    _ti_init()

    class _SeedZero(BaseProcessor):
        name = "_SeedZero"
        stage = Stage.PRE_UPDATE
        order = 5
        requires = (PsiLongField,)

        def process(self, ctx):
            if ctx.sim.step > 0:
                return
            f = ctx.data.require(PsiLongField)
            f.psi.fill(0.0)
            f.psi_prev.fill(0.0)

    class P(Pipeline):
        def __init__(self):
            super().__init__(
                error_policy=ErrorPolicy.FAIL_FAST,
                external_provides=(UnitSystem,),
            )
            self.add(AllocateWaveField(nx=8, ny=8, nz=8, dx=1.0))
            self.add(_SeedZero())
            self.add(_SeedRhoUniform())
            self.add(EnergyBudgetUpdate(field_type=PsiLongField))

    from ..runner import Runner
    from ..sinks import InMemorySink

    ctx = Runner({"session": InMemorySink()}).run(
        P(),
        name="static_test",
        params={},
        dt=0.05,
        max_steps=4,
        initial_features=[_UnitsWithC(1.0)],
    )
    assert ctx.diag.errors == [], ctx.diag.errors

    b = ctx.data.require(EnergyBudget)
    assert b.E_total == 0.0, b.E_total
    assert b.dE_dt == 0.0, b.dE_dt


def test_variable_c2_path_differs_from_const_path():
    """
    With c^2 read from WaveSpeedField, E_grad differs from the same
    field scored at constant c^2.

    Mutation caught: use_variable_c2=True silently falls back to the
    constant-c^2 kernel.
    """
    _ti_init()

    r_const = _run(grid_n=12, dt=0.05, max_steps=1, use_variable_c2=False)
    r_var = _run(grid_n=12, dt=0.05, max_steps=1, use_variable_c2=True)
    assert r_const.ctx.diag.errors == [], r_const.ctx.diag.errors
    assert r_var.ctx.diag.errors == [], r_var.ctx.diag.errors

    e_grad_const = r_const.ctx.data.require(EnergyBudget).E_grad
    e_grad_var = r_var.ctx.data.require(EnergyBudget).E_grad

    assert e_grad_const > 0.0, e_grad_const
    assert e_grad_var > 0.0, e_grad_var
    rel = abs(e_grad_var - e_grad_const) / abs(e_grad_const)
    assert rel > 1e-3, (
        f"variable-c2 E_grad {e_grad_var} vs const {e_grad_const}, "
        f"relative difference {rel} < 1e-3"
    )


# ======================================================================
# Conservation: per-step recording, spread, order
# ======================================================================


def test_standing_wave_conserves_total_energy():
    """
    A 3D standing wave at rest, integrated with the production chain,
    conserves E_total over one period to the O(dt^2) bar of plan item
    1.23.

    Mutation caught: any kernel bug a single-step read could not see.
    """
    _ti_init()
    n = 16
    dt = 0.05
    steps = 350

    r = _run(grid_n=n, dt=dt, max_steps=steps)
    assert r.ctx.diag.errors == [], r.ctx.diag.errors
    assert len(r.E_total) == steps, len(r.E_total)

    arr = np.array(r.E_total)
    spread = (arr.max() - arr.min()) / abs(arr.mean())
    assert spread < 5e-4, f"E_total spread over {steps} steps = {spread}, expected < 5e-4"


def test_conservation_at_c_other_than_1():
    """
    Same standing wave at c = 2 (so c^2 = 4).

    Mutation caught: c for c^2 in the gradient term.

    Threshold looser than the c=1 test: the spread scales as
    (c k dt)^2, four times the c=1 value at c=2.
    """
    _ti_init()
    n = 16
    dt = 0.05
    steps = 350

    r = _run(grid_n=n, dt=dt, max_steps=steps, c_value=2.0)
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    arr = np.array(r.E_total)
    spread = (arr.max() - arr.min()) / abs(arr.mean())
    # Threshold looser than the c=1 test: the spread scales as
    # (c k dt)^2, so at c=2 it is four times the c=1 value.
    assert spread < 1e-3, (
        f"E_total spread at c=2 over {steps} steps = {spread}, " f"expected < 1e-3"
    )


def test_energy_conservation_is_second_order_in_dt():
    """
    The same standing wave at dt and dt/2. Under the mean form the
    spread is O(dt^2), so the ratio must be about 4.

    Mutation caught: the kernel using the level-n+1-only gradient.
    """
    _ti_init()
    n = 16
    dt = 0.05
    steps_coarse = 350
    steps_fine = 700

    r1 = _run(grid_n=n, dt=dt, max_steps=steps_coarse)
    r2 = _run(grid_n=n, dt=dt / 2, max_steps=steps_fine)
    assert r1.ctx.diag.errors == [], r1.ctx.diag.errors
    assert r2.ctx.diag.errors == [], r2.ctx.diag.errors

    a1 = np.array(r1.E_total)
    a2 = np.array(r2.E_total)
    spread1 = (a1.max() - a1.min()) / abs(a1.mean())
    spread2 = (a2.max() - a2.min()) / abs(a2.mean())

    ratio = spread1 / spread2 if spread2 > 0.0 else float("inf")
    assert ratio >= 3.0, (
        f"spread(dt)={spread1:.3e}, spread(dt/2)={spread2:.3e}, "
        f"ratio {ratio:.2f} < 3 (expected ~4 for O(dt^2))"
    )


def test_dE_dt_matches_finite_difference():
    """
    dE_dt reported by the processor equals the finite difference of
    the recorded E_total, sign included.

    Mutation caught: dE_dt sign flipped, dE_dt divided by dt twice,
    E_prev updated before the difference, or dE_dt hard-wired.
    """
    _ti_init()
    n = 12
    dt = 0.05
    steps = 20

    r = _run(grid_n=n, dt=dt, max_steps=steps)
    assert r.ctx.diag.errors == [], r.ctx.diag.errors
    assert len(r.E_total) == steps
    assert len(r.dE_dt) == steps

    for step in range(1, steps):
        expected = (r.E_total[step] - r.E_total[step - 1]) / dt
        got = r.dE_dt[step]
        assert (
            abs(got - expected) < 1e-9
        ), f"step {step}: dE_dt {got} vs finite difference {expected}"


def test_conservation_at_nonunit_dx():
    """
    Same standing wave at dx = 0.5.

    Mutation caught: a kernel that drops the 1/dx in the gradient, or
    hard-codes dx = 1 there. Not caught: the exponent of dv, since the
    spread is relative.

    Threshold looser than the dx = 1 test: the spread scales as
    (omega dt)^2, and omega doubles at dx = 0.5.
    """
    _ti_init()
    n = 16
    dx = 0.5
    dt = 0.05
    steps = 350

    r = _run(grid_n=n, dx=dx, dt=dt, max_steps=steps)
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    arr = np.array(r.E_total)
    spread = (arr.max() - arr.min()) / abs(arr.mean())
    assert (
        spread < 1e-3
    ), f"E_total spread at dx=0.5 over {steps} steps = {spread}, expected < 1e-3"


# ======================================================================
# Boundaries: dirichlet, reflecting, periodic
# ======================================================================


def test_boundary_dirichlet_conserves():
    """
    With BoundaryProcessor(dirichlet) and BoundaryCondition(dirichlet)
    registered, E_total conserves. weight_shell is read from the
    feature, and both processors read the same feature.

    Mutation caught: the dirichlet weight changed (0.5 halves every
    shell-interior edge). Not caught: node or edge sums reverted to
    full-grid scope, since the dirichlet shell is zero and adds
    nothing; the reflecting and periodic tests catch that.
    """
    _ti_init()
    n = 16
    dt = 0.05
    steps = 350

    r = _run(grid_n=n, dt=dt, max_steps=steps, boundary_kind="dirichlet")
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    arr = np.array(r.E_total)
    spread = (arr.max() - arr.min()) / abs(arr.mean())
    assert spread < 5e-4, (
        f"E_total spread under dirichlet over {steps} steps = " f"{spread}, expected < 5e-4"
    )


def test_boundary_reflecting_conserves():
    """
    With BoundaryProcessor(reflecting), the shell copies the interior
    neighbour, so both shell-interior edges are exactly zero. The
    weight_shell = 1.0 is a formality here.

    The seed sin(pi x/L) is an eigenmode under dirichlet, not under
    reflecting; under reflecting the field evolves toward a cos-mode
    superposition. E_total is still conserved (leapfrog + holonomic
    constraint), so the spread stays O(dt^2).

    Mutation caught: node or edge sums reverted to full-grid scope,
    which double-counts the reflected shell and swings 109%.
    """
    _ti_init()
    n = 16
    dt = 0.05
    steps = 350

    r = _run(grid_n=n, dt=dt, max_steps=steps, boundary_kind="reflecting")
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    arr = np.array(r.E_total)
    spread = (arr.max() - arr.min()) / abs(arr.mean())
    assert spread < 5e-4, (
        f"E_total spread under reflecting over {steps} steps = " f"{spread}, expected < 5e-4"
    )


def test_boundary_periodic_conserves():
    """
    With BoundaryProcessor(periodic) and an asymmetric seed, mode
    (2, 2, 2). The shell copies the opposite face, so the two shell-
    interior edges are the same wrap value. weight_shell = 0.5 on
    each gives the wrap energy once. The feature is the same one
    BoundaryProcessor reads, so the two cannot disagree on the kind.

    This test uses an asymmetric seed on purpose: a mirror-symmetric
    seed (mode 1) has zero wrap, and the test would then be
    insensitive to the double-count it exists to catch. With mode 2
    the wrap is non-zero and a kernel that sums both edges at weight
    1.0 fails.

    dt is halved here (0.025, 700 steps) so the spread stays under the
    5e-4 the other boundary tests use. Mode 2 has twice the frequency
    of mode 1, and the spread scales as (omega dt)^2, four times
    larger; halving dt quarters it back.

    Mutation caught: the feature-weights mapping changed to 1.0 for
    periodic (both edges counted at full value, doubling the wrap), or
    shell-interior edges dropped entirely.
    """
    _ti_init()
    n = 16
    # dt halved and steps doubled relative to the other boundary tests:
    # this test uses mode (2,2,2), whose frequency is twice mode (1,1,1).
    # The spread scales as (omega dt)^2, four times larger, and the
    # threshold is kept at the same 5e-4 by dropping dt.
    dt = 0.025
    steps = 700

    r = _run(
        grid_n=n,
        dt=dt,
        max_steps=steps,
        mode=(2, 2, 2),
        boundary_kind="periodic",
    )
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    arr = np.array(r.E_total)
    spread = (arr.max() - arr.min()) / abs(arr.mean())
    assert spread < 5e-4, (
        f"E_total spread under periodic over {steps} steps = " f"{spread}, expected < 5e-4"
    )


# ======================================================================
# Negative control: kinetic term is required
# ======================================================================


def test_gradient_only_energy_oscillates():
    """
    Negative control. The same standing wave, scored by E_grad alone,
    has a per-step spread far above the tolerance the positive test
    uses.

    Mutation caught: E_kin silently dropped from E_total.
    """
    _ti_init()
    n = 16
    dt = 0.05
    steps = 350

    r = _run(grid_n=n, dt=dt, max_steps=steps)
    assert r.ctx.diag.errors == [], r.ctx.diag.errors

    g = np.array(r.E_grad)
    e = np.array(r.E_total)
    spread_grad_only = (g.max() - g.min()) / abs(g.mean())
    spread_full = (e.max() - e.min()) / abs(e.mean())

    assert spread_grad_only > 10.0 * spread_full, (
        f"gradient-only spread {spread_grad_only:.3e} vs full "
        f"{spread_full:.3e}; the negative control is not "
        f"distinguishable from the positive"
    )


# ======================================================================
# Build-time
# ======================================================================


def test_budget_update_requires_all_features():
    """
    EnergyBudgetUpdate.requires names its features, and the two
    flavours declare different tuples. BoundaryCondition is not in
    requires: it is an optional feature the processor try_get's, since
    a pipeline without a boundary is a legal configuration.

    Mutation caught: any of the required features dropped from
    requires, WaveSpeedField added to the constant-c^2 flavour, or
    BoundaryCondition incorrectly added to requires.
    """
    _ti_init()
    from .features import WaveSpeedField

    proc = EnergyBudgetUpdate(field_type=PsiLongField)
    assert WaveGrid in proc.requires, proc.requires
    assert PsiLongField in proc.requires, proc.requires
    assert EMCDensityField in proc.requires, proc.requires
    assert UnitSystem in proc.requires, proc.requires
    assert WaveSpeedField not in proc.requires, proc.requires
    assert BoundaryCondition not in proc.requires, proc.requires

    proc_var = EnergyBudgetUpdate(field_type=PsiLongField, use_variable_c2=True)
    assert WaveSpeedField in proc_var.requires, proc_var.requires


def test_weights_for_known_kinds():
    """
    The kind -> weight mapping is exact. dirichlet and reflecting
    share 1.0; periodic is 0.5; an unknown kind raises at process
    time (through the helper).

    Mutation caught: a kind mapped to the wrong weight, or an unknown
    kind silently accepted.
    """
    _ti_init()
    from .energy import _weight_shell_for_kind

    assert _weight_shell_for_kind("dirichlet") == 1.0
    assert _weight_shell_for_kind("reflecting") == 1.0
    assert _weight_shell_for_kind("periodic") == 0.5
    try:
        _weight_shell_for_kind("banana")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown kind")


def main() -> int:
    tests = [
        test_uniform_rho_gives_zero_deform,
        test_deficit_contributes_to_deform,
        test_components_against_numpy_at_nonunit_scales,
        test_energy_total_includes_all_three_components,
        test_dE_dt_zero_on_static_state,
        test_variable_c2_path_differs_from_const_path,
        test_standing_wave_conserves_total_energy,
        test_conservation_at_c_other_than_1,
        test_energy_conservation_is_second_order_in_dt,
        test_dE_dt_matches_finite_difference,
        test_conservation_at_nonunit_dx,
        test_boundary_dirichlet_conserves,
        test_boundary_reflecting_conserves,
        test_boundary_periodic_conserves,
        test_gradient_only_energy_oscillates,
        test_budget_update_requires_all_features,
        test_weights_for_known_kinds,
    ]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"ERROR: {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
        else:
            print(f"PASS: {t.__name__}")
            passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
