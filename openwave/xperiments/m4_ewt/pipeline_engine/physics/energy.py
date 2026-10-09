"""
Energy budget (plan Section 6, item 1.8).

EnergyBudgetUpdate integrates the field's kinetic, gradient and
deformation energy after each step and stores them in EnergyBudget.

Definitions (plan Section 6, item 1.8, and item 1.23 for the
conservation bar):
    E_kin    = integral (1/2) |dpsi/dt|^2 dV
    E_grad   = integral (1/2) c^2(rho) |grad psi|^2 dV
    E_deform = integral (1/2) kappa (rho - rho_0)^2 dV
    E_total  = E_kin + E_grad + E_deform

Gradient discretisation (edge sum, mean in time):
    E_grad is the forward-difference edge sum of plan item 1.23:

        E_grad = sum_edges (1/2) c^2_{i+1/2} ((psi_{i+1} - psi_i)/dx)^2 dv

    with c^2_{i+1/2} = (1/2)(c^2_i + c^2_{i+1}) on the variable path.

    The time level is the mean of the two endpoints the leapfrog holds,
    psi_prev (level n) and psi (level n+1):

        E_grad = 0.5 * c^2_{1/2} * ( |D+psi^{n+1}|^2 + |D+psi^n|^2 ) / 2 * dv

    Kinetic is at n+1/2, so the mean places the gradient at the same
    time. A one-sided level-n+1 gradient leaves a first-order gap; a
    cross-product form is exact but is a different quantity (the
    staggered invariant of item 1.23, which _test_variable_coeff pins).
    Plan item 1.23 names the O(dt^2) convergence bar for this energy;
    machine precision is reserved for the staggered expression.

Scoring scope (interior nodes, interior-interior edges, and shell-
interior edges weighted by the boundary kind):
    Nodes: interior only (i, j, k in [1, n-2]). Shell nodes hold one
    of three things after a BoundaryProcessor step: zero (dirichlet),
    a copy of the interior neighbour (reflecting), or a copy of the
    opposite face (periodic). Counting them adds nothing or a
    duplicate.

    Edges: every interior-interior edge, plus the two shell-interior
    edges at each end of each axis, weighted by weight_shell:
      1.0 under dirichlet: shell is zero, so the shell-interior edge
          at the lower end is psi(1), the upper end -psi(n-2). Both
          are real and additive; the sum matches the flux the 6-point
          Laplacian transports across the shell.
      1.0 under reflecting: shell is a copy of the interior neighbour,
          so both shell-interior edges are exactly zero. The weight
          is a formality.
      0.5 under periodic: shell is a copy of the opposite face, so
          both shell-interior edges are the same wrap value. 0.5 on
          each gives the wrap energy once, not twice.

    weight_shell comes from the registered BoundaryCondition feature,
    the same feature BoundaryProcessor reads. One source of truth for
    the boundary kind; a mismatch between the two processors is not
    possible through the feature alone. A BoundaryProcessor with
    kind_override set (a test-only path, see boundary.py) does break
    the link, and a run that uses both together would measure a
    different energy than the one the boundary applies.

    A pipeline without a BoundaryCondition registered scores at
    weight_shell = 1.0, the dirichlet/reflecting value. That matches
    a seed whose shell is zero by construction (the standing-wave
    fixture) and DirichletBoundaryProcessor's output. It does not
    match an arbitrary seed, which is a caller's responsibility to
    keep consistent.

    The pre-fix versions: a full-grid sum counted the duplicated shell
    nodes (109% swing under reflecting and periodic); a
    centered-difference gradient over interior voxels swung 15% under
    dirichlet. Separately, an edge sum without the shell-interior
    edges, which the 6-point Laplacian transports energy across, loses
    26% of the gradient score of a mode whose gradient peaks at the
    shell.

Accumulators are f64 to avoid f32 summation loss on larger grids.
The values are read back to Python floats once per step.

kappa is a per-instance configuration on the processor, not a field
on the feature. It is a measurement parameter, not a result.
"""

import taichi as ti

from ..pipeline import BaseProcessor, Stage
from .features import (
    BoundaryCondition,
    EMCDensityField,
    EnergyBudget,
    PsiLongField,
    WaveGrid,
    WaveSpeedField,
)
from .units import UnitSystem


@ti.func
def _edge_energy(
    psi_new: ti.template(),
    psi_prev: ti.template(),
    c2_half: ti.f32,
    inv_dx: ti.f32,
    dv: ti.f32,
    i_a: ti.i32,
    j_a: ti.i32,
    k_a: ti.i32,
    i_b: ti.i32,
    j_b: ti.i32,
    k_b: ti.i32,
):
    """
    Energy of one edge, forward-difference form, mean of two levels.
    psi_new is psi^{n+1}, psi_prev is psi^n.
    """
    d_new = (psi_new[i_b, j_b, k_b] - psi_new[i_a, j_a, k_a]) * inv_dx
    d_prev = (psi_prev[i_b, j_b, k_b] - psi_prev[i_a, j_a, k_a]) * inv_dx
    return 0.5 * c2_half * 0.5 * (d_new.norm_sqr() + d_prev.norm_sqr()) * dv


@ti.kernel
def _integrate_energy_const_c2(
    psi: ti.template(),
    psi_prev: ti.template(),
    rho: ti.template(),
    kappa: ti.f32,
    c2_const: ti.f32,
    inv_dt: ti.f32,
    inv_dx: ti.f32,
    dv: ti.f32,
    weight_shell: ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    out_kin: ti.template(),
    out_grad: ti.template(),
    out_deform: ti.template(),
):
    out_kin[None] = 0.0
    out_grad[None] = 0.0
    out_deform[None] = 0.0

    # Kinetic and deformation, interior nodes only.
    for i, j, k in ti.ndrange((1, nx - 1), (1, ny - 1), (1, nz - 1)):
        v = (psi[i, j, k] - psi_prev[i, j, k]) * inv_dt
        out_kin[None] += 0.5 * v.norm_sqr() * dv
        drho = rho[i, j, k] - 1.0
        out_deform[None] += 0.5 * kappa * drho * drho * dv

    # x-edges: interior-interior, then the two shell-interior edges.
    for i, j, k in ti.ndrange((1, nx - 2), (1, ny - 1), (1, nz - 1)):
        out_grad[None] += _edge_energy(psi, psi_prev, c2_const, inv_dx, dv, i, j, k, i + 1, j, k)
    for j, k in ti.ndrange((1, ny - 1), (1, nz - 1)):
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_const, inv_dx, dv, 0, j, k, 1, j, k
        )
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_const, inv_dx, dv, nx - 2, j, k, nx - 1, j, k
        )

    # y-edges.
    for i, j, k in ti.ndrange((1, nx - 1), (1, ny - 2), (1, nz - 1)):
        out_grad[None] += _edge_energy(psi, psi_prev, c2_const, inv_dx, dv, i, j, k, i, j + 1, k)
    for i, k in ti.ndrange((1, nx - 1), (1, nz - 1)):
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_const, inv_dx, dv, i, 0, k, i, 1, k
        )
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_const, inv_dx, dv, i, ny - 2, k, i, ny - 1, k
        )

    # z-edges.
    for i, j, k in ti.ndrange((1, nx - 1), (1, ny - 1), (1, nz - 2)):
        out_grad[None] += _edge_energy(psi, psi_prev, c2_const, inv_dx, dv, i, j, k, i, j, k + 1)
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_const, inv_dx, dv, i, j, 0, i, j, 1
        )
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_const, inv_dx, dv, i, j, nz - 2, i, j, nz - 1
        )


@ti.kernel
def _integrate_energy_var_c2(
    psi: ti.template(),
    psi_prev: ti.template(),
    rho: ti.template(),
    c2_field: ti.template(),
    kappa: ti.f32,
    inv_dt: ti.f32,
    inv_dx: ti.f32,
    dv: ti.f32,
    weight_shell: ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    out_kin: ti.template(),
    out_grad: ti.template(),
    out_deform: ti.template(),
):
    out_kin[None] = 0.0
    out_grad[None] = 0.0
    out_deform[None] = 0.0

    # Kinetic and deformation, interior nodes only.
    for i, j, k in ti.ndrange((1, nx - 1), (1, ny - 1), (1, nz - 1)):
        v = (psi[i, j, k] - psi_prev[i, j, k]) * inv_dt
        out_kin[None] += 0.5 * v.norm_sqr() * dv
        drho = rho[i, j, k] - 1.0
        out_deform[None] += 0.5 * kappa * drho * drho * dv

    # x-edges.
    for i, j, k in ti.ndrange((1, nx - 2), (1, ny - 1), (1, nz - 1)):
        c2_half = 0.5 * (c2_field[i, j, k] + c2_field[i + 1, j, k])
        out_grad[None] += _edge_energy(psi, psi_prev, c2_half, inv_dx, dv, i, j, k, i + 1, j, k)
    for j, k in ti.ndrange((1, ny - 1), (1, nz - 1)):
        c2_half = 0.5 * (c2_field[0, j, k] + c2_field[1, j, k])
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_half, inv_dx, dv, 0, j, k, 1, j, k
        )
        c2_half = 0.5 * (c2_field[nx - 2, j, k] + c2_field[nx - 1, j, k])
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_half, inv_dx, dv, nx - 2, j, k, nx - 1, j, k
        )

    # y-edges.
    for i, j, k in ti.ndrange((1, nx - 1), (1, ny - 2), (1, nz - 1)):
        c2_half = 0.5 * (c2_field[i, j, k] + c2_field[i, j + 1, k])
        out_grad[None] += _edge_energy(psi, psi_prev, c2_half, inv_dx, dv, i, j, k, i, j + 1, k)
    for i, k in ti.ndrange((1, nx - 1), (1, nz - 1)):
        c2_half = 0.5 * (c2_field[i, 0, k] + c2_field[i, 1, k])
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_half, inv_dx, dv, i, 0, k, i, 1, k
        )
        c2_half = 0.5 * (c2_field[i, ny - 2, k] + c2_field[i, ny - 1, k])
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_half, inv_dx, dv, i, ny - 2, k, i, ny - 1, k
        )

    # z-edges.
    for i, j, k in ti.ndrange((1, nx - 1), (1, ny - 1), (1, nz - 2)):
        c2_half = 0.5 * (c2_field[i, j, k] + c2_field[i, j, k + 1])
        out_grad[None] += _edge_energy(psi, psi_prev, c2_half, inv_dx, dv, i, j, k, i, j, k + 1)
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        c2_half = 0.5 * (c2_field[i, j, 0] + c2_field[i, j, 1])
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_half, inv_dx, dv, i, j, 0, i, j, 1
        )
        c2_half = 0.5 * (c2_field[i, j, nz - 2] + c2_field[i, j, nz - 1])
        out_grad[None] += weight_shell * _edge_energy(
            psi, psi_prev, c2_half, inv_dx, dv, i, j, nz - 2, i, j, nz - 1
        )


# Boundary-kind weights. One source of truth: the BoundaryCondition
# feature. Kept here as module constants so the mapping is one line.
_WEIGHT_SHELL_DIRICHLET_OR_REFLECTING = 1.0
_WEIGHT_SHELL_PERIODIC = 0.5


def _weight_shell_for_kind(kind: str) -> float:
    """
    Weight of the shell-interior edges for a BoundaryCondition kind.

    Module-level helper, not a method: the mapping is a pure function
    of the feature's kind and has no processor state.

    dirichlet and reflecting -> 1.0. Under dirichlet the shell is
    zero and both shell-interior edges are real; under reflecting the
    shell copies the interior neighbour, so both edges are zero and
    the weight is a formality.

    periodic -> 0.5. The shell copies the opposite face, so the two
    shell-interior edges at each end of each axis are the same wrap
    value. 0.5 on each gives the wrap energy once.
    """
    if kind in ("dirichlet", "reflecting"):
        return _WEIGHT_SHELL_DIRICHLET_OR_REFLECTING
    if kind == "periodic":
        return _WEIGHT_SHELL_PERIODIC
    raise ValueError(f"EnergyBudgetUpdate: unknown BoundaryCondition kind {kind!r}")


class EnergyBudgetUpdate(BaseProcessor):
    """
    Integrate the field's kinetic, gradient and deformation energy
    into EnergyBudget. Stage.MEASURE, order 5.

    Provides EnergyBudget (created in setup). Requires WaveGrid, the
    field named by field_type, EMCDensityField and UnitSystem. With
    use_variable_c2=True, requires WaveSpeedField and reads c^2 from
    it; otherwise reads c^2 = units.c ** 2 as a constant.

    kappa is the EMC deformation stiffness, a per-instance
    configuration. Default 1.0. The plan's "supplied by the unit
    system" is not done yet: UnitSystem has no kappa field, and adding
    one is a larger change.

    The boundary kind is read from the BoundaryCondition feature at
    process time (try_get, not require). A pipeline without the
    feature scores at the dirichlet/reflecting weight, 1.0, which
    matches a seed whose shell is zero by construction. A pipeline
    with the feature gets the weight that matches the boundary the
    BoundaryProcessor applies, since both read the same feature.

    Does not read or write any other feature. Does not enforce a
    conservation law; it reports, it does not police.
    """

    name = "EnergyBudgetUpdate"
    stage = Stage.MEASURE
    order = 5
    provides = (EnergyBudget,)

    def __init__(self, field_type=PsiLongField, use_variable_c2=False, kappa=1.0):
        self.field_type = field_type
        self.use_variable_c2 = use_variable_c2
        self.kappa = float(kappa)
        base = (WaveGrid, field_type, EMCDensityField, UnitSystem)
        if use_variable_c2:
            self.requires = base + (WaveSpeedField,)
        else:
            self.requires = base

    def setup(self, ctx):
        ctx.data.set(EnergyBudget())
        self._out_kin = ti.field(dtype=ti.f64, shape=())
        self._out_grad = ti.field(dtype=ti.f64, shape=())
        self._out_deform = ti.field(dtype=ti.f64, shape=())

    def process(self, ctx):
        grid = ctx.data.require(WaveGrid)
        units = ctx.data.require(UnitSystem)
        field = ctx.data.require(self.field_type)
        rho = ctx.data.require(EMCDensityField).rho
        budget = ctx.data.require(EnergyBudget)

        dt = ctx.sim.dt
        if dt <= 0.0:
            raise ValueError(f"EnergyBudgetUpdate: dt must be > 0, got {dt}")

        bc = ctx.data.try_get(BoundaryCondition)
        weight_shell = (
            _weight_shell_for_kind(bc.kind)
            if bc is not None
            else _WEIGHT_SHELL_DIRICHLET_OR_REFLECTING
        )

        inv_dt = 1.0 / dt
        inv_dx = 1.0 / grid.dx
        dv = grid.dx**3

        if self.use_variable_c2:
            c2_field = ctx.data.require(WaveSpeedField).c2_local
            _integrate_energy_var_c2(
                field.psi,
                field.psi_prev,
                rho,
                c2_field,
                self.kappa,
                inv_dt,
                inv_dx,
                dv,
                weight_shell,
                grid.nx,
                grid.ny,
                grid.nz,
                self._out_kin,
                self._out_grad,
                self._out_deform,
            )
        else:
            c2_const = float(units.c) ** 2
            _integrate_energy_const_c2(
                field.psi,
                field.psi_prev,
                rho,
                self.kappa,
                c2_const,
                inv_dt,
                inv_dx,
                dv,
                weight_shell,
                grid.nx,
                grid.ny,
                grid.nz,
                self._out_kin,
                self._out_grad,
                self._out_deform,
            )

        budget.E_kin = float(self._out_kin[None])
        budget.E_grad = float(self._out_grad[None])
        budget.E_deform = float(self._out_deform[None])

        if ctx.sim.step == 0:
            budget.E_prev = budget.E_total
            budget.dE_dt = 0.0
        else:
            budget.dE_dt = (budget.E_total - budget.E_prev) / dt
            budget.E_prev = budget.E_total
