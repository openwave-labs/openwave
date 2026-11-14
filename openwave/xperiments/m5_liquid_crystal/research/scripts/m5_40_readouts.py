"""M5.40 R4: the readouts of the author's deposited scripts, coded from their definitions.

    python3 m5_40_readouts.py <state.npz> [<state.npz> ...] --out readouts.json

The definitions are those of `static_preflight.py` (`axial_twist`), `zero_q2.py` and `lattice_source.py`
of the Card C payload (capstone v3.7), read and written out in m5_40_task_details.md. Nothing of the
payload is imported: the two energy terms that the readouts move are the kernels of m5_39_cardc.py
(gated against the archived evaluation in M5.39), and the moments are written here from the formulas.
A state is (p, f) per node; q = p / |p|, N = - q q^T + a a^T / G_time with a = tau(q) f.

Units of the printed values (the constants of `zero_q2.py`): e = 4.259, f_pi = 92.4 MeV, so the card
length is hbar c / (e f_pi) = 0.50143 fm and an inertia I in card units is I / (e^3 f_pi) in MeV^-1;
m_p = 938.272 MeV.

  r_S^2     the second moment of the winding density about its centroid. The density is read on the
            four Kuhn triangulations of every cell with the four-point degree-2 rule: at a rule point
            P = sum_k w_k v_k of a simplex with vertices v_0..v_3 (path order), the density is
            - o det[P; D_1; D_2; D_3] / |P|^4, D_m = (v_m - v_{m-1}) / h, o the orientation of the
            path; weight h^3 / (192 pi^2).
  L, M, Q   the isospin inertia density on cell midpoints, for the left action delta q = T_a q,
            delta N = [T_a, N]: 2 (1/4) (delta_a N : delta_b N) + 2 (1/2) sum_i ((delta_a q . delta_b q)
            |d_i q|^2 - (d_i q . delta_a q)(d_i q . delta_b q)); L its integral, M its integral against
            |x - x_c|^2, Q against 3 (z - z_c)^2 - |x - x_c|^2 (x_c the centroid of the winding density).
  r_V^2     tr M / tr L.
  gap       3 / (2 l) in MeV, l = tr L / 3.
  mu_V      (4/3) m_p |m_z| / (e^3 f_pi), m_z = dE / d(eB_z) at B = 0 of the two terms E_N + E_4 with
            the minimally linked U(1): the field at the far end of every link rotated in the (q_1, q_2)
            plane (N conjugated) by phi = (1/2) (x_from x x_to) . B; in the quartic every vertex of a
            simplex is transported to the first vertex of its path. The script averages the whole
            energy over the lattice and its three mirror images; for the tensor term that turns the
            card's two-sided edge weights into their mirror average (it differs from the card's rule
            only on edges lying along the twelve edges of the box, where a Card C state is vacuum),
            and `gauged_energy` uses the averaged weights. A central difference at eB = 5e-5 (the
            script's own), with the step 1e-4 as the check.
  g_A       the axial twist: q -> U q, N -> U N U^T with U the rotation by theta = eps x_i in the
            (q_0, q_a) plane, every other field held (the record, b(q0), c(q0), the mass term);
            D_ai = d(E_N, E_4) / d eps at 0, by a central difference at eps = 5e-5 (the script's own;
            1e-4 as the check, and the same derivative in closed form from the gradients);
            g_A = (2 / (9 e^2)) | sum_ai R_ai (D^N_ai + D^4_ai) |, R the orthogonal matrix that best
            maps the three lattice dipole fields onto the vector part of q in the shell 2.5 <= r < 2.8
            (orthogonal Procrustes; radii from the origin of the grid, not from the centroid). The
            number is reported times e^2, and at three couplings: the inherited 4.259, the printed
            4.311, and the rule's own e for the state, e^2 = 2 l Delta / (3 e f_pi) with Delta = 293 MeV
            and e f_pi the product of the two constants above.

The scripts read the stored unit `q`; here q = p / |p| is formed from the state, the same field.

Controls inside (a failing one raises): the gauged energy at B = 0 against terms 0 and 1 of the
evaluator (equal for a state with the card's boundary values); its invariance under a gauge transformation at finite B; the closed-form axial derivative
against the central difference.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m5_39_cardc as C  # noqa: E402
import m5_39_reads as R  # noqa: E402

HBARC = 197.3269804
E_INHERITED, F_PI_INHERITED, M_P = 4.259, 92.4, 938.272
L_FM = HBARC / (E_INHERITED * F_PI_INHERITED)
SCALE_I = 1.0 / (E_INHERITED ** 3 * F_PI_INHERITED)
STEPS = (1e-4, 5e-5)


def unpack(state):
    p = state[..., :4]
    q = np.ascontiguousarray(p / np.linalg.norm(p, axis=-1, keepdims=True))
    f = state[..., 4:].reshape(state.shape[:3] + (3, 3))
    return q, f


def triad(q):
    q0, q1, q2, q3 = np.moveaxis(q, -1, 0)
    return np.stack((np.stack((-q1, q0, q3, -q2), -1), np.stack((-q2, -q3, q0, q1), -1), np.stack((-q3, q2, -q1, q0), -1)), -1)


def tensor(q, f):
    a = triad(q) @ f
    return np.ascontiguousarray(-q[..., :, None] * q[..., None, :] + a @ np.swapaxes(a, -1, -2) / C.G_TIME)


class TwoTerms:
    """The tensor quadratic and the quaternion quartic of the card, with their gradients, from N and q."""

    def __init__(self, n, h):
        self.h = h
        self.zero = np.zeros((n, n, n, 3, 3))
        self.dN = np.zeros((n, n, n, 4, 4))
        self.work = [np.zeros((n, n, n, 3, 3)) for _ in range(3)]

    def tensor(self, N):
        sN = C._edge_terms(np.ascontiguousarray(N), self.zero, self.zero, self.zero, self.dN, *self.work)[0]
        return 0.5 * C.TENSOR * self.h * sN, C.TENSOR * self.h * self.dN.copy()

    def quartic(self, q):
        return C._quartic(np.ascontiguousarray(q), C.OFFSETS, self.h)


# ----------------------------------------------------------------------------------------------
# the moments of zero_q2.py
# ----------------------------------------------------------------------------------------------
def simplex_slices(n, r, o):
    return [tuple(slice(int(C.OFFSETS[r, o, m, a]), n - 1 + int(C.OFFSETS[r, o, m, a])) for a in range(3)) for m in range(4)]


def baryon_moments(q, x, h):
    """Total, centroid and second moment about the origin of the winding density, four-point rule."""
    n = q.shape[0]
    charge, vector, second = 0.0, np.zeros(3), 0.0
    weight = h ** 3 / (192.0 * np.pi ** 2)
    for r in range(4):
        for o in range(6):
            sl = simplex_slices(n, r, o)
            v = [q[s] for s in sl]
            c = [x[s] for s in sl]
            orient = np.linalg.det(np.array([C.OFFSETS[r, o, m + 1] - C.OFFSETS[r, o, m] for m in range(3)], dtype=float))
            D = [(v[m + 1] - v[m]) / h for m in range(3)]
            S, X = sum(v), sum(c)
            for i in range(4):
                P = C.SMALL * S + (C.LARGE - C.SMALL) * v[i]
                xp = C.SMALL * X + (C.LARGE - C.SMALL) * c[i]
                dens = -orient * np.linalg.det(np.stack([P] + D, axis=-2)) / np.sum(P * P, axis=-1) ** 2
                charge += weight * float(dens.sum())
                vector += weight * np.einsum("ijk,ijka->a", dens, xp)
                second += weight * float(np.einsum("ijk,ijka,ijka->", dens, xp, xp))
    return charge, vector / charge, second / charge


def iso_moments(q, f, x, h, origin):
    """L, M, Q of the isospin inertia density of the left action, on the cell midpoints."""
    qm, dq = R.midpoint(q, h)
    norm = np.linalg.norm(qm, axis=-1, keepdims=True)
    qm = qm / norm
    dq = [(d - qm * np.sum(qm * d, axis=-1, keepdims=True)) / norm for d in dq]
    fm, _ = R.midpoint(f, h)
    xm, _ = R.midpoint(x, h)
    xm = xm - origin
    r2 = np.sum(xm * xm, axis=-1)
    N = tensor(qm, fm)
    zq = [qm @ R.GEN[k].T for k in range(3)]
    zN = [R.GEN[k] @ N - N @ R.GEN[k] for k in range(3)]
    L, M, Q = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            d = 2.0 * C.TENSOR * np.sum(zN[a] * zN[b], axis=(-2, -1))
            for u in dq:
                d = d + 2.0 * C.QUARTIC * (np.sum(zq[a] * zq[b], -1) * np.sum(u * u, -1) - np.sum(u * zq[a], -1) * np.sum(u * zq[b], -1))
            L[a, b] = L[b, a] = h ** 3 * d.sum()
            M[a, b] = M[b, a] = h ** 3 * np.sum(d * r2)
            Q[a, b] = Q[b, a] = h ** 3 * np.sum(d * (3.0 * xm[..., 2] ** 2 - r2))
    return L, M, Q


# ----------------------------------------------------------------------------------------------
# the weak magnetic response of lattice_source.py
# ----------------------------------------------------------------------------------------------
def _rot_q(q, angle):
    c, s = np.cos(angle), np.sin(angle)
    out = q.copy()
    out[..., 1] = c * q[..., 1] - s * q[..., 2]
    out[..., 2] = s * q[..., 1] + c * q[..., 2]
    return out


def _conj_N(N, angle):
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    A = N.copy()
    A[..., 1, :] = c * N[..., 1, :] - s * N[..., 2, :]
    A[..., 2, :] = s * N[..., 1, :] + c * N[..., 2, :]
    B = A.copy()
    B[..., :, 1] = c * A[..., :, 1] - s * A[..., :, 2]
    B[..., :, 2] = s * A[..., :, 1] + c * A[..., :, 2]
    return B


def _phase(start, stop, B):
    return 0.5 * np.cross(start, stop) @ B


def gauged_energy(q, N, x, h, B, gauge=None):
    """E_N + E_4 with every link carrying the U(1) phase of a uniform field B (times e)."""
    n = q.shape[0]
    B = np.asarray(B, dtype=float)
    g = np.zeros(q.shape[:3]) if gauge is None else gauge
    idx = np.arange(n)
    low, high = (idx <= n - 2).astype(float), (idx >= 1).astype(float)
    eN = 0.0
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, n - 1), slice(1, n)
        lo, hi = tuple(lo), tuple(hi)
        d = _conj_N(N[hi], _phase(x[lo], x[hi], B) + g[lo] - g[hi]) - N[lo]
        t1, t2 = (axis + 1) % 3, (axis + 2) % 3
        s1, s2 = [1, 1, 1], [1, 1, 1]
        s1[t1], s2[t2] = n, n
        # the card's two-sided weights averaged with their mirror image in a transverse axis
        W = 0.5 * (low.reshape(s1) * low.reshape(s2) + high.reshape(s1) * high.reshape(s2)
                   + low.reshape(s1) * high.reshape(s2) + high.reshape(s1) * low.reshape(s2))
        eN += float(np.sum(W * np.sum(d * d, axis=(-2, -1))))
    eN *= h / 8.0
    e4 = 0.0
    for r in range(4):
        for o in range(6):
            sl = simplex_slices(n, r, o)
            v = [_rot_q(q[s], _phase(x[sl[0]], x[s], B) + g[sl[0]] - g[s]) for s in sl]
            D = np.stack([(v[m + 1] - v[m]) / h for m in range(3)], axis=-2)
            S = sum(v)
            for vert in v:
                P = C.SMALL * S + (C.LARGE - C.SMALL) * vert
                norm = np.linalg.norm(P, axis=-1, keepdims=True)
                u = (P / norm)[..., None, :]
                d = (D - np.sum(D * u, axis=-1, keepdims=True) * u) / norm[..., None]
                G = d @ np.swapaxes(d, -1, -2)
                e4 += float(np.sum(G[..., 0, 0] * G[..., 1, 1] + G[..., 0, 0] * G[..., 2, 2] + G[..., 1, 1] * G[..., 2, 2]
                                   - G[..., 0, 1] ** 2 - G[..., 0, 2] ** 2 - G[..., 1, 2] ** 2))
    return eN + C.QUARTIC * h ** 3 / 24.0 / 4.0 * e4


def magnetic_response(q, N, x, h, reference):
    """m_z = dE / d(eB_z) at the two steps, with the zero-source and the gauge controls."""
    zero = gauged_energy(q, N, x, h, np.zeros(3))
    if abs(zero - reference) > 1e-10 * abs(reference):
        raise RuntimeError(f"zero-source control failed: {zero} against {reference}")
    gauge = 0.15 * np.sin(x[..., 0]) * np.cos(x[..., 1]) + 0.13 * x[..., 2]
    Bt = np.array([0.03, -0.04, 0.02])
    gauge_error = abs(gauged_energy(q, N, x, h, Bt) - gauged_energy(_rot_q(q, gauge), _conj_N(N, gauge), x, h, Bt, gauge))
    if gauge_error > 1e-10:
        raise RuntimeError(f"gauge control failed: {gauge_error}")
    m = {}
    for step in STEPS:
        b = np.array([0.0, 0.0, step])
        m[step] = (gauged_energy(q, N, x, h, b) - gauged_energy(q, N, x, h, -b)) / (2.0 * step)
    return dict(m_z=float(m[STEPS[-1]]), m_z_at_the_coarser_step=float(m[STEPS[0]]), step_difference=float(abs(m[STEPS[0]] - m[STEPS[-1]])),
                zero_source_difference=float(abs(zero - reference)), gauge_invariance_difference=float(gauge_error))


# ----------------------------------------------------------------------------------------------
# the axial twist of static_preflight.py
# ----------------------------------------------------------------------------------------------
def alignment(q, x, h):
    """The orthogonal matrix R (a, i) of the fit of the vector part of q on the lattice dipole fields."""
    phi = R.lattice_dipole(q.shape[0], h, C.BETA2)
    r = np.linalg.norm(x, axis=-1)
    use = (r >= 2.5) & (r < 2.8)
    design, y = phi[use], q[use, 1:]
    u, _, vt = np.linalg.svd(design.T @ y)
    return (u @ vt).T, int(use.sum())


def axial_twist(q, N, x, h, terms):
    n = q.shape[0]
    align, nodes = alignment(q, x, h)

    def twisted(a, i, step):
        theta = step * x[..., i]
        co, si = np.cos(theta), np.sin(theta)
        qq = q.copy()
        qq[..., 0] = co * q[..., 0] - si * q[..., a]
        qq[..., a] = si * q[..., 0] + co * q[..., a]
        A = N.copy()
        A[..., 0, :] = co[..., None] * N[..., 0, :] - si[..., None] * N[..., a, :]
        A[..., a, :] = si[..., None] * N[..., 0, :] + co[..., None] * N[..., a, :]
        Bm = A.copy()
        Bm[..., :, 0] = co[..., None] * A[..., :, 0] - si[..., None] * A[..., :, a]
        Bm[..., :, a] = si[..., None] * A[..., :, 0] + co[..., None] * A[..., :, a]
        return np.array([terms.tensor(Bm)[0], terms.quartic(qq)[0]])

    D = {step: np.array([[(twisted(a, i, step) - twisted(a, i, -step)) / (2.0 * step) for i in range(3)] for a in (1, 2, 3)]) for step in STEPS}
    # the same derivative in closed form: sum_x x_i (dE_N / dN : (G N - N G) + dE_4 / dq . G q)
    _, gN = terms.tensor(N)
    _, gq = terms.quartic(q)
    closed = np.zeros((3, 3, 2))
    for a in (1, 2, 3):
        Gq = np.zeros_like(q)
        Gq[..., 0], Gq[..., a] = -q[..., a], q[..., 0]
        GN = np.zeros_like(N)
        GN[..., 0, :] -= N[..., a, :]
        GN[..., a, :] += N[..., 0, :]
        GN[..., :, 0] -= N[..., :, a]
        GN[..., :, a] += N[..., :, 0]
        dN = np.sum(gN * GN, axis=(-2, -1))
        dq = np.sum(gq * Gq, axis=-1)
        for i in range(3):
            closed[a - 1, i] = (np.sum(x[..., i] * dN), np.sum(x[..., i] * dq))
    fine = D[STEPS[-1]]
    scalars = 2.0 * np.einsum("ai,aik->k", align, fine) / 9.0
    closed_scalars = 2.0 * np.einsum("ai,aik->k", align, closed) / 9.0
    rel = float(abs(closed_scalars.sum() - scalars.sum()) / abs(scalars.sum()))
    if rel > 1e-6:
        raise RuntimeError(f"closed-form control failed: {closed_scalars} against {scalars}")
    return dict(alignment=align.tolist(), alignment_nodes=nodes, alignment_distance_from_identity=float(np.abs(align - np.eye(3)).max()),
                e2_gA_N_term=float(scalars[0]), e2_gA_wedge=float(scalars[1]), e2_gA=float(abs(scalars.sum())),
                e2_step_difference=float(np.max(np.abs(fine - D[STEPS[0]]))), e2_gA_closed_form=float(abs(closed_scalars.sum())),
                closed_form_relative_difference=rel, derivative_matrix_fine=fine.tolist(), n=n)


# ----------------------------------------------------------------------------------------------
def frozen_rule_e(inertia, delta_mev=293.0):
    """The rule of the text: e^2 = 2 I Delta / (3 e f_pi), with e f_pi the product of the script's constants."""
    return float(np.sqrt(2.0 * inertia * delta_mev / (3.0 * E_INHERITED * F_PI_INHERITED)))


def readouts(path, axial=True):
    with np.load(path, allow_pickle=False) as data:
        state, x = data["state"], data["coords"]
    n = state.shape[0]
    h = float(x[1, 0, 0, 0] - x[0, 0, 0, 0])
    q, f = unpack(state)
    N = tensor(q, f)
    terms = TwoTerms(n, h)
    eN, eP = terms.tensor(N)[0], terms.quartic(q)[0]
    Bq, center, s0 = baryon_moments(q, x, h)
    L, M, Q = iso_moments(q, f, x, h, center)
    l = float(np.trace(L) / 3.0)
    rS2 = float((s0 - center @ center) * L_FM ** 2)
    rV2 = float(np.trace(M) / np.trace(L) * L_FM ** 2)
    mag = magnetic_response(q, N, x, h, eN + eP)
    v = 4.0 * M_P * abs(mag["m_z"]) * SCALE_I / 3.0
    rigid = 2.0 * M_P * l * SCALE_I / 3.0
    out = dict(file=Path(path).name, grid=n, spacing=h, half_box=float(x[-1, 0, 0, 0]),
               constants=dict(e=E_INHERITED, f_pi_MeV=F_PI_INHERITED, m_p_MeV=M_P, length_fm=L_FM),
               tensor_quadratic=float(eN), quaternion_quartic=float(eP),
               winding_by_the_rule=float(Bq), winding_centroid=center.tolist(),
               inertia_matrix=L.tolist(), inertia_mean=l, inertia_r2_matrix=M.tolist(), inertia_quadrupole_matrix=Q.tolist(),
               rS2_fm2=rS2, rV2_fm2=rV2, rV2_by_axis_fm2=(np.diag(M) / np.diag(L) * L_FM ** 2).tolist(),
               gap_MeV=1.5 / (l * SCALE_I), magnetic=mag, muV=float(v), muV_rigid=float(rigid), muV_deviation_fraction=float(v / rigid - 1.0),
               Q_transition_fm2=float(np.sqrt(2.0) / (6.0 * l) * (Q[2, 2] - (Q[0, 0] + Q[1, 1]) / 2.0) * L_FM ** 2),
               frozen_rule_e=frozen_rule_e(l))
    if axial:
        ax = axial_twist(q, N, x, h, terms)
        ax["gA_at_e"] = {"inherited, 4.259": ax["e2_gA"] / E_INHERITED ** 2, "printed for the 55^3 reset, 4.311": ax["e2_gA"] / 4.311 ** 2,
                         "the rule's own e of this state": ax["e2_gA"] / out["frozen_rule_e"] ** 2}
        out["axial"] = ax
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-axial", action="store_true")
    a = ap.parse_args()
    doc = {}
    for p in a.states:
        r = readouts(p, not a.no_axial)
        doc[p.stem] = r
        brief = {k: r[k] for k in ("file", "winding_by_the_rule", "inertia_mean", "rS2_fm2", "rV2_fm2", "gap_MeV", "muV", "muV_rigid", "frozen_rule_e")}
        brief["magnetic"] = r["magnetic"]
        if "axial" in r:
            brief["axial"] = {k: r["axial"][k] for k in ("e2_gA_N_term", "e2_gA_wedge", "e2_gA", "e2_step_difference", "closed_form_relative_difference",
                                                         "alignment_distance_from_identity", "gA_at_e")}
        print(json.dumps(brief, indent=1), flush=True)
        if a.out:
            a.out.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
