"""Item E.2, support: checks on the group itself, with explicit D^j (su2.py) and Gamma_0 rebuilt from q1, q2.

For each rigid slot (n = dim - 1, pi = D^{n/2}, rho = pi|Gamma_0, eta = I):
  (1) slot equivariance psi_u(g h) = psi_u(g) rho(h), h = q1, q2, at random g in SU(2);
  (2) pointwise |psi_v(g)| = |v| and N(psi_z)(g) = psi_{|z|^2 z}(g) at random g (the reduction s03 uses);
  (3) Haar quadrature on SU(2) (Euler angles; exact for these band-limited integrands) reproduces Schur
      orthogonality on E_0 (Gram = I/d);
  (4) decoupling: for zeta a section at the next level L of E.1 (eta' in Hom_Gamma(rho, V_{L/2}) by
      group averaging), the E_0 component of the linearized cubic term |v|^2 zeta + 2 Re(zeta psi_v^dag) psi_v
      vanishes. (Symmetry of the real-linear operator then makes E_0 an invariant block, both ways.)
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from q5 import qmul, ONE, q1, q2
import su2
from checks import check, summary, PLANT

rig = json.load(open(os.path.join(HERE, "out", "s02.json")))
rng = np.random.default_rng(11)

# Gamma_0, numerically, from the exact closure
G = [ONE]; seen = {ONE}; fr = [ONE]
while fr:
    nx = []
    for x in fr:
        for s in (q1, q2):
            y = qmul(x, s)
            if y not in seen:
                seen.add(y); G.append(y); nx.append(y)
    fr = nx
GU = [su2.qmat(x) for x in G]


def rand_su2():
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    return su2.qmat(q)


def euler_grid(Kmax):
    Ma = int(np.ceil(2 * Kmax)) + 2
    Mg = int(np.ceil(4 * Kmax)) + 2
    Nb = int(np.ceil(Kmax)) + 3
    al = 2 * np.pi * np.arange(Ma) / Ma
    ga = 4 * np.pi * np.arange(Mg) / Mg
    xb, wb = np.polynomial.legendre.leggauss(Nb)  # in cos(beta)
    be = np.arccos(xb)
    W = (np.ones(Ma) / Ma)[:, None, None] * (wb / 2)[None, :, None] * (np.ones(Mg) / Mg)[None, None, :]
    return al, be, ga, W


def section_values(twoj, x, eta, al, be, ga):
    """psi(g)_a = sum x_m D_mm'(g) eta_m'a on the grid; D = diag(e^{-i m al}) d(be) diag(e^{-i m ga})."""
    m = su2.mvals(twoj)
    dB = np.array([su2.D(twoj, su2.Ry(b)) for b in be])  # [Nb, n+1, n+1]
    y = x[None, :] * np.exp(-1j * np.outer(al, m))          # [Na, n+1]
    z = np.einsum("am,bmk->abk", y, dB)                     # [Na, Nb, n+1]
    z = z[:, :, None, :] * np.exp(-1j * np.outer(ga, m))[None, None, :, :]
    return z @ eta                                          # [Na, Nb, Ng, d]


# Euler factorization check (convention)
a0, b0, g0 = 0.7, 1.1, 2.9
U = su2.Rz(a0) @ su2.Ry(b0) @ su2.Rz(g0)
err = max(np.abs(su2.D(t, U) - np.diag(np.exp(-1j * su2.mvals(t) * a0)) @ su2.D(t, su2.Ry(b0)) @ np.diag(np.exp(-1j * su2.mvals(t) * g0))).max() for t in range(13))
check("conv: D(Rz Ry Rz) = diag . d(beta) . diag", err < 1e-12, f"{err:.1e}")

out = {}
for lab, info in rig.items():
    n, d, L = info["n"], info["dim"], info["next level"]
    rho = lambda U: su2.D(n, U)
    # (1) equivariance
    e1 = 0.0
    for _ in range(5):
        Ug = rand_su2(); u = rng.normal(size=d) + 1j * rng.normal(size=d)
        for h in (q1, q2):
            Uh = su2.qmat(h)
            e1 = max(e1, np.abs(u @ su2.D(n, Ug @ Uh) - (u @ su2.D(n, Ug)) @ rho(Uh)).max())
    check(f"E.2 {lab}: psi_u(gh) = psi_u(g) rho(h)", e1 < 1e-10, f"{e1:.1e}")
    # (2) pointwise modulus and cubic closure
    e2 = 0.0
    for _ in range(20):
        Ug = rand_su2(); z = rng.normal(size=d) + 1j * rng.normal(size=d)
        pz = z @ su2.D(n, Ug)
        if PLANT == "nonunitary":
            pz = pz * (1 + 0.1 * np.cos(np.angle(Ug[0, 0])))  # planted: a non-isometric "section"
        e2 = max(e2, abs(np.vdot(pz, pz).real - np.vdot(z, z).real),
                 np.abs(np.vdot(pz, pz).real * pz - (np.vdot(z, z).real * z) @ su2.D(n, Ug)).max())
    check(f"E.2 {lab}: |psi_z(g)| = |z| and N(psi_z) = psi_(|z|^2 z) pointwise", e2 < 1e-10, f"{e2:.1e}")
    # eta' at level L by averaging over Gamma_0
    X = rng.normal(size=(L + 1, d)) + 1j * rng.normal(size=(L + 1, d))
    etaL = sum(su2.D(L, Uh) @ X @ np.linalg.inv(rho(Uh)) for Uh in GU) / len(GU)
    e3 = max(np.abs(su2.D(L, Uh) @ etaL - etaL @ rho(Uh)).max() for Uh in GU)
    check(f"E.2 {lab}: eta' at level {L} intertwines and is nonzero", e3 < 1e-10 and np.abs(etaL).max() > 1e-3, f"err {e3:.1e}, |eta'| {np.abs(etaL).max():.2f}")
    # (3)+(4) quadrature
    Kmax = L / 2 + 2 * n + 1
    al, be, ga, W = euler_grid(Kmax)
    basis = []
    for k in range(d):
        e = np.zeros(d, complex); e[k] = 1
        basis.append(section_values(n, e, np.eye(d), al, be, ga))
    gram = np.array([[np.sum(W * np.einsum("...a,...a->...", basis[p], basis[q].conj())) for q in range(d)] for p in range(d)])
    # Schur: int D_ka conj(D_k'a') = delta delta / d ; summed over the row index a this gives Gram = I.
    # (A first version of this check expected I/d and failed; the expectation, not the quadrature, was wrong.)
    e4 = np.abs(gram - np.eye(d)).max()
    single = np.array([[np.sum(W * basis[p][..., 0] * basis[q][..., 0].conj()) for q in range(d)] for p in range(d)])
    e4b = np.abs(single - np.eye(d) / d).max()
    check(f"E.2 {lab}: quadrature Gram of E_0 = I and int D_k0 conj D_k'0 = delta/d (Schur)", max(e4, e4b) < 1e-12, f"{e4:.1e}, {e4b:.1e}")
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    psv = section_values(n, v, np.eye(d), al, be, ga)
    wz = rng.normal(size=L + 1) + 1j * rng.normal(size=L + 1)
    zeta = section_values(L, wz, etaL, al, be, ga)
    if PLANT == "inE0":
        zeta = section_values(n, rng.normal(size=d) + 1j * rng.normal(size=d), np.eye(d), al, be, ga)
    nv2 = np.vdot(v, v).real
    Lz = nv2 * zeta + 2 * np.einsum("...a,...a->...", zeta, psv.conj()).real[..., None] * psv
    scale = np.sqrt(np.sum(W * np.einsum("...a,...a->...", Lz, Lz.conj()).real))
    proj = []
    for k in range(d):
        for ph in (1, 1j):
            proj.append(np.sum(W * np.einsum("...a,...a->...", Lz, (ph * basis[k]).conj()).real))
    rel = np.abs(proj).max() / scale
    check(f"E.2 {lab}: E_0 component of the linearized cubic term on a level-{L} zeta vanishes", rel < 1e-12, f"max |<L zeta, e>|/|L zeta| = {rel:.1e}")
    out[lab] = {"next level used": L, "relative E0 component": float(rel)}
if PLANT is None:
    json.dump(out, open(os.path.join(HERE, "out", "s04.json"), "w"), indent=1)
summary()
