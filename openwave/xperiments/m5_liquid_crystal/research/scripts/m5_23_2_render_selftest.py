"""M5.23.2 — the render-package selftest (loader / tracer / isosurface / demo).

Gates the four arms of the render package against the research references,
per-gap (the M5.24 selftest pattern):

  L — arm (3), the research-ENDPOINT loader (engine1_seeds.load_npz_M):
      grid fit (crop / embed), the covariant sign convention (measured:
      V4 = 759.4/cell at the +g storage form vs ~0 flipped), kin anchor on
      the loaded fixed-J endpoint, and a live production hold.
  T — arm (1), the disclination-line tracer (m5_23_2_tracer): finds the
      polar rods on the certified electron endpoint from the field alone,
      closes on a charged-ring seed, Stage D split-value anchors.
  I — arm (4), the energy isosurface (taichi marching cubes): sphere
      area anchor, closed surface on the electron state, budget honesty.
  D — arm (2), the J/mu twist demo: the rod-sample eigenframe advances at
      the M5.23.1 visible rate under SET-J on the demo xparameter flow.

Run:  cd openwave && /opt/anaconda3/envs/master312/bin/python \
      openwave/xperiments/m5_liquid_crystal/research/scripts/m5_23_2_render_selftest.py
"""

import json
import math
import os
import sys

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, log_level=ti.WARN, random_seed=0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from openwave.xperiments.m5_liquid_crystal import engine2_pde as pde  # noqa: E402
from openwave.xperiments.m5_liquid_crystal import engine1_seeds as seeds  # noqa: E402
from openwave.xperiments.m5_liquid_crystal.medium import TensorField  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

sys.path.insert(0, HERE)
import m5_23_2_tracer as tracer  # noqa: E402

fails, total = [], [0]


def check(name, ok, detail=""):
    total[0] += 1
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


W1 = pde.W1_SPECTRAL
DT = 0.005
H_RES = 1.5
RENV = 10.0

# ================================================================
# L — arm (3): the research-ENDPOINT loader
# ================================================================
print("[L] the endpoint loader (engine1_seeds.load_npz_M)")

FIXJ = os.path.join(DATA, "m5_21_9_fixedj_conj_om0.2_end.npz")
with open(os.path.join(DATA, "m5_21_9_fixedj_conj_om0.2.json")) as f:
    REC = json.load(f)
KIN_ANCHOR = REC["final"]["kin_final"]  # 0.121014
OM_STAR = REC["final"]["omega_star_final"]

# L1 — 31^3 CROP of the 32^3 4x4 endpoint + the covariant flip
N1 = 31
tf1 = TensorField([N1 * H_RES * 1e-18] * 3, N1**3)
assert tf1.nx == N1, f"grid {tf1.nx} != {N1}"
info1 = seeds.load_npz_M(tf1, FIXJ)
src = np.load(FIXJ)["M"].astype(np.float32)
got = tf1.M_am.to_numpy()
flip_view = src[:N1, :N1, :N1].copy()
flip_view[..., 0, 0] *= -1.0
err1 = float(np.abs(got - flip_view).max())
check(
    "L1 crop-fit load + covariant flip (4x4: storage +g -> -g)",
    info1["ok"]
    and info1["fit"] == "crop"
    and info1["covariant"]
    and err1 == 0.0
    and float(got[0, 0, 0, 0, 0]) < 0.0,
    f"fit {info1['fit']}, max|diff| {err1:.1e}, far M00 {got[0, 0, 0, 0, 0]:+.1f}",
)

# L2 — production kin on the loaded state == the research anchor
raw_norm = pde.compute_clock_flow(tf1, H_RES, RENV)
kin_l = pde.kin_canonical(tf1, H_RES)
rel2 = abs(kin_l - KIN_ANCHOR) / KIN_ANCHOR
check(
    "L2 kin anchor on the loaded endpoint (crop arena)",
    rel2 < 5e-3,
    f"kin {kin_l:.6f} vs recorded {KIN_ANCHOR:.6f} (rel {rel2:.2e}; "
    f"crop removes one far plane)",
)

# L3 — live production hold on the loaded state: SET-J + 100 steps
info_set = pde.set_fixed_j(tf1, H_RES, RENV, OM_STAR, DT, shell=1)
js0 = pde.read_carried_j(tf1, H_RES, RENV, DT)
G1 = float(tf1.lc_g)
D1 = float(tf1.lc_delta)
for _ in range(100):
    pde.compute_eta_flux(tf1, 0, H_RES)
    pde.evolve_M_eta_start(tf1, DT, H_RES)
    pde.compute_eta_flux(tf1, 1, H_RES)
    pde.evolve_M_eta_finish(tf1, DT, H_RES, G1, D1, W1)
    tf1.swap_matrix_buffers()
js1 = pde.read_carried_j(tf1, H_RES, RENV, DT)
m_l3 = tf1.M_am.to_numpy()
check(
    "L3 loaded state is LIVE: SET-J + 100-step production hold bounded, J coherent",
    info_set["set"]
    and np.isfinite(m_l3).all()
    and np.abs(m_l3).max() < 50.0
    and abs(js0 - OM_STAR) / OM_STAR < 1e-2
    and js1 / js0 > 0.5,
    f"J_self {js0:+.5f} -> {js1:+.5f}, max|M| {np.abs(m_l3).max():.1f}",
)

# L4 — 3x3 metadata state EMBED into a 63^3 arena + the launcher-path flip
T32 = os.path.join(DATA, "m5_21_6_end_t32_A.npz")
N4 = 63
tf4 = TensorField([N4 * H_RES * 1e-18] * 3, N4**3)
info4 = seeds.load_npz_M(tf4, T32, delta_cfg=0.3)
pde.flip_time_axis(tf4)  # the canonical activation's own flip (3x3 path)
src4 = np.load(T32)
m_src4 = src4["M"].astype(np.float32)
got4 = tf4.M_am.to_numpy()
off = (N4 - 32) // 2
int_err = float(
    np.abs(got4[off : off + 32, off : off + 32, off : off + 32, 1:, 1:] - m_src4).max()
)
pad_ok = float(np.abs(got4[0, off + 16, off + 16, 1:, 1:] - m_src4[0, 16, 16]).max())
check(
    "L4 embed-fit 3x3 load (t32 mu-family state) + launcher flip",
    info4["ok"]
    and info4["fit"].startswith("embed")
    and not info4["covariant"]
    and info4["h"] == 1.5
    and int_err == 0.0
    and float(got4[0, 0, 0, 0, 0]) == -float(tf4.lc_g)
    and pad_ok == 0.0,
    f"fit {info4['fit']}, h {info4['h']}, interior max|diff| {int_err:.1e}, "
    f"edge-extend anchor {pad_ok:.1e}, far M00 {got4[0, 0, 0, 0, 0]:+.1f}",
)

# ================================================================
# T — arm (1): the disclination-line tracer
# ================================================================
print("[T] the disclination-line tracer (m5_23_2_tracer)")

# T1 — the certified electron endpoint: rods FOUND from the field alone
M_e = np.load(FIXJ)["M"].astype(np.float64)[..., 1:, 1:]  # spatial block
res_t1 = tracer.trace(M_e, h=H_RES)
n_lines = len(res_t1["lines"])
core_vox = sum(ln["n_vox"] for ln in res_t1["lines"])
check(
    "T1 electron endpoint: defect line(s) found, no seed knowledge",
    n_lines >= 1 and core_vox >= 3,
    f"{n_lines} line(s), {core_vox} core voxels, bulk split {res_t1['bulk_split']:.3f}, "
    f"thr {res_t1['thr']:.3f}",
)

# T2 — closure: a charged-ring seed CLOSES; the rod state does not
N_R = 63
tfr = TensorField([N_R * 1.5 * 1e-18] * 3, N_R**3)
seeds.seed_charged_ring_M(tfr, N_R // 2, N_R // 2, N_R // 2, 0.10 * N_R, 3.0, 3.0, 0.3)
M_ring = tfr.M_am.to_numpy().astype(np.float64)[..., 1:, 1:]
res_ring = tracer.trace(M_ring, h=1.0)
ring_closed = [ln for ln in res_ring["lines"] if ln["verdict"] == "closed-loop"]
check(
    "T2 charged-ring seed: the traced cord CLOSES (Euler-characteristic verdict)",
    len(ring_closed) >= 1,
    f"lines {[(ln['n_vox'], ln['verdict'], ln['chi']) for ln in res_ring['lines']]}",
)

# T3 — Stage D split anchors: ~0 on the defect core vs ~delta in the bulk
core_min = res_t1["split_min_on_lines"]
bulk = res_t1["bulk_split"]
check(
    "T3 Stage D anchors: core split ~ 0, bulk split ~ delta",
    core_min < 0.05 and 0.2 < bulk < 0.4,
    f"core min {core_min:.4f} vs bulk {bulk:.3f} (Stage D: 0.000 vs ~0.265)",
)

# ================================================================
# I — arm (4): the energy-density isosurface (marching tetrahedra)
# ================================================================
print("[I] the energy isosurface (engine4_render.marching_tetrahedra)")

from openwave.xperiments.m5_liquid_crystal import engine4_render as viz  # noqa: E402
from openwave.xperiments.m5_liquid_crystal.medium import FieldObservables  # noqa: E402


def mesh_area_and_verts(tf_mesh):
    n_tri = min(int(tf_mesh.iso_tri_count[None]), tf_mesh.iso_max_tris)
    v = tf_mesh.iso_vertices.to_numpy()[: 3 * n_tri].reshape(n_tri, 3, 3).astype(np.float64)
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
    return n_tri, area, v


def open_edge_count(v_tris, tol_render=1e-4):
    """Count edges not shared by exactly two triangles (0 = closed 2-manifold).

    Vertices are identified by TOLERANCE clustering (cKDTree union-find),
    not fixed-grid rounding: adjacent cells compute the same shared vertex
    through different f32 arithmetic ((i+0.5)/N + 1/N vs (i+1.5)/N), so a
    rounding grid at any resolution mislabels the jitter-straddlers (~2.6%
    of edges measured at try 1). tol_render ~ 6e-3 voxel at a 63-grid:
    orders below the mesh's distinct-vertex separation."""
    from collections import Counter

    from scipy.spatial import cKDTree

    pts = v_tris.reshape(-1, 3)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(tol_render, output_type="ndarray")
    parent = np.arange(len(pts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    labels = np.array([find(a) for a in range(len(pts))])
    cnt = Counter()
    for t in range(len(v_tris)):
        va, vb, vc = labels[3 * t], labels[3 * t + 1], labels[3 * t + 2]
        if va == vb or vb == vc or vc == va:
            continue  # sliver (level through a tet vertex): zero-area, no boundary
        for a, b in ((va, vb), (vb, vc), (vc, va)):
            cnt[frozenset((int(a), int(b)))] += 1
    return sum(1 for c in cnt.values() if c != 2)


# I1 — sphere anchor: linear-in-r density, exact spherical level set
NS = 63
tfs = TensorField([NS * 1.0 * 1e-18] * 3, NS**3)
obs_s = FieldObservables((NS, NS, NS))
ctr = (NS - 1) / 2.0
ii, jj, kk = np.indices((NS, NS, NS)).astype(np.float64)
r_vox = np.sqrt((ii - ctr) ** 2 + (jj - ctr) ** 2 + (kk - ctr) ** 2)
R_BIG = 24.0
dens = np.maximum(R_BIG - r_vox, 0.0)
dens[0, :, :] = dens[-1, :, :] = 0.0
dens[:, 0, :] = dens[:, -1, :] = 0.0
dens[:, :, 0] = dens[:, :, -1] = 0.0
obs_s.energyH_density_aJ.from_numpy(dens.astype(np.float32))
viz.iso_density_max(tfs, obs_s)
dmax = float(tfs.iso_level_max[None])
R_TGT = 15.3  # NON-lattice radius: exact-hit corners (r = 15 at (15,0,0),
# (9,12,0), ...) put the level exactly on tet vertices -> degenerate slivers
level = float(R_BIG - R_TGT)  # sphere of radius R_TGT voxels
viz.marching_tetrahedra(tfs, obs_s, level)
viz.collapse_iso_tail(tfs)
n_tri, area, v_tris = mesh_area_and_verts(tfs)
area_true = 4.0 * math.pi * (R_TGT / NS) ** 2  # render units (max_dim = NS)
a_err = abs(area - area_true) / area_true
verts = v_tris.reshape(-1, 3) * NS - 0.5
r_verts = np.linalg.norm(verts - ctr, axis=1)
r_dev = float(np.abs(r_verts - R_TGT).max())
check(
    "I1 sphere anchor: area within 2%, every vertex within one voxel of the surface",
    n_tri > 0 and a_err < 0.02 and r_dev < 1.0,
    f"{n_tri} tris, area {area:.5f} vs 4piR^2 {area_true:.5f} (rel {a_err:.2e}), "
    f"max radial dev {r_dev:.2f} vox, dmax {dmax:.1f}",
)
check(
    "I1b sphere: closed 2-manifold (every edge shared by exactly 2 triangles)",
    open_edge_count(v_tris) == 0,
    f"open edges {open_edge_count(v_tris)}",
)

# I2 — the electron state: no interior cracks; openings ONLY at clip planes.
# MEASURED at this build (re-specifies the planned "closed surface" gate —
# that expectation was WRONG for this state): the electron's energyH
# density concentrates along the DISCLINATION ROD, which runs boundary to
# boundary (the tracer's own "boundary" verdict, T1) — so its energy
# isosurface is an OPEN TUBE around the rod at every level fraction
# (above-level extent k = 1..29 measured at fractions 0.35-0.9), clipped at
# the marched-region edge like any marcher. Also measured: the grid density
# max sits at the rod / pin-shell boundary junction ((15,16,1), 1.06e-2 =
# 4.2x the core peak) — hence iso_density_max's 3-voxel interior margin.
# The honest watertightness statement: every open edge lies AT a clip
# plane; the surface has NO interior cracks.
def open_edge_positions(v_tris, tol_render=1e-4):
    """Midpoints (render units) of edges not shared by exactly 2 triangles."""
    from collections import defaultdict

    from scipy.spatial import cKDTree

    pts = v_tris.reshape(-1, 3)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(tol_render, output_type="ndarray")
    parent = np.arange(len(pts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    labels = np.array([find(a) for a in range(len(pts))])
    edges = defaultdict(list)
    for t in range(len(v_tris)):
        va, vb, vc = labels[3 * t], labels[3 * t + 1], labels[3 * t + 2]
        if va == vb or vb == vc or vc == va:
            continue  # sliver
        for a, b, pa, pb in (
            (va, vb, 3 * t, 3 * t + 1),
            (vb, vc, 3 * t + 1, 3 * t + 2),
            (vc, va, 3 * t + 2, 3 * t),
        ):
            edges[frozenset((int(a), int(b)))].append(0.5 * (pts[pa] + pts[pb]))
    return np.array([mids[0] for key, mids in edges.items() if len(mids) != 2]).reshape(-1, 3)


tfe = TensorField([31 * H_RES * 1e-18] * 3, 31**3)
seeds.load_npz_M(tfe, FIXJ)
obs_e = FieldObservables((31, 31, 31))
observables_mod = __import__(
    "openwave.xperiments.m5_liquid_crystal.engine3_observables", fromlist=["x"]
)
observables_mod.compute_energyH_density_eta(
    tfe, obs_e, DT, H_RES, float(tfe.lc_g), float(tfe.lc_delta), W1, 1.0
)
viz.iso_density_max(tfe, obs_e)
dmax_e = float(tfe.iso_level_max[None])  # interior-margin max (core scale)
viz.marching_tetrahedra(tfe, obs_e, 0.8 * dmax_e)
viz.collapse_iso_tail(tfe)
n_e, area_e, v_e = mesh_area_and_verts(tfe)
raw_e = int(tfe.iso_tri_count[None])
open_mid = open_edge_positions(v_e)
if len(open_mid):
    vox_mid = open_mid * 31 - 0.5
    clip_lo, clip_hi = 1.0 + 0.6, 29.0 - 0.6  # clip planes of marched cells [1, 28]
    at_clip = (vox_mid.min(axis=1) < clip_lo) | (vox_mid.max(axis=1) > clip_hi)
    n_cracks = int((~at_clip).sum())
else:
    n_cracks = 0
check(
    "I2 electron endpoint: rod-tube surface, NO interior cracks (openings only at clip planes)",
    0 < n_e < tfe.iso_max_tris and raw_e == n_e and len(open_mid) > 0 and n_cracks == 0,
    f"{n_e} tris (raw {raw_e}), {len(open_mid)} clip-plane open edges, "
    f"{n_cracks} interior cracks, interior dmax {dmax_e:.3e}",
)

# I3 — the ellipsoid-covered variant: strided samples ON the surface
viz.marching_tetrahedra(tfs, obs_s, level)  # back to the sphere
viz.collapse_iso_tail(tfs)
N_COVER = 500
viz.update_iso_ellipsoids(tfs, 0.02, N_COVER)
ell_v = tfs.ellipsoid_mesh_vertices.to_numpy()
tverts = tfs.ellipsoid_tverts
n_slots_total = tfs.ellipsoid_max_centers * tfs.ellipsoid_max_dirs
slot_v = ell_v.reshape(n_slots_total, tverts, 3)
active = np.abs(slot_v).sum(axis=(1, 2)) > 0
cent = slot_v[active].mean(axis=1) * NS - 0.5
r_cent = np.linalg.norm(cent - ctr, axis=1)
cover_dev = float(np.abs(r_cent - R_TGT).max()) if active.any() else 99.0
check(
    "I3 iso-cover: sample budget respected, every sample centered on the surface",
    int(active.sum()) == N_COVER and cover_dev < 1.5,
    f"{int(active.sum())}/{N_COVER} active samples, max radial dev {cover_dev:.2f} vox",
)

# ================================================================
# D — arm (2): the J/mu twist demo (headless verification)
# ================================================================
print("[D] the twist demo (rod-sample axis rotation + xparameter configs)")

# D1 — the rod-sample delta frame advances at the visible rate under the
# carried clock (the loaded electron + SET-J from L3-style flow, fresh)
tfd = TensorField([31 * H_RES * 1e-18] * 3, 31**3)
seeds.load_npz_M(tfd, FIXJ)
info_d = pde.set_fixed_j(tfd, H_RES, RENV, OM_STAR, DT, shell=1)


def probe_axis(tf_real, iv, jv, kv):
    m = tf_real.M_am.to_numpy().astype(np.float64)[iv, jv, kv, 1:4, 1:4]
    lam, V = np.linalg.eigh(m)
    return V[:, 1]  # middle eigenvector = the delta clock-hand axis


# the S5-proven probe (r = 4 vox = 6 research units, inside the clock
# envelope w ~ 0.88). Try-1 lesson (real physics, kept as a demo note):
# at a ring-row position (offset (4,0,5), r = 9.6 units, w ~ 0.43) the
# measured rate was 6x slower — the envelope scales the LOCAL twist rate,
# so the outer ring samples visibly lag the core. The gate probes where
# the rate prediction omega*/|a0_raw| was measured.
probe_d = (16 + 4, 16, 16)
ax0_d = probe_axis(tfd, *probe_d)
SUBS_D = 64
N_FR = 5
phis_d = []
for _ in range(N_FR):
    for _ in range(SUBS_D):
        pde.compute_eta_flux(tfd, 0, H_RES)
        pde.evolve_M_eta_start(tfd, DT, H_RES)
        pde.compute_eta_flux(tfd, 1, H_RES)
        pde.evolve_M_eta_finish(tfd, DT, H_RES, float(tfd.lc_g), float(tfd.lc_delta), W1)
        tfd.swap_matrix_buffers()
    axf = probe_axis(tfd, *probe_d)
    c = float(np.clip(abs(np.dot(ax0_d, axf)), 0.0, 1.0))
    phis_d.append(math.acos(c))
t_d = N_FR * SUBS_D * DT
rate_d = max(phis_d) / t_d
rate_pred = info_d["om_star"] / max(info_d["a0_raw_norm"], 1e-9)
check(
    "D1 twist demo: the rod-sample delta axis advances at the visible rate",
    max(phis_d) > 0.01 and 0.3 < rate_d / rate_pred < 3.0,
    f"axis advance {max(phis_d):.4f} rad over t = {t_d:.1f} "
    f"({rate_d:.4f} rad/tau vs omega*/|a0_raw| = {rate_pred:.4f} predicted)",
)

# D2 — the four demo xparameter configs: importable + wired to this build
XP = os.path.join(HERE, "..", "..", "xparameters")
sys.path.insert(0, XP)
import importlib  # noqa: E402

cfg_checks = []
for mod_name, need in (
    ("_topo_fixedj_rods", {"MODE": "biaxial_hedgehog", "FIXEDJ_OMEGA": 0.2}),
    ("_topo_npz_electron", {"MODE": "npz_file", "FIXEDJ_OMEGA": 0.19923}),
    ("_topo_npz_mu", {"MODE": "npz_file", "FIXEDJ_OMEGA": 0.0}),
    ("_topo_npz_tau", {"MODE": "npz_file", "FIXEDJ_OMEGA": 0.0}),
):
    m_cfg = importlib.import_module(mod_name)
    topo = m_cfg.TOPOLOGY_SEED
    ok_cfg = all(topo.get(k) == v for k, v in need.items())
    ok_cfg = ok_cfg and topo.get("INTEGRATOR_4D") == "canonical"
    if topo.get("MODE") == "npz_file":
        p_cfg = os.path.join(HERE, "..", "..", topo["PATH"])
        ok_cfg = ok_cfg and os.path.isfile(p_cfg)
        n_cfg = int(round(m_cfg.TARGET_VOXELS ** (1.0 / 3.0)))
        n_src_cfg = int(np.load(p_cfg)["M"].shape[0])
        ok_cfg = ok_cfg and (n_cfg - 1 == n_src_cfg - 1 or n_cfg - 1 >= n_src_cfg)
    cfg_checks.append((mod_name, ok_cfg))
check(
    "D2 demo xparameters: importable, canonical, files on disk, grids fit",
    all(ok for _, ok in cfg_checks),
    ", ".join(f"{n} {'ok' if ok else 'BAD'}" for n, ok in cfg_checks),
)

print(flush=True)
if fails:
    print(f"RESULT: {len(fails)}/{total[0]} FAILED: {fails}")
    raise SystemExit(1)
print(f"RESULT: ALL {total[0]} CHECKS GREEN")
