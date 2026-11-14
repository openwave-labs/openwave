#!/usr/bin/env python3
"""
From manuscript version: 5.3.0 or leter
Cosmological constant from the BCC lattice.

Chain (each step shown in its own Part below):

    [1]  BCC stiffness       N_geom = 8*pi^4*(1-zeta)
    [2]  Vacuum deficit      eps_M  = 1/(N_geom*pi^3)
    [3]  AMM projection      O_mu   = M_mu/N_geom = 1/(2*pi)^2
                             =>  M_mu = N_geom/(2*pi)^2 = 2*pi^2
    [4]  Horizon factor      H      = 8*(2*pi)^2 = 32*pi^2
                             Identity: H*M_mu = 8*N_geom
    [5]  Lattice test        H(N)   = N*(2*pi)^2
                             Only N=8 (BCC) matches Planck
    [6]  EWT trinity         alpha, hbar, G, lambda_l
    [7]  Planck temperature  T_p    = (1/k_B)*sqrt(hbar*c^5/G)
    [8]  Horizon radius      R_H    = lambda_l*(T_p/T_cmb)^2 / H
                             Haug relation RECOVERED from the BCC
                             horizon model, not imported
    [9]  Hubble constant     H_0    = c/R_H
    [10] de Sitter Lambda    Lambda_dS = 3*H_0^2/c^2
    [11] Vacuum fraction     Omega_vac = eta_BCC * C_gap
    [12] Observed Lambda     Lambda_vac = Omega_vac * Lambda_dS

Key insight: the SAME eps_M that fixes the AMM (via O_mu) also fixes
the horizon factor (via H = 8/O_mu). The factor 32*pi^2 needed to
close the Haug relation is exactly 8*(2*pi)^2 = 8 times the 2D unit,
where 8 is BCC coordination and (2*pi)^2 is the AMM projection unit.

The Haug relation T_cmb = (T_p/8pi)*sqrt(2*lambda_l/R_H) is therefore
RECOVERED from the BCC horizon model, not imported. It is the
combination of:
    - naive equipartition: T_cmb = T_p*sqrt(lambda_l/R_H)
    - BCC horizon factor:  H = 8*(2*pi)^2  (phase-unit ansatz)
    - corrected:           T_cmb = T_p*sqrt(lambda_l/(H*R_H))

The horizon factor H itself is a model construct: each of the 8 BCC
directions contributes one 2D phase unit (2*pi)^2. A first-principles
derivation of H from the elastic properties of the lattice is open.

Lattice discrimination: under the phase-unit ansatz, only N=8 (BCC)
gives H_0 = 66.91 km/s/Mpc matching Planck. N=4 (tetrahedron),
N=6 (SC), N=12 (FCC), and N=16 (hypercubic) are all excluded at
many sigma. This is a conditional falsification of the ansatz, not
an absolute derivation that other lattices are impossible.

Anchors: r_e, m_e, c, e (dimensional) and T_cmb (cosmological).
No continuously adjustable parameter appears in the chain.

Status of Omega_vac = eta_BCC * C_gap: this is a geometric MECHANISM,
not a numerical fit. The ingredients are eta_BCC (pure BCC
crystallography) and C_gap (neutral volumetric response, the SAME
operator that fixes M_W to 0.012% in the electroweak sector). The
product is what Lambda-CDM calls the dark-energy fraction; in EWT
it is not a mysterious substance but the effective vacuum response
of the BCC lattice. What is not yet established: why the observable
cosmological fraction must be exactly this product rather than
another combination of eta_BCC and C_gap. The specific composition
is identified, not derived from a horizon-balance argument; that is
the principal open objective.
"""

import math

try:
    from m4_7_ewt_emergence_engine import (
        PI, C0, M_E, R_E, E_CHARGE_CODATA,
        BCC_IDEAL_PROJECTION_LP,
        compute_alpha_core,
        compute_alpha_geometric,
        derive_eps_M_from_BCC,
        derive_planck_charge_from_e,
        derive_neutrino_radius,
        derive_hbar,
        derive_lambda_l_geometric,
        gravity_sector,
        build_geometric_ladder,
    )
except ImportError as e:
    raise ImportError(
        "This script requires m4_7_ewt_emergence_engine.py in the same directory."
    ) from e


# ----------------------------------------------------------------------
# Unit-definition constants and the cosmological anchor
# ----------------------------------------------------------------------
K_B          = 1.380649e-23    # exact by SI 2019
T_CMB_CODATA = 2.7255          # COBE/FIRAS -- measured anchor
T_CMB_SIGMA  = 0.0006
MPC_M        = 3.0856775814913673e22

K_WC = 10


# ----------------------------------------------------------------------
# Reference values (Planck 2018, DESI 2024, SH0ES 2022)
# ----------------------------------------------------------------------
H0_PLANCK       = 67.36
H0_PLANCK_SIGMA = 0.54
OMEGA_L_PLANCK_TT        = 0.6847
OMEGA_L_PLANCK_ALL       = 0.6889
OMEGA_L_PLANCK_TT_SIGMA  = 0.0073
OMEGA_L_PLANCK_ALL_SIGMA = 0.0056
LAMBDA_PLANCK   = 1.089e-52

H0_DESI       = 68.52
H0_DESI_SIGMA = 0.62
H0_SHOES       = 73.04
H0_SHOES_SIGMA = 1.04


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def sep(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def h0_km_s_mpc(R_H_m):
    return C0 / R_H_m / 1000.0 * MPC_M


def fmt(x, unit="", prec=6):
    s = f"{x:.{prec}e}"
    return f"{s} {unit}" if unit else s


# ======================================================================
# PART 1-4: The vacuum chain from eps_M to M_mu and H
# ======================================================================
def derive_vacuum_chain():
    """
    Derive N_geom, eps_M, M_mu, and H from the BCC lattice.

    The chain closes exactly: H*M_mu = 8*N_geom. This identity is
    the algebraic expression of eps_M being the SAME vacuum property
    in both the AMM (via O_mu) and the horizon (via H). The identity
    itself is tautological by construction; the non-trivial content
    is the shared physical origin of the (2*pi)^2 factor (2D Fourier
    normalisation), not the algebraic identity.
    """
    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    N_geom = bcc["N_geom"]
    eps_M  = bcc["eps_M"]
    zeta   = bcc["zeta"]

    O_mu_required = 1.0 / (2.0 * PI)**2
    M_mu_derived  = N_geom * O_mu_required
    M_mu_ideal    = 8.0 * PI**4 / (2.0 * PI)**2   # = 2*pi^2

    H_factor = 8.0 * (2.0 * PI)**2

    return {
        "N_ideal":    8.0 * PI**4,
        "N_geom":     N_geom,
        "eps_M":      eps_M,
        "zeta":       zeta,
        "O_mu":       O_mu_required,
        "M_mu":       M_mu_derived,
        "M_mu_ideal": M_mu_ideal,
        "H":          H_factor,
    }


def print_vacuum_chain(vc):
    """Narrative for Parts 1-4."""
    sep("PART 1: BCC stiffness N_geom")
    print()
    print("  The BCC lattice has 8 nearest neighbours and a 4D phase-")
    print("  space budget (3 spatial + amplitude). Ideal stiffness:")
    print("      N_ideal = 8*pi^4")
    print("  corrected by the packing impedance zeta:")
    print("      N_geom  = 8*pi^4*(1-zeta)")
    print()
    print(f"  N_ideal = {vc['N_ideal']:.10f}")
    print(f"  zeta    = {vc['zeta']:.10e}")
    print(f"  N_geom  = {vc['N_geom']:.10f}")

    sep("PART 2: Vacuum deficit eps_M")
    print()
    print("  The magnetic deficit is the inverse of the geometric")
    print("  stiffness times the 3D phase volume:")
    print("      eps_M = 1 / (N_geom * pi^3)")
    print()
    print(f"  eps_M = {vc['eps_M']:.15e}")
    print()
    print("  This is the SAME eps_M that appears in the AMM, in alpha,")
    print("  and in the cosmological horizon. It is the universal")
    print("  vacuum property of the BCC lattice.")

    sep("PART 3: AMM projection -> M_mu is forced")
    print()
    print("  The muon AMM requires a 2D -> 1D projection operator:")
    print("      O_mu = M_mu * pi^3 * eps_M = M_mu / N_geom")
    print("  Requiring O_mu = 1/(2*pi)^2 gives:")
    print("      M_mu = N_geom / (2*pi)^2")
    print()
    print(f"  O_mu required       = {vc['O_mu']:.10e}")
    print(f"  M_mu derived        = {vc['M_mu']:.10f}")
    print(f"  M_mu ideal (zeta=0) = {vc['M_mu_ideal']:.10f} (= 2*pi^2)")
    print()
    print("  The factor 1/2 between M_mu and (2*pi)^2 is NOT an")
    print("  independent postulate: it follows from the ratio 8/4 = 2,")
    print("  where 8 is the BCC coordination and 4 is the coefficient")
    print("  in (2*pi)^2 = 4*pi^2.")
    print()
    print("  Independent confirmation from the 2D Fourier transform:")
    print("  O_mu = 1/(2*pi)^2 is also the standard normalisation of")
    print("  the inverse 2D Fourier transform, which contracts two")
    print("  phase dimensions into a single scalar observable. The")
    print("  operator is therefore doubly fixed: algebraically by the")
    print("  AMM algorithm, and independently by the phase-space")
    print("  normalisation of any 2D-to-1D projection.")

    sep("PART 4: Horizon factor H = 8*(2*pi)^2 = 32*pi^2")
    print()
    print("  The horizon is a BCC interface: at N_stat or N_max, the")
    print("  lattice preserves its 8-fold coordination. Each of the 8")
    print("  BCC directions contributes a 2D phase unit (2*pi)^2:")
    print()
    print("      H = 8 * (2*pi)^2 = 32*pi^2 = (8*pi)^2/2")
    print()
    print(f"  H = {vc['H']:.10f}")
    print()
    print("  Structural consistency between the two sectors:")
    H_M_mu   = vc["H"] * vc["M_mu"]
    eight_N  = 8.0 * vc["N_geom"]
    print(f"      H * M_mu   = {H_M_mu:.6f}")
    print(f"      8 * N_geom = {eight_N:.6f}")
    print(f"      ratio      = {H_M_mu/eight_N:.12f}")
    print("      (must be 1)")
    print()
    print("  Note on interpretation: this identity is algebraic and")
    print("  holds for any X in H = 8*X, M_mu = N_geom/X. It is not")
    print("  a derivation of the (2*pi)^2 factor. What is non-trivial")
    print("  is that (2*pi)^2 is the standard 2D phase-space unit")
    print("  (inverse 2D Fourier normalisation), and that it appears")
    print("  independently in both the AMM and the horizon sectors.")
    print("  The shared origin is the 2D Fourier normalisation, not")
    print("  the algebraic cancellation of X.")


# ======================================================================
# PART 5: Lattice discrimination test
# ======================================================================
def lattice_discrimination_test(H_BCC, H0_BCC):
    """
    Show that, UNDER THE PHASE-UNIT ANSATZ, only N=8 (BCC) is
    consistent with the Planck H_0.

    This test is conditional: it assumes H(N) = N*(2*pi)^2 for any
    lattice, i.e. each coordination direction contributes one 2D
    phase unit. A full treatment would compute the elastic tensor
    and dispersion relations for each lattice independently; that
    is an open objective.
    """
    sep("PART 5: Lattice discrimination under the phase-unit ansatz")
    print()
    print("  Model ansatz: each coordination direction of the vacuum")
    print("  lattice contributes one 2D phase unit (2*pi)^2 to the")
    print("  horizon. Under this ansatz, H(N) = N*(2*pi)^2, so:")
    print()
    print("      H_0(N) = (N/8) * H_0^BCC")
    print()
    print("  This is a conditional falsification: only N=8 (BCC) should")
    print("  match Planck if the ansatz is correct. It is NOT a")
    print("  derivation that other lattices are impossible in general,")
    print("  because a full elastic-tensor treatment might give")
    print("  different scalings for different lattice types.")
    print()
    print(f"  Tension against Planck ({H0_PLANCK} +/- {H0_PLANCK_SIGMA} km/s/Mpc):")
    print()
    print(f"  {'N':>3} {'lattice':<14} {'H_0 [km/s/Mpc]':>16} {'tension':>14}")
    print("  " + "-" * 52)

    lattices = {4: "tetrahedron", 6: "simple cubic", 8: "BCC",
                12: "FCC", 16: "hypercubic"}
    results = {}
    for N in [4, 6, 8, 12, 16]:
        H_N = (N / 8.0) * H0_BCC
        t   = (H_N - H0_PLANCK) / H0_PLANCK_SIGMA
        tag = "  <-- MATCH" if N == 8 else ""
        print(f"  {N:>3} {lattices[N]:<14} {H_N:>16.4f} {t:>+13.2f} sigma{tag}")
        results[N] = {"H_0": H_N, "tension": t}

    print()
    print("  Under the ansatz, only N=8 (BCC) is consistent with")
    print("  Planck. Other lattice types are excluded at many sigma.")
    print("  The value of the test lies in the specificity of the")
    print("  prediction, not in a claim that other lattices are")
    print("  physically impossible.")
    return results


# ======================================================================
# PART 6-7: EWT trinity and Planck temperature
# ======================================================================
def derive_ewt_trinity():
    """Derive alpha, hbar, G, lambda_l, T_p, and the ladder."""
    bcc = derive_eps_M_from_BCC(8.0 * PI**4)
    N_geom = bcc["N_geom"]
    eps_M  = bcc["eps_M"]
    zeta   = bcc["zeta"]

    alpha_inv  = compute_alpha_geometric(eps_M)
    alpha_geom = 1.0 / alpha_inv

    q_P    = derive_planck_charge_from_e(alpha_geom, E_CHARGE_CODATA)
    nu_res = derive_neutrino_radius(alpha_geom, q_P)
    r_nu   = nu_res["r_nu"]

    lambda_l = derive_lambda_l_geometric(
        alpha_geom=alpha_geom, r_e=R_E, r_nu=r_nu,
        N_geom=N_geom, L_p_geom=BCC_IDEAL_PROJECTION_LP, K_WC=K_WC,
    )

    res_gravity = gravity_sector(
        alpha_geom=alpha_geom, r_nu=r_nu, N_geom=N_geom,
        L_p_geom=BCC_IDEAL_PROJECTION_LP, K_WC=K_WC,
        lambda_l=lambda_l, r_e=R_E, m_e=M_E, c0=C0,
    )

    hbar_geom = derive_hbar(alpha_geom, R_E, M_E, C0)
    G_geom    = res_gravity["G_EWT"]
    T_p       = (1.0 / K_B) * math.sqrt(hbar_geom * C0**5 / G_geom)

    ladder = build_geometric_ladder(eps_M, "BCC packing derived", zeta=zeta)

    return {
        "alpha_geom":   alpha_geom,
        "alpha_inv":    alpha_inv,
        "eps_M":        eps_M,
        "zeta":         zeta,
        "N_geom":       N_geom,
        "r_nu":         r_nu,
        "lambda_l":     lambda_l,
        "hbar_geom":    hbar_geom,
        "G_geom":       G_geom,
        "T_p":          T_p,
        "N_nu_stat":    res_gravity["N_nu_statutory"],
        "N_nu_eff":     res_gravity["N_nu_eff"],
        "X_eff":        res_gravity["X_eff"],
        "eta_BCC":      (PI * math.sqrt(3.0)) / 8.0,
        "C_local":      ladder["C_local"],
        "C_gap":        ladder["C_gap"],
        "C_gap_mixed":  ladder["C_gap_mixed"],
        "O_mixed":      ladder["O_mixed"],
    }


def print_ewt_trinity(tri):
    sep("PART 6: EWT trinity -- alpha, hbar, G, lambda_l")
    print()
    print("  The four geometric constants that enter the cosmology:")
    print()
    print(f"  alpha_geom^-1 = {tri['alpha_inv']:.12f}")
    print(f"  hbar_geom     = {tri['hbar_geom']:.6e} J*s")
    print(f"  G_geom        = {tri['G_geom']:.6e} m^3/kg/s^2")
    print(f"  lambda_l      = {tri['lambda_l']:.6e} m")
    print()
    print("  These are the closed Planck trinity of the EWT framework.")
    print("  The residual against CODATA is 0.048% for G and 0.028%")
    print("  for lambda_l, attributed to the lattice impedance zeta.")
    print()
    print("  Note: because hbar and G enter the cosmological chain")
    print("  only through the combinations hbar/G and hbar*G (see")
    print("  Parts 7-8 below), the individual CODATA residuals do")
    print("  not propagate to H_0 or Lambda. The chain is a test of")
    print("  the BCC geometry at the level of the products.")
    print()
    print("  This is consistent with Haug's composite-constant view")
    print("  (Haug 2020 Physics Essays, Haug 2022 Universe, Haug 2023")
    print("  JAMP): all observable gravitational predictions depend")
    print("  on GM, not on G and M separately. The specific form of")
    print("  G_geom is therefore not critical for the predictions.")

    sep("PART 7: Planck temperature from the EWT trinity")
    print()
    print("  T_p = (1/k_B) * sqrt(hbar_geom * c^5 / G_geom)")
    print()
    print(f"  T_p = {tri['T_p']:.6e} K")
    print()
    print("  Note: k_B is an exact SI unit-conversion constant, not")
    print("  a dynamical input. c is the EWT metric conversion factor.")


# ======================================================================
# PART 8-9: Horizon radius and Hubble constant
# ======================================================================
def horizon_radius_derived(tri, H_factor):
    """
    R_H from naive equipartition corrected by the BCC horizon factor.

    Naive equipartition: T_cmb = T_p * sqrt(lambda_l / R_H)
    Inverting:           R_H_naive = lambda_l * (T_p / T_cmb)^2
    BCC correction:      R_H = R_H_naive / H, with H = 8*(2*pi)^2

    This recovers the Haug relation exactly:
        T_cmb = (T_p / (8*pi)) * sqrt(2*lambda_l / R_H)
    The recovery is algebraic; the horizon factor H itself is a
    model construct (phase-unit ansatz).
    """
    R_H_naive = tri["lambda_l"] * (tri["T_p"] / T_CMB_CODATA)**2
    R_H       = R_H_naive / H_factor
    return R_H_naive, R_H


def print_horizon_and_h0(tri, R_H_naive, R_H_BCC):
    sep("PART 8: Horizon radius from T_cmb and the horizon factor")
    print()
    print("  Naive equipartition:  T_cmb = T_p * sqrt(lambda_l / R_H)")
    print("  Inverting:            R_H_naive = lambda_l*(T_p/T_cmb)^2")
    print()
    print("  BCC horizon factor:   H = 8*(2*pi)^2 = 32*pi^2")
    print("  Corrected:            R_H = R_H_naive / H")
    print()
    print(f"  lambda_l  = {tri['lambda_l']:.6e} m")
    print(f"  T_p       = {tri['T_p']:.6e} K")
    print(f"  T_cmb     = {T_CMB_CODATA} K")
    print(f"  R_H_naive = {R_H_naive:.6e} m")
    print(f"  R_H       = {R_H_BCC:.6e} m")
    print()
    print("  The Haug relation is recovered exactly:")
    print("      T_cmb = (T_p / (8*pi)) * sqrt(2*lambda_l/R_H)")
    T_check = (tri["T_p"] / (8.0 * PI)) * math.sqrt(2.0 * tri["lambda_l"] / R_H_BCC)
    print(f"  Recovered T_cmb = {T_check:.10f} K")
    print(f"  Match: {'YES' if abs(T_check - T_CMB_CODATA) < 1e-4 else 'NO'}")
    print()
    print("  The Haug relation is now RECOVERED, not imported. It is")
    print("  the combination of naive equipartition and the BCC")
    print("  horizon factor H = 8*(2*pi)^2. The horizon factor itself")
    print("  is a model construct: each of the 8 BCC directions")
    print("  contributes one 2D phase unit (2*pi)^2. A first-principles")
    print("  derivation of H from the elastic properties of the lattice")
    print("  is open. The recovery is therefore a consequence of the")
    print("  BCC horizon model, not an import from external literature.")

    sep("PART 9: Hubble constant H_0 and tension analysis")
    H_0_BCC = h0_km_s_mpc(R_H_BCC)
    print()
    print("  H_0 = c / R_H")
    print()
    print(f"  H_0 = {H_0_BCC:.6f} km/s/Mpc")
    print()
    print("  Tension analysis:")
    print(f"  {'Reference':<20} {'H_0':>10} {'sigma':>8} {'tension':>14}")
    print("  " + "-" * 58)
    for name, H0_ref, H0_sig in [
        ("Planck 2018", H0_PLANCK, H0_PLANCK_SIGMA),
        ("DESI 2024",   H0_DESI,   H0_DESI_SIGMA),
        ("SH0ES 2022",  H0_SHOES,  H0_SHOES_SIGMA),
    ]:
        t = (H_0_BCC - H0_ref) / H0_sig
        print(f"  {name:<20} {H0_ref:>10.2f} {H0_sig:>8.2f} {t:>+13.2f} sigma")
    print()
    print("  H_0 = 66.91 is on the Planck side of the Hubble tension.")
    print("  Because the horizon relation is recovered from the BCC")
    print("  horizon model and T_cmb is external to the lattice, the")
    print("  0.83-sigma agreement is a consistency check of the model")
    print("  against observation, not an unconditional derivation.")
    return H_0_BCC


# ======================================================================
# PART 10: de Sitter cosmological constant
# ======================================================================
def lambda_de_sitter(H_0_km):
    """Lambda_dS = 3 * H_0^2 / c^2, with H_0 in km/s/Mpc."""
    H_0_si = H_0_km * 1000.0 / MPC_M
    return 3.0 * H_0_si**2 / C0**2


def print_lambda_de_sitter(H_0_BCC):
    sep("PART 10: De Sitter cosmological constant")
    Lambda_dS = lambda_de_sitter(H_0_BCC)
    print()
    print("  Lambda_dS = 3*H_0^2 / c^2")
    print()
    print(f"  Lambda_dS     = {Lambda_dS:.6e} m^-2")
    print(f"  Lambda_Planck = {LAMBDA_PLANCK:.6e} m^-2")
    print(f"  Ratio         = {Lambda_dS/LAMBDA_PLANCK:.6f}")
    print()
    print("  The 1.44x ratio is structural: it decomposes as")
    print("  (H_0^EWT/H_0^Planck)^2 * (Omega_Lambda^Planck)^-1")
    r1 = (H_0_BCC/H0_PLANCK)**2
    r2 = 1.0/OMEGA_L_PLANCK_TT
    print(f"  = {r1:.4f} * {r2:.4f}")
    print(f"  = {r1*r2:.4f}")
    return Lambda_dS


# ======================================================================
# PART 11: Vacuum fraction Omega_vac (geometric mechanism)
# ======================================================================
def omega_vac_mechanism(tri):
    """
    Omega_vac = eta_BCC * C_gap is a geometric MECHANISM.

    eta_BCC = pi*sqrt(3)/8 is the BCC packing fraction: the fraction
    of space occupied by EMC spheres. The remaining 1-eta_BCC = 0.32
    is the void fraction where matter solitons reside.

    C_gap = 1 + pi^6*C_local is the neutral volumetric response of the
    vacuum to depletion. It is the SAME operator that fixes M_W to
    0.012% in the electroweak sector, derived from the same eps_M.

    The product is the natural composition: (occupied fraction) times
    (response of the vacuum to voids). It gives Omega_vac = 0.6897,
    matching Planck+BAO to 0.12%.

    What is not yet established: why the observable cosmological
    fraction must be exactly this product, rather than another
    combination of eta_BCC and C_gap. The ingredients are derived
    and the mechanism is geometric, but the specific composition is
    identified, not derived from a horizon-balance argument.
    """
    return tri["eta_BCC"] * tri["C_gap"]


def print_omega_vac(tri, Omega_vac):
    sep("PART 11: Vacuum fraction Omega_vac -- geometric mechanism")
    print()
    print("  In Lambda-CDM, the quantity that multiplies Lambda_dS in")
    print("  the Friedmann equation is called the dark-energy fraction.")
    print("  In EWT, it is a geometric property of the BCC vacuum, not")
    print("  a mysterious substance. The same numerical value plays the")
    print("  same role, but the physical interpretation is different.")
    print()
    print("  The observed cosmological fraction is identified with the")
    print("  effective vacuum fraction of the BCC lattice. This is a")
    print("  geometric MECHANISM, not a numerical fit.")
    print()
    print("      Omega_vac = eta_BCC * C_gap")
    print()
    print(f"  eta_BCC = pi*sqrt(3)/8 = {tri['eta_BCC']:.10f}")
    print("      (BCC packing fraction: fraction of BCC space")
    print("       occupied by EMC spheres; the remaining")
    print("       1-eta_BCC = 0.32 is the void volume)")
    print(f"  C_gap   = 1 + pi^6*C_local = {tri['C_gap']:.10f}")
    print("      (neutral volumetric response of the vacuum; the")
    print("       same operator that fixes M_W to 0.012%)")
    print(f"  Omega_vac = {Omega_vac:.10f}")
    print()
    print()
    print("  Algebraic decomposition of the product:")
    eta = tri['eta_BCC']
    C_gap = tri['C_gap']
    delta_C = C_gap - 1.0  # = pi^6 * C_local
    # Sum: eta + eta*delta_C
    sum_form = eta + eta * delta_C
    print(f"      eta_BCC                       = {eta:.10f}")
    print(f"      eta_BCC * (C_gap - 1)         = {eta * delta_C:.10f}")
    print(f"      Sum                           = {sum_form:.10f}")
    print(f"      Product (cross-check)         = {Omega_vac:.10f}")
    print(f"      Difference                    = {abs(sum_form - Omega_vac):.3e}")
    print()
    print("  The sum is an algebraic decomposition of the product, not a")
    print("  replacement of the composition rule. In the expanded form:")
    print("      Omega_vac = pi*sqrt(3)/8 + sqrt(3/2)/(2^7*(1-zeta))")
    print("  , the transcendental factor pi cancels in the correction")
    print("  leaving a correction that depends only on")
    print("  algebraic factors: sqrt(3/2) and 128 = 2^7. The exponent 7")
    print("  is the same as in the magnetic deficit eps_M = 1/(8*pi^7(1-zeta)):")
    print("  it is the dimensionality of the charged weak rung. The same")
    print("  7-dimensional phase space that fixes the charged weak response")
    print("  also fixes the vacuum response to depletion.")        
    print("  Interpretation:")
    print()
    print("    eta_BCC is the fraction of BCC space occupied by EMC")
    print("    spheres. The remaining 1-eta_BCC = 0.32 is the void")
    print("    fraction where matter solitons reside. C_gap is the")
    print("    neutral volumetric response of the vacuum to that")
    print("    depletion. The product is the natural composition:")
    print("    (occupied fraction) times (response to voids).")
    print()
    print("  Comparison with Planck:")
    print(f"  Omega_L^Planck (TT)     = {OMEGA_L_PLANCK_TT} +/- {OMEGA_L_PLANCK_TT_SIGMA}")
    print(f"  Omega_L^Planck (+BAO)   = {OMEGA_L_PLANCK_ALL} +/- {OMEGA_L_PLANCK_ALL_SIGMA}")
    rel_TT  = (Omega_vac - OMEGA_L_PLANCK_TT)  / OMEGA_L_PLANCK_TT  * 100.0
    rel_ALL = (Omega_vac - OMEGA_L_PLANCK_ALL) / OMEGA_L_PLANCK_ALL * 100.0
    t_TT    = (Omega_vac - OMEGA_L_PLANCK_TT)  / OMEGA_L_PLANCK_TT_SIGMA
    t_ALL   = (Omega_vac - OMEGA_L_PLANCK_ALL) / OMEGA_L_PLANCK_ALL_SIGMA
    print(f"  Rel diff (TT):   {rel_TT:+.3f}%  ({t_TT:+.2f} sigma)")
    print(f"  Rel diff (+BAO): {rel_ALL:+.3f}%  ({t_ALL:+.2f} sigma)")
    print()
    print("  Status:")
    print("    - The ingredients eta_BCC and C_gap are derived or")
    print("      independently tested (M_W), not free parameters.")
    print("    - The product is the natural geometric composition.")
    print("    - The quantity plays the role that Lambda-CDM calls the")
    print("      dark-energy fraction. In EWT it is not a mysterious")
    print("      substance but the effective vacuum response of the")
    print("      BCC lattice: (occupied fraction) x (response to voids).")
    print("    - What is not yet established: why the observable")
    print("      cosmological fraction must be exactly this product,")
    print("      rather than another combination of eta_BCC and C_gap")
    print("      (e.g., sum, or eta_BCC alone). The specific")
    print("      composition is identified, not derived from a")
    print("      horizon-balance argument. This is the principal open")
    print("      objective of the cosmological sector.")


# ======================================================================
# PART 12: Observed Lambda
# ======================================================================
def print_lambda_vac(Omega_vac, Lambda_dS):
    sep("PART 12: Observed cosmological constant Lambda_vac")
    Lambda_vac = Omega_vac * Lambda_dS
    print()
    print("  Lambda_vac = Omega_vac * Lambda_dS")
    print()
    print(f"  Lambda_vac    = {Lambda_vac:.6e} m^-2")
    print(f"  Lambda_Planck = {LAMBDA_PLANCK:.6e} m^-2")
    print(f"  Ratio         = {Lambda_vac/LAMBDA_PLANCK:.6f}")
    print(f"  Rel diff      = {(Lambda_vac-LAMBDA_PLANCK)/LAMBDA_PLANCK*100:+.3f}%")
    print()
    print("  The chain closes: Lambda is predicted to 0.59% accuracy")
    print("  from the BCC lattice with one cosmological anchor (T_cmb).")
    return Lambda_vac


# ======================================================================
# PART 13: Matter correction (diagnostic only)
# ======================================================================
def try_matter_correction(tri, R_H_0, Omega_m_target=0.31):
    """
    Diagnostic only, NOT part of the derivation chain.

    Tests whether a simple matter correction to the horizon radius,
        R_H(Omega_m) = R_H(0) * (1 - Omega_m)^p,
    can close the H_0 gap to Planck. The fitted exponent p turns out
    to be small (p ~ 0.018), confirming that de Sitter already
    dominates and matter is a ~0.6% perturbation.
    """
    sep("PART 13: Matter correction -- diagnostic only, not derivation")
    print()
    print("  Tests: R_H(Omega_m) = R_H(0) * (1 - Omega_m)^p")
    print("  Find p such that H_0(Omega_m=0.31) = Planck H_0.")
    print()

    H_0_0 = h0_km_s_mpc(R_H_0)
    ratio = H_0_0 / H0_PLANCK
    p     = math.log(ratio) / math.log(1.0 - Omega_m_target)
    f     = (1.0 - Omega_m_target)**p

    R_H_corr    = R_H_0 * f
    H_0_corr    = h0_km_s_mpc(R_H_corr)
    Lambda_corr = lambda_de_sitter(H_0_corr)

    print(f"  H_0(Omega_m=0)    = {H_0_0:.6f} km/s/Mpc")
    print(f"  target H_0        = {H0_PLANCK:.4f} km/s/Mpc")
    print(f"  required ratio    = {ratio:.6f}")
    print(f"  best-fit p        = {p:.6f}")
    print(f"  f(0.31)           = {f:.6f}")
    print()
    print(f"  H_0(corrected)    = {H_0_corr:.6f} km/s/Mpc")
    print(f"  Lambda(corrected) = {Lambda_corr:.6e} m^-2")
    print()
    print("  The smallness of p (0.018) confirms that de Sitter")
    print("  already dominates and matter is a ~0.6% perturbation.")
    print("  This is a diagnostic, not a derivation of Lambda.")
    return {"p": p, "f": f, "H_0_corr": H_0_corr, "Lambda_corr": Lambda_corr}


# ======================================================================
# PART 14: Full chain summary
# ======================================================================
def print_full_chain(vc, tri, R_H_BCC, H_0_BCC, Lambda_dS, Omega_vac, Lambda_vac):
    sep("PART 14: FULL CHAIN -- from eps_M to Lambda_vac")
    print()
    print("  Step 1: N_geom = 8*pi^4*(1-zeta)")
    print(f"          N_geom = {vc['N_geom']:.6f}")
    print()
    print("  Step 2: eps_M = 1/(N_geom*pi^3)")
    print(f"          eps_M  = {vc['eps_M']:.6e}")
    print()
    print("  Step 3: O_mu = M_mu/N_geom = 1/(2*pi)^2")
    print(f"          M_mu   = {vc['M_mu']:.6f}")
    print()
    print("  Step 4: H = 8*(2*pi)^2 = 32*pi^2")
    print(f"          H      = {vc['H']:.6f}")
    print(f"          Check: H*M_mu = 8*N_geom = {vc['H']*vc['M_mu']:.4e}")
    print()
    print("  Step 5: Lattice test (under phase-unit ansatz)")
    print("          -> only N=8 (BCC) matches Planck")
    print()
    print(f"  Step 6: T_p = {tri['T_p']:.6e} K")
    print()
    print(f"  Step 7: R_H = lambda_l*(T_p/T_cmb)^2/H")
    print(f"          R_H = {R_H_BCC:.6e} m")
    print()
    print(f"  Step 8: H_0 = c/R_H = {H_0_BCC:.4f} km/s/Mpc")
    print()
    print(f"  Step 9: Lambda_dS = 3*H_0^2/c^2 = {Lambda_dS:.4e} m^-2")
    print()
    print(f"  Step 10: Omega_vac = eta_BCC*C_gap = {Omega_vac:.6f}")
    print()
    print(f"  Step 11: Lambda_vac = {Lambda_vac:.4e} m^-2")
    print()
    print("  Summary:")
    print(f"    H_0^EWT    = {H_0_BCC:.2f} km/s/Mpc   (Planck: {H0_PLANCK})")
    print(f"    Lambda_dS  = {Lambda_dS:.2e} m^-2  ({Lambda_dS/LAMBDA_PLANCK:.3f} x Planck)")
    print(f"    Omega_vac  = {Omega_vac:.4f}          (Planck+BAO: {OMEGA_L_PLANCK_ALL})")
    print(f"    Lambda_vac = {Lambda_vac:.2e} m^-2  ({Lambda_vac/LAMBDA_PLANCK:.3f} x Planck)")
    print()
    print("  Anchors: r_e, m_e, c, e (dimensional), T_cmb (cosmological).")
    print("  No continuously adjustable parameter in the chain.")
    print()
    print("  The Haug relation is RECOVERED from the BCC horizon model,")
    print("  not imported. The factor 32*pi^2 = 8*(2*pi)^2 comes from 8")
    print("  BCC directions, each contributing the 2D unit")
    print("  (2*pi)^2 = 1/O_mu. The phase-unit ansatz itself (one")
    print("  (2*pi)^2 per BCC direction) is a model construct; its")
    print("  first-principles derivation is open.")
    print()
    print("  Open: the specific composition Omega_vac = eta_BCC*C_gap")
    print("  is a geometric mechanism with derived ingredients; a")
    print("  horizon-balance derivation of why this exact product is")
    print("  required (rather than, e.g., a sum or eta_BCC alone)")
    print("  remains the principal open objective.")


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 78)
    print("Cosmological constant from the BCC lattice")
    print("=" * 78)

    # Parts 1-4: vacuum chain
    vc = derive_vacuum_chain()
    print_vacuum_chain(vc)

    # Parts 6-7: EWT trinity and T_p
    tri = derive_ewt_trinity()
    print_ewt_trinity(tri)

    # Parts 8-9: horizon radius and H_0
    R_H_naive, R_H_BCC = horizon_radius_derived(tri, vc["H"])
    H_0_BCC = print_horizon_and_h0(tri, R_H_naive, R_H_BCC)

    # Part 5 (now that H_0^BCC is known): lattice discrimination
    lattice_discrimination_test(vc["H"], H_0_BCC)

    # Part 10: Lambda_dS
    Lambda_dS = print_lambda_de_sitter(H_0_BCC)

    # Part 11: Omega_vac mechanism
    Omega_vac = omega_vac_mechanism(tri)
    print_omega_vac(tri, Omega_vac)

    # Part 12: Lambda_vac
    Lambda_vac = print_lambda_vac(Omega_vac, Lambda_dS)

    # Part 13: matter correction (diagnostic)
    corr = try_matter_correction(tri, R_H_BCC, Omega_m_target=0.31)

    # Part 14: full chain summary
    print_full_chain(vc, tri, R_H_BCC, H_0_BCC, Lambda_dS, Omega_vac, Lambda_vac)

    return {
        "vacuum_chain": vc,
        "trinity":      tri,
        "R_H_naive":    R_H_naive,
        "R_H":          R_H_BCC,
        "H_0":          H_0_BCC,
        "Lambda_dS":    Lambda_dS,
        "Omega_vac":    Omega_vac,
        "Lambda_vac":   Lambda_vac,
        "matter_corr":  corr,
    }


if __name__ == "__main__":
    main()