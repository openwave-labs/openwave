# From manuscript version: 5.0.0
# =============================================================================
# SECTION 1: GLOBAL INPUTS & CONSTANTS
# -----------------------------------------------------------------------------
# This section contains:
#   - mathematical constants
#   - experimental reference values (CODATA 2022 / PDG 2022)
#   - pure BCC lattice geometry (independent of the EWT model)
#
# Every quantity is explicitly labelled:
#   INPUT        : experimental value used as an anchor or metric input
#   MATHEMATICAL : pure mathematical constant
#   BCC_GEOMETRY : crystallographic fact about the BCC lattice
#
# The model itself will not define any additional hidden numerical parameters
# in this section. All later parameters must be either DERIVED, POSTULATED,
# or explicitly marked as CALIBRATION (to be eliminated in future versions).
# =============================================================================

import math

# -----------------------------------------------------------------------------
# 1.1 MATHEMATICAL CONSTANTS
# -----------------------------------------------------------------------------

PI      = math.pi                    # circle ratio
EULER   = math.e                     # Euler's number
SQRT2   = math.sqrt(2.0)             # diagonal of unit square
SQRT3   = math.sqrt(3.0)             # diagonal of unit cube face

# -----------------------------------------------------------------------------
# 1.2 CODATA 2022 / PDG 2022 EXPERIMENTAL INPUTS
# -----------------------------------------------------------------------------

# Metric and electromagnetic scales
C0                = 299792458.0               # speed of light [m/s] (exact SI)
M_E               = 9.1093837015e-31          # electron mass [kg]
R_E               = 2.8179403262e-15          # classical electron radius [m]
ALPHA_INV_CODATA  = 137.035999084              # inverse fine-structure constant
A_E_CODATA        = 0.0011596521816            # electron AMM (g-2)/2

# Gravitational scale
G_CODATA          = 6.674305e-11              # gravitational constant [m^3 kg^-1 s^-2]

# Electroweak and flavour scales
M_Z_CODATA        = 91.1876                   # Z boson mass [GeV]
M_W_PDG           = 80.377                    # W boson mass, PDG 2022 average [GeV]
M_W_CDFII         = 80.4335                   # W boson mass, CDF II 2022 [GeV]
M_H_CODATA        = 125.25                    # Higgs mass [GeV]
SIN2_THETA_W      = 0.23122                   # Weinberg angle sin^2(theta_W)
SIN_THETA_C_PDG   = 0.2243                    # Cabibbo angle sin(theta_C)
R_INF_CODATA      = 10973731.568157           # Rydberg constant [m^-1]
A0_CODATA         = 5.29177210903e-11         # Bohr radius [m]
LAMBDA_C_CODATA   = 2.42631023867e-12         # electron Compton wavelength [m]

# Quark masses (MS-bar, PDG 2022) used in flavour mixing test
M_D_PDG           = 0.004692                  # d-quark mass [GeV]
M_S_PDG           = 0.094954                  # s-quark mass [GeV]

A_MU_EXP          = 1.1659206100e-3        # muon AMM from Fermilab/Brookhaven
A_TAU_EXP         = 1.177210e-3            # tau AMM from PDG

# Fundamental length scale (Planck charge interpreted as geometric amplitude)
# This is currently treated as an INPUT, but will later be reclassified
# as DERIVED once the geometric origin of q_P is explicitly shown.
Q_P_INPUT         = 1.87554603778e-18         # Planck charge as length [m]
# Fundamental EMC spacing (Planck length) [m]
LAMBDA_L          = 1.6162e-35
# Elementary charge (CODATA 2022) -- empirical anchor of the amplitude scale
E_CHARGE_CODATA   = 1.602176634e-19

# -----------------------------------------------------------------------------
# 1.3 PURE BCC LATTICE GEOMETRY
# -----------------------------------------------------------------------------
# These are purely mathematical / crystallographic facts about the BCC lattice.
# They are independent of the Enhanced EWT model and contain no physics inputs.
# -----------------------------------------------------------------------------

BCC_NEIGHBOURS           = 8                      # nearest neighbours in BCC
BCC_PACKING_FRACTION     = (SQRT3 * PI) / 8.0     # ~0.6801748
BCC_IDEAL_PROJECTION_LP  = 2.0 / SQRT3            # inverse nearest-neighbour distance

# -----------------------------------------------------------------------------
# 1.4 STATUS LABELS
# -----------------------------------------------------------------------------
# These labels will be used in all reporting functions to make the origin
# of each quantity transparent.

STATUS = {
    "INPUT":        "Experimental input (CODATA/PDG)",
    "MATHEMATICAL": "Pure mathematical constant",
    "BCC_GEOMETRY": "Pure BCC lattice geometry",
    "POSTULATE":    "Geometric postulate of the model",
    "DERIVED":      "Derived from model postulates/constraints",
    "CALIBRATION":  "Calibrated parameter (to be eliminated)",
}

# =============================================================================
# SECTION 2: GEOMETRIC POSTULATES & ALPHA CORE EMERGENCE
# -----------------------------------------------------------------------------
# The alpha core emerges from three geometric postulates:
#   P1: charge = amplitude (x)
#   P2: emission surface = sphere + cone
#   P3: natural length ratios r = x, l = pi*x
#
# These postulates lead directly to the geometric ratio:
#   alpha_core^{-1} = S_total / x^2 = 4*pi^3 + pi^2 + pi
# =============================================================================

def compute_emission_surface(x: float) -> dict:
    """
    Compute the total emission surface and the resulting alpha core
    from the geometric postulates.

    Parameters
    ----------
    x : float
        Wave amplitude (charge). The final ratio is independent of x.

    Returns
    -------
    dict with:
        r         : base radius of the cone (equals x)
        l         : propagation distance (equals pi*x)
        S_sphere  : surface area of the spherical part
        S_cone    : total surface area of the conical part
        S_total   : total emission surface
        A_pi      : alpha core inverse = S_total / x^2
    """
    r = x
    l = PI * x

    S_sphere = 4.0 * PI * l**2
    S_cone = PI * r * l + PI * r**2

    S_total = S_sphere + S_cone
    A_pi = S_total / x**2

    return {
        "r": r,
        "l": l,
        "S_sphere": S_sphere,
        "S_cone": S_cone,
        "S_total": S_total,
        "A_pi": A_pi,
    }

def compute_alpha_core() -> float:
    """
    Compute the purely geometric core of the fine-structure constant.
    It is obtained from the emission surface ratio for a unit amplitude.

    Returns
    -------
    float
        A_pi = 4*pi^3 + pi^2 + pi
    """
    return compute_emission_surface(1.0)["A_pi"]

# =============================================================================
# SECTION 2B: GEOMETRIC LADDER EMERGENCE
# -----------------------------------------------------------------------------
# The geometric ladder is not an input. It is derived from the active
# degrees of freedom of a soliton embedded in the BCC vacuum lattice.
#
# The central modulator eps_M is treated as an INPUT to this module.
# In the current zero-calibration version it is obtained from pure BCC geometry:
#   - eps_M derived from packing fraction eta and lattice impedance zeta
#   - historical calibrated variant (N_final) is no longer used
#
# Later sections will compute eps_M and pass it here.
# =============================================================================

def build_geometric_ladder(eps_M: float, source_label: str = "") -> dict:
    """
    Build the geometric ladder from the magnetic deficit eps_M.

    Parameters
    ----------
    eps_M : float
        Magnetic deficit parameter.
    source_label : str
        Optional label describing the source of eps_M.

    Returns
    -------
    dict with:
        eps_M     : magnetic deficit (input)
        source    : source label
        C_local   : local coupling
        C_gap     : volumetric operator (pi^6)
        C_fermion : surface operator (pi^5)
        ladder    : dictionary {3: pi^3, ..., 7: pi^7}
    """
    C_local = eps_M / (2.0 * SQRT2)

    return {
        "eps_M": eps_M,
        "source": source_label,
        "C_local": C_local,
        "C_gap": 1.0 + PI**6 * C_local,
        "C_fermion": (1.0 + PI**5 * C_local) ** 2,
        "ladder": {n: PI**n for n in range(3, 8)},
    }


def report_ladder(lad: dict) -> None:
    """Print the contents of a ladder dictionary in a readable form."""
    print(f"  Source     : {lad.get('source', 'unknown')}")
    print(f"  eps_M      : {lad['eps_M']:.15e}")
    print(f"  C_local    : {lad['C_local']:.15e}")
    print(f"  C_gap      : {lad['C_gap']:.15f}")
    print(f"  C_fermion  : {lad['C_fermion']:.15f}")
    print("  Ladder:")
    for n in sorted(lad["ladder"]):
        print(f"    pi^{n} = {lad['ladder'][n]:.15f}")

# =============================================================================
# SECTION 3: BCC-DERIVED LATTICE IMPEDANCE
# -----------------------------------------------------------------------------
# Functions that derive eps_M from pure BCC geometry using the packing
# fraction and lattice impedance.
# =============================================================================

def lattice_impedance(N: float, eta: float = BCC_PACKING_FRACTION) -> float:
    """
    Compute the lattice impedance zeta for a given geometric stiffness N.

    Formula:
        zeta = (1 - eta) / (eta * N)

    Parameters
    ----------
    N : float
        Geometric stiffness modulus (e.g. 8*pi^4, 8*pi^7).
    eta : float
        Packing fraction of the BCC lattice.

    Returns
    -------
    float
        Lattice impedance.
    """
    return (1.0 - eta) / (eta * N)


def derive_eps_M_from_BCC(N_input: float = 8.0 * PI**4) -> dict:
    """
    Derive eps_M from BCC geometry using the lattice impedance.

    Parameters
    ----------
    N_input : float
        Ideal geometric stiffness to use as base.
        Default is 8*pi^4 (BCC saturation stiffness).

    Returns
    -------
    dict with:
        eta      : BCC packing fraction
        N_ideal  : input ideal stiffness
        zeta     : lattice impedance
        N_geom   : effective stiffness after impedance correction
        eps_M    : magnetic deficit derived from BCC geometry
    """
    eta = BCC_PACKING_FRACTION
    zeta = lattice_impedance(N_input, eta)
    N_geom = N_input * (1.0 - zeta)
    eps_M = 1.0 / (N_geom * PI**3)

    return {
        "eta": eta,
        "N_ideal": N_input,
        "zeta": zeta,
        "N_geom": N_geom,
        "eps_M": eps_M,
    }

# =============================================================================
# SECTION 4: TEST FUNCTIONS (EXPERIMENTAL VALIDATION)
# -----------------------------------------------------------------------------
# These functions compare model predictions with experimental data.
# They do NOT influence the model itself.
# =============================================================================

def weinberg_sector(C_gap: float, M_Z: float, sin2_W: float) -> dict:
    """
    Compute the W boson mass from the Weinberg angle using the pi^6 operator.

    Formula:
        M_W = M_Z * sqrt( (1 - sin^2(theta_W)) * C_gap )

    Parameters
    ----------
    C_gap : float
        Volumetric lattice operator from the pi^6 rung.
    M_Z : float
        Experimental Z boson mass [GeV].
    sin2_W : float
        Experimental Weinberg angle sin^2(theta_W).

    Returns
    -------
    dict with:
        M_W_pred       : predicted W mass [GeV]
        rel_err_PDG    : relative error to PDG average
        rel_err_CDFII  : relative error to CDF II measurement
    """
    M_W_pred = M_Z * math.sqrt((1.0 - sin2_W) * C_gap)

    rel_err_PDG = abs(M_W_pred - M_W_PDG) / M_W_PDG * 100.0
    rel_err_CDFII = abs(M_W_pred - M_W_CDFII) / M_W_CDFII * 100.0

    return {
        "M_W_pred": M_W_pred,
        "rel_err_PDG": rel_err_PDG,
        "rel_err_CDFII": rel_err_CDFII,
    }


def cabibbo_sector(C_fermion: float, M_d: float, M_s: float) -> dict:
    """
    Compute the Cabibbo angle from quark masses using the pi^5 operator.

    Formula:
        sin(theta_C) = sqrt( M_d / M_s ) * C_fermion

    Parameters
    ----------
    C_fermion : float
        Surface lattice operator from the pi^5 rung.
    M_d : float
        Down quark mass [GeV].
    M_s : float
        Strange quark mass [GeV].

    Returns
    -------
    dict with:
        sin_C_pred : predicted sin(theta_C)
        rel_err    : relative error to PDG target
    """
    sin_C_pred = math.sqrt(M_d / M_s) * C_fermion
    rel_err = abs(sin_C_pred - SIN_THETA_C_PDG) / SIN_THETA_C_PDG * 100.0

    return {"sin_C_pred": sin_C_pred, "rel_err": rel_err}


def run_sector_tests(lad: dict) -> None:
    """
    Run experimental validation tests on a given ladder dictionary.

    Parameters
    ----------
    lad : dict
        Ladder dictionary produced by build_geometric_ladder().
    """
    result_W = weinberg_sector(lad["C_gap"], M_Z_CODATA, SIN2_THETA_W)
    print(f"  M_W prediction      : {result_W['M_W_pred']:.6f} GeV")
    print(f"  rel. err PDG        : {result_W['rel_err_PDG']:.4f} %")
    print(f"  rel. err CDF II     : {result_W['rel_err_CDFII']:.4f} %")

    result_C = cabibbo_sector(lad["C_fermion"], M_D_PDG, M_S_PDG)
    print(f"  sin theta_C         : {result_C['sin_C_pred']:.10f}")
    print(f"  rel. err PDG        : {result_C['rel_err']:.6f} %")


# =============================================================================
# SECTION 5: ALPHA CORE EMERGENCE
# -----------------------------------------------------------------------------
# The alpha core is a purely geometric object emerging from three postulates:
#   P1: charge = amplitude
#   P2: cone--sphere emission geometry
#   P3: natural length ratios (r = x, l = pi*x)
#
# This section contains only two pure functions:
#   compute_alpha_core()  -> A_pi
#   compute_alpha_geometric(eps_M) -> full geometric inverse alpha
# =============================================================================

def compute_alpha_core() -> float:
    """
    Compute the purely geometric core of the fine-structure constant
    from the emission surface ratio.

    The derivation uses three geometric postulates:
      P1: charge = amplitude (x)
      P2: emission surface = sphere + cone
      P3: natural length ratios: r = x, l = pi*x

    Returns
    -------
    float
        A_pi = S_total / x^2 = 4*pi^3 + pi^2 + pi
    """
    x = 1.0
    r = x
    l = PI * x

    S_sphere = 4.0 * PI * l**2
    S_cone   = PI * r * l + PI * r**2

    S_total = S_sphere + S_cone
    A_pi = S_total / x**2

    return A_pi


def compute_alpha_geometric(eps_M: float) -> float:
    """
    Compute the full geometric inverse fine-structure constant.

    Formula:
        alpha^{-1} = A_pi - eps_M

    Parameters
    ----------
    eps_M : float
        Magnetic deficit parameter.

    Returns
    -------
    float
        Geometric inverse fine-structure constant.
    """
    A_pi = compute_alpha_core()
    return A_pi - eps_M

def test_alpha_sector(alpha_inv_pred: float) -> dict:
    """
    Compare the predicted inverse fine-structure constant with CODATA 2022.

    Parameters
    ----------
    alpha_inv_pred : float
        Predicted inverse fine-structure constant.

    Returns
    -------
    dict with:
        alpha_inv_pred : predicted value
        abs_err        : absolute error vs CODATA
        rel_err        : relative error in percent
    """
    abs_err = abs(alpha_inv_pred - ALPHA_INV_CODATA)
    rel_err = abs_err / ALPHA_INV_CODATA * 100.0

    return {
        "alpha_inv_pred": alpha_inv_pred,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }

# =============================================================================
# SECTION 6: NEUTRINO & ELECTRON ANCHORS
# -----------------------------------------------------------------------------
# This section derives hbar, Planck charge, uncorrected wavelength, and
# the neutrino radius from geometric alpha and electron anchors.
# =============================================================================

def derive_planck_charge_from_e(alpha_geom: float, e_charge: float) -> float:
    """
    Derive Planck charge (fundamental geometric amplitude) from 
    geometric alpha and the elementary charge amplitude e.

    Formula:
        q_P = e / sqrt(alpha_geom)

    Note: This is NOT a tautology because alpha_geom is derived
    from pure BCC geometry (A_pi - eps_M), not from e or q_P.
    """
    return e_charge / math.sqrt(alpha_geom)


def derive_hbar(alpha_geom: float, r_e: float, m_e: float, c0: float) -> float:
    """
    Derive the reduced Planck constant from geometric alpha and electron data.

    Formula:
        hbar = m_e * c * r_e / alpha_geom

    Parameters
    ----------
    alpha_geom : float
        Geometric fine-structure constant (dimensionless).
    r_e : float
        Classical electron radius [m].
    m_e : float
        Electron mass [kg].
    c0 : float
        Speed of light [m/s].

    Returns
    -------
    float
        Reduced Planck constant [kg m^2 / s].
    """
    return (m_e * c0 * r_e) / alpha_geom


def derive_lambda_uncorr(q_P: float) -> float:
    """
    Compute the uncorrected neutrino wavelength from Planck charge.

    Formula:
        lambda_uncorr = 2 * q_P * e^2

    where e is Euler's number.

    Parameters
    ----------
    q_P : float
        Planck charge (in EWT interpreted as length).

    Returns
    -------
    float
        Uncorrected wavelength [m].
    """
    return 2.0 * q_P * EULER**2


def derive_neutrino_radius(alpha_geom: float, q_P: float) -> dict:
    """
    Derive the neutrino radius r_nu from the geometric fixed point of g_v.

    The derivation uses the self-consistent equation:
        S_tot = alpha_geom_inv/(8+pi) + e + (1-g_v)*(sqrt2-1)
        S_tot = 2 e^2 / g_v
    which leads to a quadratic equation for g_v.

    Parameters
    ----------
    alpha_geom : float
        Geometric fine-structure constant.
    q_P : float
        Planck charge / geometric amplitude [m].

    Returns
    -------
    dict with:
        alpha_inv_geom : geometric inverse alpha
        S_proj         : static lattice projection
        S_exp          : dynamic wave expansion (e)
        delta_imp      : lattice impedance term
        gv_pred        : geometric fixed point for g_v
        S_tot          : total scaling factor
        r_nu           : neutrino radius [m]
    """
    alpha_inv_geom = 1.0 / alpha_geom

    S_proj = alpha_inv_geom / (BCC_NEIGHBOURS + PI)
    S_exp = EULER

    a_coef = SQRT2 - 1.0
    b_coef = -(S_proj + EULER + SQRT2 - 1.0)
    c_coef = 2.0 * EULER**2

    discriminant = b_coef**2 - 4.0 * a_coef * c_coef
    if discriminant < 0:
        raise ValueError("Negative discriminant in g_v fixed point equation")

    sqrt_disc = math.sqrt(discriminant)
    root1 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    root2 = (-b_coef - sqrt_disc) / (2.0 * a_coef)

    if 0.0 < root1 < 1.0:
        gv_pred = root1
    elif 0.0 < root2 < 1.0:
        gv_pred = root2
    else:
        raise ValueError("No physical root for g_v in (0,1)")

    delta_imp = (1.0 - gv_pred) * (SQRT2 - 1.0)
    S_tot = S_proj + S_exp + delta_imp
    r_nu = q_P * S_tot

    return {
        "alpha_inv_geom": alpha_inv_geom,
        "S_proj": S_proj,
        "S_exp": S_exp,
        "delta_imp": delta_imp,
        "gv_pred": gv_pred,
        "S_tot": S_tot,
        "r_nu": r_nu,
    }


def test_decadic_resonance(r_e: float, r_nu: float) -> dict:
    """
    Test the 1:100 radial resonance between electron and neutrino.

    Parameters
    ----------
    r_e : float
        Classical electron radius [m].
    r_nu : float
        Neutrino radius [m].

    Returns
    -------
    dict with:
        ratio      : r_e / r_nu
        expected   : 100.0
        rel_err    : relative error in percent
        K_implied  : ratio^5
    """
    ratio = r_e / r_nu
    rel_err = abs(ratio - 100.0) / 100.0 * 100.0
    K_implied = ratio**5
    return {
        "ratio": ratio,
        "expected": 100.0,
        "rel_err": rel_err,
        "K_implied": K_implied,
    }

def compute_C_unif(alpha_geom: float, L_p_geom: float, K_WC: int) -> float:
    """
    Compute the unified coupling operator C_Unif.

    Formula:
        C_Unif = 1/K_WC + 1 + alpha_geom / (pi * L_p_geom)

    Parameters
    ----------
    alpha_geom : float
        Geometric fine-structure constant.
    L_p_geom : float
        Ideal BCC projection factor.
    K_WC : int
        Number of wave centres in the electron.

    Returns
    -------
    float
        Unified coupling operator.
    """
    return (1.0 / K_WC) + 1.0 + (alpha_geom / (PI * L_p_geom))


def compute_X_eff(alpha_geom: float, L_p_geom: float, K_WC: int) -> float:
    """
    Compute the geometric dilution factor X_eff.

    Formula:
        X_eff = A_pi * 3 * K_WC * sqrt(2) / C_Unif

    Parameters
    ----------
    alpha_geom : float
        Geometric fine-structure constant.
    L_p_geom : float
        Ideal BCC projection factor.
    K_WC : int
        Number of wave centres in the electron.

    Returns
    -------
    float
        Dilution factor X_eff.
    """
    A_pi = compute_alpha_core()
    C_Unif = compute_C_unif(alpha_geom, L_p_geom, K_WC)
    return (A_pi * 3.0 * K_WC * SQRT2) / C_Unif

def derive_lambda_l_geometric(
    alpha_geom: float,
    r_e: float,
    r_nu: float,
    N_geom: float,
    L_p_geom: float,
    K_WC: int,
) -> float:
    """
    Derive the fundamental EMC length lambda_l from pure geometry.

    The formula is the analytic solution of the self-consistent equation
    obtained by combining the geometric G with the Planck length definition.
    It does NOT use hbar or G as input.

    Formula:
        lambda_l = [ r_e^2 * sqrt(X_eff) /
                     ( alpha_geom * A_pi^4 * N_geom^3 * K_WC *
                       (r_nu/(2*e))^(3/2) ) ]^2

    Parameters
    ----------
    alpha_geom : float
        Geometric fine-structure constant.
    r_e : float
        Classical electron radius [m].
    r_nu : float
        Neutrino radius [m].
    N_geom : float
        Effective BCC stiffness.
    L_p_geom : float
        Ideal BCC projection factor.
    K_WC : int
        Number of wave centres in the electron.

    Returns
    -------
    float
        Derived lambda_l [m].
    """
    A_pi = compute_alpha_core()
    X_eff = compute_X_eff(alpha_geom, L_p_geom, K_WC)

    r_nu_factor = (r_nu / (2.0 * EULER)) ** (3.0 / 2.0)

    bracket = (
        r_e**2 * math.sqrt(X_eff)
        / (alpha_geom * A_pi**4 * N_geom**3 * K_WC * r_nu_factor)
    )

    return bracket**2

# =============================================================================
# SECTION 7: GRAVITY SECTOR EMERGENCE
# -----------------------------------------------------------------------------
# Gravity is derived from the electron soliton and the BCC lattice geometry.
# No calibrated N_final and no calibrated L_p are used.
#
#   N_geom       : derived from BCC packing impedance
#   L_p_geom     : ideal BCC projection factor 2/sqrt(3)
#   alpha_geom   : derived from A_pi and eps_M
#
# The effective volume deficit is computed via the unified coupling:
#   C_Unif = 1/K_WC + 1 + alpha_geom/(pi * L_p_geom)
# =============================================================================

def gravity_sector(
    alpha_geom: float,
    r_nu: float,
    N_geom: float,
    L_p_geom: float,
    K_WC: int,
    lambda_l: float,
    r_e: float,
    m_e: float,
    c0: float,
) -> dict:
    """
    Derive the gravitational constant G from pure geometry and electron anchors.

    Parameters
    ----------
    alpha_geom : float
        Geometric fine-structure constant.
    r_nu : float
        Neutrino radius [m].
    N_geom : float
        Effective BCC stiffness derived from packing impedance.
    L_p_geom : float
        Ideal BCC projection factor (2/sqrt(3)).
    K_WC : int
        Number of wave centres in the electron (10).
    lambda_l : float
        Fundamental EMC spacing derived from geometry [m].
    r_e : float
        Classical electron radius [m].
    m_e : float
        Electron mass [kg].
    c0 : float
        Speed of light [m/s].

    Returns
    -------
    dict with:
        G_Base        : base soliton scaling
        N_nu_statutory: statutory background density
        C_Unif        : unified coupling operator
        X_eff         : dilution factor
        N_nu_eff      : effective volume deficit
        G_EWT         : predicted gravitational constant
        rel_err       : relative error vs CODATA
    """
    G_Base = (c0**2 * r_e) / m_e
    A_pi = compute_alpha_core()
    C_Unif = compute_C_unif(alpha_geom, L_p_geom, K_WC)
    X_eff = compute_X_eff(alpha_geom, L_p_geom, K_WC)
    N_nu_statutory = (r_nu / (2.0 * lambda_l * EULER)) ** 3
    N_nu_eff = N_nu_statutory / X_eff

    G_EWT = (
        (G_Base / A_pi)
        * (1.0 / (N_geom * A_pi)) ** 3
        * (1.0 / (K_WC * math.sqrt(N_nu_eff)))
    )

    rel_err = abs(G_EWT - G_CODATA) / G_CODATA * 100.0

    return {
        "G_Base": G_Base,
        "N_nu_statutory": N_nu_statutory,
        "C_Unif": C_Unif,
        "X_eff": X_eff,
        "N_nu_eff": N_nu_eff,
        "G_EWT": G_EWT,
        "rel_err": rel_err,
    }


def report_gravity(res: dict) -> None:
    """Print gravity sector results."""
    print(f"  G_Base        : {res['G_Base']:.15e}")
    print(f"  N_nu_statutory: {res['N_nu_statutory']:.15e}")
    print(f"  C_Unif        : {res['C_Unif']:.15f}")
    print(f"  X_eff         : {res['X_eff']:.10f}")
    print(f"  N_nu_eff      : {res['N_nu_eff']:.15e}")
    print(f"  G_EWT         : {res['G_EWT']:.15e}")
    print(f"  G_CODATA      : {G_CODATA:.15e}")
    print(f"  rel_error     : {res['rel_err']:.6f} %")

# =============================================================================
# MAIN
# =============================================================================

def rigidity_test(
    alpha_geom: float,
    r_e: float,
    r_nu: float,
    N_geom: float,
    L_p_geom: float,
    K_WC: int,
    lambda_l_geom: float,
    m_e: float,
    c0: float,
) -> None:
    """
    Test the geometric rigidity of the model by perturbing key parameters.

    Shows how relative changes in N, L_p, or lambda_l propagate into
    alpha, G, and the derived lambda_l.
    """
    print("\n" + "=" * 78)
    print("RIGIDITY TEST")
    print("=" * 78)

    print("\n[1] Vary N_geom")
    print("-" * 78)
    print(f"{'N_geom':>12} {'alpha_inv':>14} {'lambda_l [m]':>16} {'G_EWT [m^3/kg/s^2]':>22} {'G_err %':>10}")
    for factor in [0.995, 0.998, 1.0, 1.002, 1.005]:
        N_test = N_geom * factor
        eps_M_test = 1.0 / (N_test * PI**3)
        alpha_inv_test = compute_alpha_geometric(eps_M_test)
        alpha_test = 1.0 / alpha_inv_test

        lambda_test = derive_lambda_l_geometric(
            alpha_geom=alpha_test,
            r_e=r_e,
            r_nu=r_nu,
            N_geom=N_test,
            L_p_geom=L_p_geom,
            K_WC=K_WC,
        )

        res = gravity_sector(
            alpha_geom=alpha_test,
            r_nu=r_nu,
            N_geom=N_test,
            L_p_geom=L_p_geom,
            K_WC=K_WC,
            lambda_l=lambda_test,
            r_e=r_e,
            m_e=m_e,
            c0=c0,
        )

        print(
            f"{N_test:12.5f} {alpha_inv_test:14.9f} {lambda_test:16.6e} "
            f"{res['G_EWT']:22.15e} {res['rel_err']:10.6f}"
        )

    print("\n[2] Vary L_p_geom")
    print("-" * 78)
    print(f"{'L_p':>10} {'alpha_inv':>14} {'lambda_l [m]':>16} {'G_EWT [m^3/kg/s^2]':>22} {'G_err %':>10}")
    for factor in [0.995, 0.998, 1.0, 1.002, 1.005]:
        Lp_test = L_p_geom * factor

        lambda_test = derive_lambda_l_geometric(
            alpha_geom=alpha_geom,
            r_e=r_e,
            r_nu=r_nu,
            N_geom=N_geom,
            L_p_geom=Lp_test,
            K_WC=K_WC,
        )

        res = gravity_sector(
            alpha_geom=alpha_geom,
            r_nu=r_nu,
            N_geom=N_geom,
            L_p_geom=Lp_test,
            K_WC=K_WC,
            lambda_l=lambda_test,
            r_e=r_e,
            m_e=m_e,
            c0=c0,
        )

        print(
            f"{Lp_test:10.6f} {1.0/alpha_geom:14.9f} {lambda_test:16.6e} "
            f"{res['G_EWT']:22.15e} {res['rel_err']:10.6f}"
        )

    print("\n[3] Vary lambda_l directly")
    print("-" * 78)
    print(f"{'lambda_l [m]':>16} {'G_EWT [m^3/kg/s^2]':>22} {'G_err %':>10}")
    for factor in [0.99, 1.0, 1.01]:
        lambda_test = lambda_l_geom * factor
        res = gravity_sector(
            alpha_geom=alpha_geom,
            r_nu=r_nu,
            N_geom=N_geom,
            L_p_geom=L_p_geom,
            K_WC=K_WC,
            lambda_l=lambda_test,
            r_e=r_e,
            m_e=m_e,
            c0=c0,
        )
        print(
            f"{lambda_test:16.6e} {res['G_EWT']:22.15e} {res['rel_err']:10.6f}"
        )

    print("=" * 78)

def input_rigidity_test(base_alpha_geom, base_r_nu, base_N_geom, base_Lp):
    print("\n[4] Vary fundamental inputs r_e and m_e")
    print("-" * 78)
    print(f"{'factor':>8} {'r_e [m]':>16} {'m_e [kg]':>16} {'lambda_l [m]':>18} {'G_err %':>10}")

    for factor in [0.998, 0.999, 1.0, 1.001, 1.002]:
        r_e_test = R_E * factor
        m_e_test = M_E * factor

        lambda_test = derive_lambda_l_geometric(
            alpha_geom=base_alpha_geom,
            r_e=r_e_test,
            r_nu=base_r_nu,
            N_geom=base_N_geom,
            L_p_geom=base_Lp,
            K_WC=10,
        )

        res = gravity_sector(
            alpha_geom=base_alpha_geom,
            r_nu=base_r_nu,
            N_geom=base_N_geom,
            L_p_geom=base_Lp,
            K_WC=10,
            lambda_l=lambda_test,
            r_e=r_e_test,
            m_e=m_e_test,
            c0=C0,
        )

        print(
            f"{factor:8.3f} {r_e_test:16.6e} {m_e_test:16.6e} "
            f"{lambda_test:18.6e} {res['rel_err']:10.6f}"
        )

# =============================================================================
# SECTION 8: LEPTON ANOMALOUS MAGNETIC MOMENTS (AMM)
# =============================================================================

def get_AMMi_K(n: int) -> int:
    """
    Recursive nodal count for the lepton generations.

    K_1 = 10
    K_n = K_{n-1} + round(10^(n-1) * 2*pi^2)

    Parameters
    ----------
    n : int
        Generation number (1 = electron, 2 = muon, 3 = tau).

    Returns
    -------
    int
        Total nodal count K_n.
    """
    if n == 1:
        return 10
    else:
        delta_K = round(10 ** (n - 1) * (2.0 * PI**2))
        return get_AMMi_K(n - 1) + delta_K


def compute_lepton_amms(
    alpha_geom: float,
    eps_M: float,
    L_mu_dim: int = 5,
    L_tau_dim: int = 34,
    K_WC_e: int = 10,
) -> dict:
    """
    Compute the full anomalous magnetic moments for the lepton family.
    """
    A_pi = compute_alpha_core()

    a_e_ppm = (alpha_geom / (2.0 * PI)) * (1.0 - eps_M * (PI**3)) * 1e6

    K_e = K_WC_e
    K_mu_total = get_AMMi_K(2)
    K_mu_delta = K_mu_total - K_e
    M_mu_shell = K_mu_delta / K_e

    B_mu_scale = (3.0 * A_pi * PI**3) / (2.0 * L_mu_dim**2)
    a_mu_shell_ppm = B_mu_scale * (1.0 - eps_M) ** (M_mu_shell * PI**3)

    O_mu = 1.0 / (4.0 * PI**2)
    a_mu_shell_correction = a_mu_shell_ppm * O_mu
    a_mu_ppm = a_e_ppm + a_mu_shell_correction

    K_tau_total = get_AMMi_K(3)
    M_tau_rel = K_tau_total / K_e

    B_tau_base = ((3.0 * A_pi * PI**3) / (8.0 * SQRT2)) + (A_pi / 2.0)
    a_tau_shell_raw_ppm = B_tau_base * (1.0 - eps_M) ** (M_tau_rel * PI**3)

    a_tau_shell_total_ppm = a_mu_shell_ppm + a_tau_shell_raw_ppm + L_mu_dim**2

    O_tau = 1.0
    a_tau_ppm = a_e_ppm + O_tau * (a_tau_shell_total_ppm - a_e_ppm)

    return {
        "a_e_ppm": a_e_ppm,
        "a_mu_ppm": a_mu_ppm,
        "a_tau_ppm": a_tau_ppm,
        "a_mu_shell_ppm": a_mu_shell_ppm,
        "a_tau_shell_ppm": a_tau_shell_total_ppm,
    }


def report_lepton_amms(res: dict) -> None:
    """Print lepton AMM results and compare with experimental targets."""
    print("--- Lepton Anomalous Magnetic Moments ---")
    print(f"  Electron a_e   : {res['a_e_ppm']:.6f} ppm  (CODATA: {A_E_CODATA*1e6:.6f} ppm)")
    print(f"  Muon shell     : {res['a_mu_shell_ppm']:.6f} ppm")
    print(f"  Muon full a_mu : {res['a_mu_ppm']:.6f} ppm  (Exp: {A_MU_EXP*1e6:.6f} ppm)")
    print(f"  Tau shell      : {res['a_tau_shell_ppm']:.6f} ppm")
    print(f"  Tau full a_tau : {res['a_tau_ppm']:.6f} ppm  (Exp: {A_TAU_EXP*1e6:.6f} ppm)")

def amm_rigidity_test(
    base_alpha_geom: float,
    base_eps_M: float,
    L_mu_dim: int = 5,
    L_tau_dim: int = 34,
) -> None:
    """
    Test the rigidity of the lepton AMM sector against changes in eps_M.
    """
    print("\n" + "=" * 78)
    print("AMM RIGIDITY TEST (vary eps_M)")
    print("=" * 78)
    print(f"{'factor':>8} {'eps_M':>18} {'alpha_inv':>14} {'a_e [ppm]':>14} {'a_mu [ppm]':>14} {'a_tau [ppm]':>14}")
    print("-" * 78)

    for factor in [0.998, 0.999, 1.0, 1.001, 1.002]:
        eps_test = base_eps_M * factor
        alpha_inv_test = compute_alpha_geometric(eps_test)
        alpha_test = 1.0 / alpha_inv_test

        res = compute_lepton_amms(
            alpha_geom=alpha_test,
            eps_M=eps_test,
            L_mu_dim=L_mu_dim,
            L_tau_dim=L_tau_dim,
        )

        print(
            f"{factor:8.3f} {eps_test:18.6e} {alpha_inv_test:14.9f} "
            f"{res['a_e_ppm']:14.6f} {res['a_mu_ppm']:14.6f} {res['a_tau_ppm']:14.6f}"
        )

    print("-" * 78)
    print("Reference experimental:")
    print(f"  a_e   : {A_E_CODATA*1e6:.6f} ppm")
    print(f"  a_mu  : {A_MU_EXP*1e6:.6f} ppm")
    print(f"  a_tau : {A_TAU_EXP*1e6:.6f} ppm")
    print("=" * 78)

# =============================================================================
# SECTION 9: ATOMIC SCALES FROM PURE GEOMETRY
# =============================================================================

def compute_atomic_scales(alpha_geom: float, r_e_geom: float) -> dict:
    """
    Compute atomic scales from geometric alpha and geometric electron radius.
    """
    R_inf = (alpha_geom**3) / (4.0 * PI * r_e_geom)
    a0 = r_e_geom / (alpha_geom**2)
    lambda_C = (2.0 * PI * r_e_geom) / alpha_geom

    return {
        "R_inf": R_inf,
        "a0": a0,
        "lambda_C": lambda_C,
    }


def report_atomic_scales(res: dict) -> None:
    """Print atomic scale results and compare with CODATA 2022."""
    print("--- Atomic Scales from Pure Geometry ---")
    print(f"  Rydberg R_inf   : {res['R_inf']:.8f} m^-1")
    print(f"    CODATA        : {R_INF_CODATA:.8f} m^-1")
    print(f"    rel_error     : {abs(res['R_inf'] - R_INF_CODATA)/R_INF_CODATA*100:.6f} %")

    print(f"  Bohr radius a0  : {res['a0']:.15e} m")
    print(f"    CODATA        : {A0_CODATA:.15e} m")
    print(f"    rel_error     : {abs(res['a0'] - A0_CODATA)/A0_CODATA*100:.6f} %")

    print(f"  Compton lambda_C: {res['lambda_C']:.15e} m")
    print(f"    CODATA        : {LAMBDA_C_CODATA:.15e} m")
    print(f"    rel_error     : {abs(res['lambda_C'] - LAMBDA_C_CODATA)/LAMBDA_C_CODATA*100:.6f} %")

def main():
    print("ENHANCED EWT - EMERGENCE ENGINE (PROTOTYPE)")

    A_pi = compute_alpha_core()
    print(f"\nAlpha core A_pi = 4*pi^3 + pi^2 + pi = {A_pi:.15f}")

    bcc_derived = derive_eps_M_from_BCC(8.0 * PI**4)
    eps_M_bcc = bcc_derived["eps_M"]

    print("\nDerived eps_M from BCC packing:")
    print(f"  eta      = {bcc_derived['eta']:.10f}")
    print(f"  N_ideal  = {bcc_derived['N_ideal']:.10f}")
    print(f"  zeta     = {bcc_derived['zeta']:.10f}")
    print(f"  N_geom   = {bcc_derived['N_geom']:.10f}")
    print(f"  eps_M    = {bcc_derived['eps_M']:.15e}")

    alpha_inv_bcc = compute_alpha_geometric(eps_M_bcc)
    alpha_geom = 1.0 / alpha_inv_bcc

    res_alpha_bcc = test_alpha_sector(alpha_inv_bcc)

    print("\nAlpha inverse fine-structure constant test:")
    print(f"  Pure BCC geometry  : pred = {res_alpha_bcc['alpha_inv_pred']:.12f} | rel_err = {res_alpha_bcc['rel_err']:.6f} %")
    print(f"  CODATA 2022        : {ALPHA_INV_CODATA:.12f}")

    hbar_geom = derive_hbar(alpha_geom, R_E, M_E, C0)
    q_P_geom = derive_planck_charge_from_e(alpha_geom, E_CHARGE_CODATA)

    print(f"Derived q_P         : {q_P_geom:.15e}")
    print(f"Reference Q_P_INPUT : {Q_P_INPUT:.15e}")

    ratio_qP_e_derived = q_P_geom / E_CHARGE_CODATA
    ratio_qP_e_expected = 1.0 / math.sqrt(alpha_geom)

    print(f"\nGeometric ratio q_P / e:")
    print(f"  derived  q_P / e = {ratio_qP_e_derived:.10f}")
    print(f"  expected 1/sqrt(alpha_geom) = {ratio_qP_e_expected:.10f}")
    print(f"  difference              = {abs(ratio_qP_e_derived - ratio_qP_e_expected):.6e}")

    lambda_unc = derive_lambda_uncorr(q_P_geom)

    print(f"\nDerived hbar        : {hbar_geom:.15e}")
    print(f"Using q_P input     : {q_P_geom:.15e}")
    print(f"lambda_uncorr       : {lambda_unc:.15e}")

    nu_result = derive_neutrino_radius(alpha_geom, q_P_geom)
    print(f"\nNeutrino radius r_nu : {nu_result['r_nu']:.15e} m")
    print(f"g_v fixed point      : {nu_result['gv_pred']:.15f}")

    res_nu = test_decadic_resonance(R_E, nu_result['r_nu'])
    print(f"\nElectron/neutrino ratio : {res_nu['ratio']:.12f}")
    print(f"Expected                : {res_nu['expected']}")
    print(f"Relative error         : {res_nu['rel_err']:.6f} %")
    print(f"K_implied (r^5)        : {res_nu['K_implied']:.12e}")

    lambda_l_geom = derive_lambda_l_geometric(
        alpha_geom=alpha_geom,
        r_e=R_E,
        r_nu=nu_result["r_nu"],
        N_geom=bcc_derived["N_geom"],
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=10,
    )

    print(f"\nDerived lambda_l (geometry): {lambda_l_geom:.15e} m")
    print(f"CODATA lambda_l            : {LAMBDA_L:.15e} m")
    print(f"Relative error             : {abs(lambda_l_geom - LAMBDA_L)/LAMBDA_L*100:.6f} %")

    print("\n--- Gravity Sector (Geometric L_p, derived lambda_l) ---")
    gravity_res = gravity_sector(
        alpha_geom=alpha_geom,
        r_nu=nu_result["r_nu"],
        N_geom=bcc_derived["N_geom"],
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=10,
        lambda_l=lambda_l_geom,
        r_e=R_E,
        m_e=M_E,
        c0=C0,
    )
    report_gravity(gravity_res)

    input_rigidity_test(alpha_geom, nu_result["r_nu"], bcc_derived["N_geom"], BCC_IDEAL_PROJECTION_LP)

    rigidity_test(
        alpha_geom=alpha_geom,
        r_e=R_E,
        r_nu=nu_result["r_nu"],
        N_geom=bcc_derived["N_geom"],
        L_p_geom=BCC_IDEAL_PROJECTION_LP,
        K_WC=10,
        lambda_l_geom=lambda_l_geom,
        m_e=M_E,
        c0=C0,
    )

    ladder_bcc = build_geometric_ladder(eps_M_bcc, "BCC packing derived")

    print(f"\n--- Geometric Ladder (Pure BCC) ---")
    report_ladder(ladder_bcc)
    run_sector_tests(ladder_bcc)

    amm_results = compute_lepton_amms(alpha_geom, eps_M_bcc)
    report_lepton_amms(amm_results)
    amm_rigidity_test(alpha_geom, eps_M_bcc)

    r_e_geom = 100.0 * nu_result["r_nu"]
    atomic_res = compute_atomic_scales(alpha_geom, r_e_geom)
    report_atomic_scales(atomic_res)

    print("\n" + "=" * 78)
    print("END OF TEST")
    print("=" * 78)

if __name__ == "__main__":
    main()
