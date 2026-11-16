"""M5.41 R1 and R4: the calibration relations of the capstone's recalibration, as closed forms.

With the inherited length unit L and the bare gap Delta held fixed (m5_41_task_details.md, section 2):

  e0^2     = 2 I0 Delta L / (3 hbar c)      hbar c / L = e f_pi of the inherited constants (4.259 x 92.4 MeV)
  f_pi     = hbar c / (L e0)
  unit_E   = f_pi / e0                      the action energy unit
  inv_I    = e0^3 f_pi                      the inverse-inertia unit
  g_A      = (g_A e^2)_0 / e0^2
  mu_V     = 2 m_p |m_z| / (I Delta)        = the script's mu_V times the state's own gap over Delta
  rigid    = m_p / Delta

I0 and (g_A e^2)_0 are numbers of a state (or zero-spacing intercepts); nothing here is fitted.

--control reproduces the author's `calibration33.json` from the readouts of M5.40 on the author's
33^3 field, and the printed e and f_pi of the 55^3 field, and writes the comparison as JSON.
"""

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E_INHERITED, F_PI_INHERITED = 4.259, 92.4
FREQUENCY_UNIT = E_INHERITED * F_PI_INHERITED  # hbar c / L, MeV
DELTA = 293.0  # the calibrated bare gap, MeV
M_P = 938.272  # MeV, the value of m5_40_readouts.py
HBARC = 197.3269804  # MeV fm


def calibrate(inertia, gA_e2=None):
    """The frozen rule applied to an inertia (and, if given, to g_A e^2)."""
    e2 = 2.0 * inertia * DELTA / (3.0 * FREQUENCY_UNIT)
    e = math.sqrt(e2)
    f_pi = FREQUENCY_UNIT / e
    out = dict(inertia=inertia, e=e, f_pi_MeV=f_pi, action_energy_unit_MeV=f_pi / e, inverse_inertia_unit_MeV=e2 * FREQUENCY_UNIT,
               gap_check_MeV=1.5 * e2 * FREQUENCY_UNIT / inertia)
    if gA_e2 is not None:
        out.update(gA_e2=gA_e2, gA=gA_e2 / e2)
    return out


def own_gap(inertia):
    """The gap of a state at the inherited constants, (3/2) e^3 f_pi / I."""
    return 1.5 * E_INHERITED ** 3 * F_PI_INHERITED / inertia


def muV_calibrated(muV_script, inertia):
    """The script's mu_V (inherited constants) carried to the calibrated gap: 2 m_p |m_z| / (I Delta)."""
    return muV_script * own_gap(inertia) / DELTA


def rigid_muV():
    return M_P / DELTA


def rel(a, b):
    return abs(a - b) / abs(b)


def control(calibration_file, readouts_file):
    cal = json.loads(Path(calibration_file).read_text())
    rd = json.loads(Path(readouts_file).read_text())
    g33, g55 = rd["m5_40_author_g33"], rd["m5_40_author_g55"]
    if abs(g33["constants"]["m_p_MeV"] - M_P) > 0:
        raise RuntimeError("the proton mass of the readouts differs from this script's")
    c33 = calibrate(g33["inertia_mean"])
    first = [f for f in cal["fields"] if f["grid"] == 33 and f["stencil"] == "edge"][0]
    checks = []

    def check(name, ours, theirs, bound):
        d = rel(ours, theirs)
        checks.append(dict(check=name, ours=ours, file=theirs, relative_difference=d, bound=bound, met=bool(d <= bound)))

    check("frequency unit, hbar c / L", FREQUENCY_UNIT, cal["frequency_unit_MeV"], 1e-12)
    check("length unit L, fm", HBARC / FREQUENCY_UNIT, cal["length_unit_fm"], 1e-12)
    check("inertia of the 33^3 field (our readout against the file)", g33["inertia_mean"], first["inertia"], 1e-10)
    check("effective_e", c33["e"], cal["effective_e"], 1e-12)
    check("effective_f_pi_MeV", c33["f_pi_MeV"], cal["effective_f_pi_MeV"], 1e-12)
    check("inverse_inertia_unit_MeV", c33["inverse_inertia_unit_MeV"], cal["inverse_inertia_unit_MeV"], 1e-12)
    check("action_energy_unit_MeV", c33["action_energy_unit_MeV"], cal["action_energy_unit_MeV"], 1e-12)
    check("calibrated_bare_gap_MeV", c33["gap_check_MeV"], first["calibrated_bare_gap_MeV"], 1e-12)
    check("minimal_muV_muN", muV_calibrated(g33["muV"], g33["inertia_mean"]), first["minimal_muV_muN"], 1e-9)
    check("rigid_muV_muN", rigid_muV(), first["rigid_muV_muN"], 1e-9)
    c55 = calibrate(g55["inertia_mean"], g55["axial"]["e2_gA"])
    printed = [dict(check="printed e of the 55^3 field", ours=c55["e"], printed="4.311", met=f"{c55['e']:.3f}" == "4.311"),
               dict(check="printed f_pi of the 55^3 field, MeV", ours=c55["f_pi_MeV"], printed="91.29", met=f"{c55['f_pi_MeV']:.2f}" == "91.29")]
    return dict(constants=dict(e_inherited=E_INHERITED, f_pi_inherited_MeV=F_PI_INHERITED, frequency_unit_MeV=FREQUENCY_UNIT,
                               delta_MeV=DELTA, m_p_MeV=M_P), from_the_33_field=c33, from_the_55_field=c55,
                against_the_file=checks, against_the_printed_digits=printed,
                all_met=bool(all(c["met"] for c in checks + printed)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--calibration-file", type=Path, help="the author's calibration33.json")
    ap.add_argument("--readouts", type=Path, default=HERE.parent / "data" / "m5_40_r4_readouts.json")
    ap.add_argument("--inertia", type=float)
    ap.add_argument("--gA-e2", type=float)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.control:
        out = control(a.calibration_file, a.readouts)
    elif a.inertia is not None:
        out = calibrate(a.inertia, a.gA_e2)
    else:
        ap.error("give --control or --inertia")
    text = json.dumps(out, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    print(text)
    if a.control and not out["all_met"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
