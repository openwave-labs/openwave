"""M5.39 R0: the integrity pass on the capstone v3.4 source archive, and the reruns of what ships.

    python3 m5_39_r0_integrity.py --zip <sm_from_ldgs_v3.4_source.zip> --work <an empty scratch folder>
                                  [--v32 <the unpacked v3.2 source tree>]

The archive is untrusted input: it is unpacked into --work (outside the repository), every shipped
script runs there as a subprocess with the interpreter isolated (-I for a standalone script, -E -s
for one that imports its own neighbors), on one thread, and nothing is imported into this process.

Checks, each with the input that makes it fail:
  1 the archive's md5 against the Zenodo record's;
  2 the three Card C code files against the digests of the Card C payload manifest;
  3 the checksum files of the two packets, every line;
  4 (with --v32) byte identity of the Card C code, the packets, the scorer and the five hadronic
    scripts between v3.2 and v3.4;
  5 the scorer's self-test;
  6 the five hadronic scripts v3.2 ships, against the values the text prints (REPRODUCED at the
    printed digits, DIFFERS with the difference, NOT RUNNABLE with the missing input);
  7 the three scripts of the STRING audit supplement, the same labels;
  8 one printed line of arithmetic (the hidden-local-symmetry ratio a and the coupling g_H).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RECORD_MD5 = "0986077ff1aed3d47c86d128ddcd8375"  # Zenodo record 24195369, the source archive
CARD_CODE = "frozen_packets/STRING_v0.6/sector_b_action"
PACKETS = ["frozen_packets/STRING_v0.6", "frozen_packets/LdGS_frozen_packets_v0.4_definition_freeze"]
FIVE = ["skyrme_shoot.py", "rho_soliton.py", "rho_sextic.py", "induced_rho.py", "parent_gmor.py"]
AUDIT = "frozen_packets/STRING_v0.6/audit_v0.6"

# (script, what, regex on the script's output, the value the text prints, half a unit of its last digit)
PRINTED = [
    ("skyrme_shoot.py", "Adkins-Nappi check, M_N (MeV)", r"Adkins-Nappi massive check.*?M_N=(\d+)", 938, 0.5),
    ("skyrme_shoot.py", "Adkins-Nappi check, Delta - N (MeV)", r"Adkins-Nappi massive check.*?Delta-N=(\d+)", 293, 0.5),
    ("skyrme_shoot.py", "heavy-rho limit, classical mass (MeV)", r"native e=g_rhopipi.*?M_cl=(\d+)", 1145, 0.5),
    ("skyrme_shoot.py", "heavy-rho limit, M_N (MeV)", r"native e=g_rhopipi.*?M_N=(\d+)", 1336, 0.5),
    ("skyrme_shoot.py", "heavy-rho limit, Delta - N (MeV)", r"native e=g_rhopipi.*?Delta-N=(\d+)", 763, 0.5),
    ("rho_soliton.py", "dynamical rho, classical mass (MeV)", r"rho\s+: classical mass (\d+)", 1025, 0.5),
    ("rho_soliton.py", "dynamical rho, rms baryon radius (fm)", r"rho\s+: classical mass \d+ MeV, baryon rms radius ([\d.]+)", 0.26, 0.005),
    ("rho_soliton.py", "Skyrme term alone, rms baryon radius (fm)", r"skyrme\s*: classical mass \d+ MeV, baryon rms radius ([\d.]+)", 0.36, 0.005),
    ("rho_sextic.py", "sextic at 1/4, mass (GeV)", r"sextic = 0.25 x.*?mass (\d+)", 1.30, 0.005),
    ("rho_sextic.py", "sextic at 1/2, mass (GeV)", r"sextic = 0.50 x.*?mass (\d+)", 1.43, 0.005),
    ("rho_sextic.py", "sextic at 1, mass (GeV)", r"sextic = 1.00 x.*?mass (\d+)", 1.59, 0.005),
    ("rho_sextic.py", "sextic at 1/4, rms (fm)", r"sextic = 0.25 x.*?baryon rms ([\d.]+)", 0.46, 0.005),
    ("rho_sextic.py", "sextic at 1/2, rms (fm)", r"sextic = 0.50 x.*?baryon rms ([\d.]+)", 0.52, 0.005),
    ("rho_sextic.py", "sextic at 1, rms (fm)", r"sextic = 1.00 x.*?baryon rms ([\d.]+)", 0.59, 0.005),
    ("induced_rho.py", "comparator, f_pi (MeV)", r"M=400, Lambda=650:.*?f_pi=([\d.]+)", 94.8, 0.05),
    ("induced_rho.py", "comparator, induced coupling", r"M=400, Lambda=650:.*?induced g_rho=([\d.]+)", 10.33, 0.005),
    ("parent_gmor.py", "parent loop, f_pi in the chiral limit (MeV)", r"f_pi\(chiral\)=([\d.]+)", 94.8, 0.05),
    ("parent_gmor.py", "parent loop, condensate scale (MeV)", r"<qq>\^\(1/3\)=-([\d.]+)", 173, 0.5),
    ("parent_gmor.py", "parent loop, pion mass (MeV)", r"GMOR m_pi=([\d.]+)", 139.5, 0.05),
]
GEV = {"sextic at 1/4, mass (GeV)", "sextic at 1/2, mass (GeV)", "sextic at 1, mass (GeV)"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(py, flags, script, cwd, timeout=1800):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    p = subprocess.run([py, *flags, script], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    here = str(Path(cwd).resolve())
    root = here.split("/replay")[0].split("/pristine")[0]  # the scratch folder stays out of the record
    return p.returncode, p.stdout.replace(root, "<work>"), p.stderr.replace(root, "<work>")


def check_sums(folder):
    rows, bad = (folder / "SHA256SUMS").read_text().splitlines(), []
    for row in rows:
        digest, name = row.split("  ", 1)
        target = folder / name
        if not target.exists() or sha256(target) != digest:
            bad.append(name)
    return len(rows), bad


def numeric_difference(a, b):
    """Largest relative difference between the numbers of two JSON documents of the same shape.
    Entries below 1e-11 in both documents are zero to round-off or error estimates of that size (the
    files hold differences such as E2 - 2 E1 of order 1e-15) and carry no digits to compare."""
    worst = 0.0
    if isinstance(a, dict):
        for k in a:
            worst = max(worst, numeric_difference(a[k], b[k]))
    elif isinstance(a, list):
        for x, y in zip(a, b):
            worst = max(worst, numeric_difference(x, y))
    elif isinstance(a, float) and isinstance(b, (float, int)):
        if abs(a) > 1e-11 or abs(b) > 1e-11:
            worst = abs(a - b) / max(abs(a), abs(b))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--v32", type=Path)
    ap.add_argument("--v35", type=Path, help="the unpacked v3.5 source tree, for the identity its release note states")
    ap.add_argument("--v36", type=Path, help="the unpacked v3.6 source tree, the same check")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "m5_39_r0_integrity.json")
    a = ap.parse_args()
    py = sys.executable
    out, ok = dict(task="M5.39 R0"), True

    md5 = hashlib.md5(a.zip.read_bytes()).hexdigest()
    out["archive"] = dict(md5=md5, record_md5=RECORD_MD5, matches=md5 == RECORD_MD5)
    ok &= md5 == RECORD_MD5
    if a.work.exists() and any(a.work.iterdir()):
        sys.exit("--work must be empty")
    pristine, replay = a.work / "pristine", a.work / "replay"
    with zipfile.ZipFile(a.zip) as z:
        z.extractall(pristine)
    shutil.copytree(pristine, replay)
    files = [p for p in pristine.rglob("*") if p.is_file()]
    out["archive"].update(files=len(files), scripts=sum(p.suffix == ".py" for p in files))

    # 2 the payload manifest
    manifest = {}
    for row in (pristine / "frozen_packets/CardC_payload_manifest/SHA256SUMS").read_text().splitlines():
        digest, name = row.split("  ", 1)
        manifest[Path(name).name] = manifest.get(Path(name).name, []) + [digest]
    code = {}
    for name in ("bq_core.py", "dn87_relax.py", "fast_quartic.py"):
        d = sha256(pristine / CARD_CODE / name)
        code[name] = dict(sha256=d, in_manifest=d in manifest.get(name, []))
        ok &= code[name]["in_manifest"]
    out["card_c_code"] = dict(files=code, manifest_entries_not_in_the_archive=sorted(
        n for n in ("final.npz", "run.json", "kinetic_operator.py") if not list(pristine.rglob(n))))

    # 3 the packets' checksum files
    out["packets"] = {}
    for packet in PACKETS:
        n, bad = check_sums(pristine / packet)
        out["packets"][packet] = dict(lines=n, failed=bad)
        ok &= not bad

    # 4 identity with v3.2
    if a.v32:
        names = [f"{CARD_CODE}/{n}" for n in ("bq_core.py", "dn87_relax.py", "fast_quartic.py", "inertia.py")]
        names += [f"scripts/{n}" for n in FIVE] + ["score_photon_point.py"]
        names += [str(p.relative_to(pristine)) for p in (pristine / "frozen_packets").rglob("*") if p.is_file()]
        names = sorted(set(names))
        differing = [n for n in names if not (a.v32 / n).exists() or sha256(a.v32 / n) != sha256(pristine / n)]
        only_32 = [str(p.relative_to(a.v32)) for p in (a.v32 / "frozen_packets").rglob("*") if p.is_file() and not (pristine / p.relative_to(a.v32)).exists()]
        out["identity_with_v3_2"] = dict(files_compared=len(names), differing=differing, only_in_v3_2=only_32)
        ok &= not differing and not only_32

    for key, later in (("identity_with_v3_5", a.v35), ("identity_with_v3_6", a.v36)):  # later releases: the same files, and the pre-registration
        if not later:
            continue
        names = [f"{CARD_CODE}/{n}" for n in ("bq_core.py", "dn87_relax.py", "fast_quartic.py", "inertia.py")]
        names += ["score_photon_point.py", "docs/prereg/photon_point_preregistration.md"]
        names += [str(p.relative_to(pristine)) for p in (pristine / "frozen_packets").rglob("*") if p.is_file()]
        names = sorted(set(names))
        differing = [n for n in names if not (later / n).exists() or sha256(later / n) != sha256(pristine / n)]
        only_later = [str(p.relative_to(later)) for p in (later / "frozen_packets").rglob("*") if p.is_file() and not (pristine / p.relative_to(later)).exists()]
        five = [n for n in FIVE if not (later / "scripts" / n).exists() or sha256(later / "scripts" / n) != sha256(pristine / "scripts" / n)]
        files_later = [p for p in later.rglob("*") if p.is_file()]
        out[key] = dict(files_compared=len(names), differing=differing, only_in_the_later_release=only_later, hadronic_scripts_differing=five,
                                         files=len(files_later), scripts=sum(p.suffix == ".py" for p in files_later))
        ok &= not differing and not only_later

    # 5 the scorer
    rc, so, se = run(py, ["-I", "score_photon_point.py"], "--selftest", replay)
    out["scorer_selftest"] = dict(exit=rc, passed=rc == 0 and "SELFTEST PASSED" in so, last_line=so.strip().splitlines()[-1] if so.strip() else "")
    ok &= out["scorer_selftest"]["passed"]

    # 6 the five hadronic scripts
    rows, outputs = [], {}
    for name in FIVE:
        rc, so, se = run(py, ["-I"], name, replay / "scripts")
        outputs[name] = dict(exit=rc, stdout=so.strip().splitlines(), stderr_tail=se.strip().splitlines()[-2:])
    for name, what, pattern, printed, half in PRINTED:
        so = "\n".join(outputs[name]["stdout"])
        m = re.search(pattern, so)
        if outputs[name]["exit"] != 0 or not m:
            rows.append(dict(script=name, quantity=what, printed=printed, label="NOT RUNNABLE"))
            ok = False
            continue
        value = float(m.group(1)) / (1000.0 if what in GEV else 1.0)
        label = "REPRODUCED" if round(abs(value - printed) * 1e6) < round(half * 1e6) else "DIFFERS"  # strictly inside half a unit
        ok &= label == "REPRODUCED"
        rows.append(dict(script=name, quantity=what, printed=printed, returned=value, label=label))
        if label == "DIFFERS" and what == "sextic at 1/2, mass (GeV)":
            rows[-1]["note"] = ("the script prints 1425 MeV, itself a rounding (1424.5 unrounded, found by the audit): two decimals of that are "
                                "1.42 GeV, and the text's 1.43 rounds the rounded value")
    m = re.search(r"skyrme\s*: classical mass (\d+)", "\n".join(outputs["rho_soliton.py"]["stdout"]))
    out["hadronic_scripts"] = dict(rows=rows, outputs=outputs, note=(
        "the variational script returns %s MeV for the Skyrme-term hedgehog whose mass the shooting script returns, and the text "
        "prints, as 1145 MeV" % (m.group(1) if m else "?")))

    # 7 the STRING audit supplement
    audit = {}
    rc, so, se = run(py, ["-I"], "scripts/verify_hashes.py", replay / AUDIT)
    audit["verify_hashes.py"] = dict(exit=rc, label="REPRODUCED" if rc == 0 and "PASS" in so else "NOT RUNNABLE",
                                     missing_input=None if rc == 0 else "audit_v0.6/SHA256SUMS (the packet's checksum file sits one level up)",
                                     stderr_tail=se.strip().splitlines()[-1:] if rc else [])
    rc, so, se = run(py, ["-I"], "scripts/replay_radial_strand.py", replay / AUDIT)
    m = re.search(r'"physical_tension_MeV_per_fm": ([\d.eE+-]+)', so)
    tension = float(m.group(1)) if m else None
    shipped = json.loads((pristine / AUDIT / "results/RADIAL_REPLAY.json").read_text())
    replayed = json.loads((replay / AUDIT / "results/RADIAL_REPLAY.json").read_text()) if rc == 0 else None
    audit["replay_radial_strand.py"] = dict(
        exit=rc, printed=0.24001768, returned=tension,
        label="REPRODUCED" if tension is not None and abs(tension - 0.24001768) <= 5e-9 else "DIFFERS",
        rewritten_json_identical_bytes=replayed is not None and sha256(pristine / AUDIT / "results/RADIAL_REPLAY.json") == sha256(replay / AUDIT / "results/RADIAL_REPLAY.json"),
        rewritten_json_largest_relative_difference=numeric_difference(shipped, replayed) if replayed else None)
    ok &= audit["replay_radial_strand.py"]["label"] == "REPRODUCED"
    rc, so, se = run(py, ["-E", "-s"], "scripts/screened_source_control.py", replay / AUDIT)
    audit["screened_source_control.py"] = dict(exit=rc, label="REPRODUCED" if rc == 0 else "NOT RUNNABLE",
                                               missing_input=None if rc == 0 else "audit_v0.6/archived_card_c/ (the folder its import path names)",
                                               stderr_tail=se.strip().splitlines()[-1:] if rc else [])
    if rc != 0:  # a declared substitution: the same three modules from the packet's own folder, digests as in the manifest
        sub = replay / AUDIT / "archived_card_c"
        sub.mkdir()
        for name in ("bq_core.py", "dn87_relax.py", "fast_quartic.py"):
            shutil.copy(pristine / CARD_CODE / name, sub / name)
        rc2, so2, se2 = run(py, ["-E", "-s"], "scripts/screened_source_control.py", replay / AUDIT)
        shipped = json.loads((pristine / AUDIT / "results/SCREENED_CONTROL.json").read_text())
        again = json.loads((replay / AUDIT / "results/SCREENED_CONTROL.json").read_text()) if rc2 == 0 else None
        audit["screened_source_control.py"]["with_the_packet_modules_substituted"] = dict(
            exit=rc2, largest_relative_difference_from_the_shipped_json=numeric_difference(shipped, again) if again else None,
            label="REPRODUCED" if again is not None and numeric_difference(shipped, again) <= 1e-10 else "DIFFERS")
    out["string_audit_supplement"] = audit

    # 8 the printed arithmetic of the hidden-local-symmetry anchors
    g, fpi, mrho = 5.98, 92.3, 775.26
    ratio = (mrho / (g * fpi)) ** 2
    aa = 4.0 / ratio
    rows = [dict(quantity="m_rho at a = 2 (MeV)", printed=780.6, returned=2 ** 0.5 * g * fpi, half=0.05),
            dict(quantity="(m_rho / 5.98 f_pi)^2", printed=1.973, returned=ratio, half=0.0005),
            dict(quantity="a", printed=2.0275, returned=aa, half=0.00005),
            dict(quantity="g_H", printed=5.899, returned=2 * g / aa, half=0.0005)]
    for r in rows:
        r["label"] = "REPRODUCED" if abs(r["returned"] - r["printed"]) <= r.pop("half") else "DIFFERS"
        ok &= r["label"] == "REPRODUCED"
    out["hidden_local_symmetry_arithmetic"] = dict(inputs=dict(g_rhopipi=g, f_pi=fpi, m_rho=mrho), rows=rows,
                                                   note="the printed a needs the rho mass 775.26 MeV; the 775.3 of the same sentence returns 2.0273")

    out["all_passed"] = bool(ok)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for r in out["hadronic_scripts"]["rows"]:
        print(f"{r['script']:18s} {r['quantity']:45s} printed {r['printed']} returned {r.get('returned')} {r['label']}")
    for k, v in audit.items():
        print(f"{k:28s} {v['label']}" + (f" (missing: {v['missing_input']})" if v.get("missing_input") else ""))
    for r in rows:
        print(f"{'arithmetic':18s} {r['quantity']:45s} printed {r['printed']} returned {r['returned']:.5f} {r['label']}")
    print("packets", out["packets"], "\ncode", {k: v["in_manifest"] for k, v in code.items()}, "\nidentity", out.get("identity_with_v3_2"),
          "\nidentity with v3.5", out.get("identity_with_v3_5"), "\nidentity with v3.6", out.get("identity_with_v3_6"))
    print("ALL PASSED" if ok else "SEE THE LABELS", flush=True)


if __name__ == "__main__":
    main()
