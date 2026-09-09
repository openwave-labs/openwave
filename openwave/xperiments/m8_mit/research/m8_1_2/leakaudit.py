#!/usr/bin/env python3
"""
leakaudit.py -- gate the blind handout before a unit sees it.

conventions-and-worklist.md asks a solver to compute values from scratch.  If any of those values
is already sitting in the handout, the run measures nothing.  This checks the handout for every
value the worklist asks for, and then plants one of each to prove the check can still fail.

Run it from anywhere; it finds the handout beside itself.

    python3 leakaudit.py

Exit 0 when the handout is clean, 1 when anything leaked or the handout is missing.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(HERE, "conventions-and-worklist.md")

# every value the worklist asks for, as it would have to appear if it leaked
TARGETS = {
    "N in both sectors":        [r"\b1287\b", r"\b2288\b", r"\b1716\b",
                                 r"d\s*/\s*\(\s*7\s*-\s*d", r"\{7\s*-\s*d\}"],
    "the four ray values":      [r"\b400\b", r"\b288\b", r"\b463\b"],
    "pyramid / prism values":   [r"1188", r"8800", r"9/35", r"200/903", r"12/25", r"23/10"],
    "octahedral row":           [r"24/77", r"6/11", r"1/7"],
    "norm of R_6":              [r"12/7", r"d\(7\s*-\s*d\)"],
    # NOT \b924\b: the handout legitimately asks for C(12,6)*||rho_6||^2, so 924 is
    # question-side. The alternating row itself is the answer, and that is what is watched.
    "Lambda":                   [r"-\s*6\s*,\s*15", r"\b15\s*,\s*-\s*20",
                                 r"1\s*,\s*36\s*,\s*225", r"\b12012\b"],
    # A3's constant is a VALUE, so a value gate can reach it. sqrt{7} is absent from the
    # handout, so this category costs nothing and closes one more graded claim.
    "the M_0 constant":         [r"\\sqrt\{?\s*7\}?", r"sqrt\(\s*7\s*\)"],
    "the constant c":           [r"7\s*}\s*{\s*13", r"7/13", r"13/7",
                                 r"\b12012\b", r"\\sqrt\{?\s*91", r"sqrt\(\s*91"],
    "w_0 and the ratio":        [r"w_0\s*=\s*7", r"7\s*-\s*d\}\{13"],
    "Sym^3 decomposition":      [r"1\s*,\s*3\s*,\s*4\s*,\s*5", r"no\s+\$?`?V_8"],
    "multiplicities 1 and 4":   [r"\\dim\s*\\mathcal\{E\}", r"=\s*1\b.*=\s*4\b"],
    "the 2I branching":         [r"\\mathbf\{3\}", r"3'\s*\+\s*4"],
    "constellation names":      [r"octahedr", r"prism", r"pyramid", r"hexagon", r"anticoheren"],
    "the D_3 quartic":          [r"148", r"231", r"-\s*20\s*x"],
    # --- added when the claims table was regrouped: these were CONVENTIONS in the first
    # --- version of the handout and are GRADED CLAIMS now, so the old list passed a handout
    # --- that printed two of them outright.  A forbidden list tracks a claims table; when the
    # --- table moves and the list does not, the gate certifies against yesterday's run.
    "group order and structure": [r"\b120\b", r"order\s+120", r"binary\s+icosahedral",
                                  r"\b2I\b", r"McKay", r"\bA_?5\b"],
    "the invariant-degree row":  [r"1\s*,\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*,\s*1",
                                  r"levels?\s+0\s+and\s+12", r"invariants?\s+occur\s+at"],
    "the branching":             [r"3\s*\\oplus\s*4", r"3\s*\+\s*4\s*=\s*7",
                                  r"multiplicity-free", r"two\s+constituents",
                                  r"complementary\s+in", r"\\sigma_3", r"\\sigma_4"],
    # "2 V_3" alone matches the legitimate "Sym^2 V_3" of worklist item 2, so the pattern has
    # to carry the decomposition context rather than the bare coefficient.
    "Sym^3 multiplicity":        [r"\b84\b", r"\\oplus\s*2\s*V_3", r"\+\s*2\s*V_3"],
    "the spin-8 evaluation":     [r"\b273\b", r"\b1092\b"],
    "the sector gap":            [r"49/156", r"28/39", r"21/52"],
    "the second plane":          [r"N_0", r"N_6", r"span\\?\{\s*N"],
    "the B vs rho witness":      [r"25/84", r"5\s*\\sqrt\{21\}", r"\\sqrt\{21\}/42"],
    "the degree-12 form":        [r"twelve\s+simple", r"I_\{?12", r"simple\s+roots"],
}


def _expand_binomials(text):
    """Rewrite \binom{a}{b} as its integer value.

    A constant leaked in binomial form escapes every numeral pattern: the gate caught
    "1287" and missed "\binom{13}{6} d/(7-d)", which is the same claim. The handout's own
    register is binomial, so this is the form a leak would most naturally take.
    """
    def sub(m):
        try:
            a, b = int(m.group(1)), int(m.group(2))
        except ValueError:
            return m.group(0)
        if not (0 <= b <= a <= 60):
            return m.group(0)
        num = 1
        for k in range(b):
            num = num * (a - k) // (k + 1)
        return str(num)
    return re.sub(r"\\binom\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}", sub, text)


def audit(text, label):
    quiet = label is None
    text = _expand_binomials(text)
    hits = []
    for name, pats in TARGETS.items():
        for p in pats:
            for m in re.finditer(p, text, re.I):
                ln = text[:m.start()].count("\n") + 1
                hits.append((name, p, ln, text.splitlines()[ln - 1].strip()[:90]))
    if not quiet:
        print("%-28s %d hit(s)" % (label, len(hits)))
        for n, p, ln, l in hits:
            print("   %-26s /%s/  line %d: %s" % (n, p, ln, l))
    return hits


def main():
    # A gate that reports PASS on a handout that is not there would be worse than no gate.
    if not os.path.exists(H):
        print("handout not found beside this script: %s" % H)
        print("leakaudit.py must sit in the same directory as conventions-and-worklist.md")
        return 1

    text = open(H).read()
    real = audit(text, "handout as written")

    # The audit must be able to fail, and per CATEGORY, not merely in aggregate. Planting one
    # blob and watching the total rise proves that SOME pattern fires; it says nothing about a
    # category whose patterns never match anything. Each category below gets its own probe and
    # must catch it, so a dead pattern is reported instead of passing quietly.
    print()
    # Probes are sourced from the CLAIMS TABLE, in the surface form a leak would most likely
    # take, and deliberately NOT copied from the pattern list. A harness whose probes are the
    # patterns only proves a regex matches itself; it cannot detect a category that watches one
    # form of a constant and misses two others. That is how "the constant c" came to catch
    # 7/13 and miss -sqrt(91)/12012, which is the form the paper actually writes.
    PROBES = {
        "N in both sectors":         r"N = \binom{13}{6}\, d/(7-d)",
        "the four ray values":       "the hexagon at 463",
        "pyramid / prism values":    "stationary at 12/25, value 9/35",
        "octahedral row":            "1/7, 0, 0, 0, 6/11, 0, 24/77",
        "norm of R_6":               "||R_6||^2 = d(7-d)/7",
        "Lambda":                    "Lambda = (1, -6, 15, -20, 15, -6, 1)",
        "the M_0 constant":          r"M_0(u) = -\\lVert u\\rVert^2 u/\\sqrt{7}",
        "the constant c":            r"c = -\sqrt{91}/12012",
        "w_0 and the ratio":         "w_0 = 7 and w_6/w_0 = (7-d)/(13d)",
        "Sym^3 decomposition":       "irreducible dimensions 1, 3, 4, 5",
        "multiplicities 1 and 4":    r"\dim \mathcal{E}_8 = 1",
        "the 2I branching":          r"\mathbf{3}' and \mathbf{4}",
        "constellation names":       "a trigonal prism and an octahedron",
        "the D_3 quartic":           "100|z|^4 - 20x^2 + 148y^2 + 463",
        "group order and structure": "the binary icosahedral group",
        "the invariant-degree row":  "1, 0, 0, 0, 0, 0, 1",
        "the branching":             r"multiplicity-free, 3 \oplus 4",
        "Sym^3 multiplicity":        r"V_1 \oplus 2 V_3 \oplus V_4",
        "the spin-8 evaluation":     r"\sqrt{273}/1092",
        "the sector gap":            "the gap is 49/156",
        "the second plane":          r"span\{N_0, N_6\}",
        "the B vs rho witness":      "||rho_2(v_3)||^2 = 25/84",
        "the degree-12 form":        "twelve simple roots",
    }
    missing = [c for c in TARGETS if c not in PROBES]
    if missing:
        print("AUDIT INCOMPLETE: no probe for %s" % ", ".join(missing))
        return 1
    dead = []
    for cat, probe in PROBES.items():
        caught = audit(text + "\n\nPLANTED: " + probe + "\n", None)
        if not any(h[0] == cat for h in caught):
            dead.append(cat)
    if dead:
        print("AUDIT IS DEAD for %d categor%s: %s"
              % (len(dead), "y" if len(dead) == 1 else "ies", ", ".join(dead)))
        print("Those patterns cannot fire, so they certify nothing. Do not fire a unit on this.")
        return 1
    print("mutation check: all %d categories caught their own planted probe." % len(PROBES))

    if real:
        print("\nHANDOUT LEAKS. Do not fire a unit on it until the hits above are removed.")
        return 1
    print("handout is clean.")
    print()
    print("SCOPE OF THIS GATE, so that the line above is not over-read:")
    print("  checked     values, names, and symbolic forms of values, including binomial and")
    print("              radical registers of the same constant.")
    print("  NOT checked a claim stated in prose, a structural hint, or a question that hands")
    print("              over the form of its own answer. Nothing here can reach those: several")
    print("              graded claims have no numeral for a forbidden list to watch.")
    print("  therefore   a clean run means no value leaked. It does NOT mean the handout is")
    print("              blind. That judgement belongs to the semantic audit, and this gate")
    print("              cannot substitute for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
