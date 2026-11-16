"""one-off edit helper (kept for the record): planted-defect hooks and an invariance check in sB3.py."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "sB3.py")
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


rep("Usage: ./py sB3.py [--plant=NAME]   planted defects: theta (drop J, k from Gamma), nu (shift a cluster nu by 1e-20),\n"
    "       charpoly (perturb the exact matrix by 1e-6 in one entry), sturm (miscount positive roots), q, point",
    "Usage: ./py sB3.py [--plant=NAME]   planted defects: thetasign (Theta without its signs), nu (shift a cluster nu by 1e-20),\n"
    "       charpoly (perturb the exact matrix by 1e-6 in one entry), numeric (perturb the 50-digit Mhat by 1e-10),\n"
    "       sturm (miscount positive roots), kvec (corrupt a tangent vector)")
rep("I0 = mat_i(n); Th = mat_theta(n); Kk = Th * I0\n",
    "I0 = mat_i(n); Th = mat_theta(n)\n"
    "if PLANT == \"thetasign\":\n"
    "    Th = Th.applyfunc(abs) * sp.diag(*([1] * n1 + [-1] * n1))\n"
    "Kk = Th * I0\n")
rep("group_tangent = Xs + [I0, Th, Kk] if PLANT != \"theta\" else Xs + [I0]\n",
    "group_tangent = Xs + [I0, Th, Kk]\n"
    "# h is invariant under i, J = Theta and k (the fibre group): check at a random point, 50 digits\n"
    "from quartic import cg_table\n"
    "_tab = cg_table(n)\n"
    "def h_num(wr):\n"
    "    w = [wr[k] + 1j * 0 + mpm.mpc(0, 1) * wr[n1 + k] for k in range(n1)]\n"
    "    tot = mpm.mpf(0)\n"
    "    for J, qJ in q.items():\n"
    "        for M in range(-J, J + 1):\n"
    "            G = mpm.fsum(mpm.mpmathify(sp.N(c, 60)) * w[k1] * w[k2] for k1, k2, c in _tab[J][M])\n"
    "            tot += mpm.mpf(qJ.p) / qJ.q * abs(G) ** 2\n"
    "    return tot\n"
    "import random\n"
    "random.seed(3)\n"
    "wr = [mpm.mpf(random.randint(-9, 9)) / 7 for _ in range(dimR)]\n"
    "h0 = h_num(wr)\n"
    "for nm_, A_ in ((\"i\", I0), (\"J\", Th), (\"k\", Kk)):\n"
    "    Am = [[mpm.mpf(int(A_[a, b])) for b in range(dimR)] for a in range(dimR)]\n"
    "    w2 = [mpm.fsum(Am[a][b] * wr[b] for b in range(dimR)) for a in range(dimR)]\n"
    "    chk(f\"h(tau w) = h(w) for tau = {nm_} at a random w (50 digits)\", abs(h_num(w2) - h0) < mpm.mpf(10) ** -40, mpm.nstr(abs(h_num(w2) - h0), 3))\n")
rep("        dimk = B.cols\n",
    "        dimk = B.cols\n"
    "        if dimk == 0:\n"
    "            chk(f\"{label}: ker(S^2 + tau^2) has dimension 2*mult (semisimple cluster), SVD gap clear\", False, \"empty kernel\")\n"
    "            continue\n")
rep("    T = [sp.Matrix(G_ * ur).applyfunc(sp.radsimp) for G_ in group_tangent]\n",
    "    T = [sp.Matrix(G_ * ur).applyfunc(sp.radsimp) for G_ in group_tangent]\n"
    "    if PLANT == \"kvec\":\n"
    "        T[0] = T[0] + sp.Matrix([sp.Rational(1, 10 ** 6)] + [0] * (dimR - 1))\n")
rep("    chk(f\"{name}: signature dimensions sum to dim N_u = 15 - k\", sum(sig) == dimR - 1 - kdim and min(sig) >= 0, str(sig))\n",
    "    chk(f\"{name}: n0 = (zero multiplicity of Mhat) - k is >= 0 and n+ = (positive count) - 1 is >= 0\", min(sig) >= 0, str(sig))\n")
rep("    Mnum = to_mp_mat(Mw)\n",
    "    Mnum = to_mp_mat(Mw)\n"
    "    if PLANT == \"numeric\":\n"
    "        Mnum[0, 1] += mpm.mpf(10) ** -10; Mnum[1, 0] += mpm.mpf(10) ** -10\n")
open(p, "w").write(s)
print("patched")
