"""The distinct critical orbits (item 4) with an exact representative each (unit vectors)."""
import sympy as sp
import classes as K

s2 = sp.sqrt(2)
R = sp.Rational
def unit(v):
    n = sp.sqrt(sp.nsimplify(sum(sp.expand(z * sp.conjugate(z)) for z in v)))
    return (v / n).applyfunc(sp.radsimp)

ORBITS = {
    'v3':  dict(u=K.vec({3: 1}), stab='SO(2)', where='line class v3; poles of B, D, E'),
    'v2':  dict(u=K.vec({2: 1}), stab='SO(2)', where='line class v2; poles of A (v-2), C, D'),
    'v1':  dict(u=K.vec({1: 1}), stab='SO(2)', where='line class v1; poles of A, B'),
    'v0':  dict(u=K.vec({0: 1}), stab='O(2)', where='line class v0; z=0 of F and G'),
    'xyz': dict(u=K.CLASSES1['xyz']['u'], stab='O', where='line class xyz; circle of C; omitted point of F; z=+-2/sqrt5 of G'),
    'cat': dict(u=K.CLASSES1['cat']['u'], stab='D6', where='line class cat; circle of E; z=+-sqrt(3/5) of F; omitted point of G'),
    'A*':  dict(u=unit(K.vec({1: 1, -2: R(1, 2)})), stab='C3', where='circle |z|^2=1/4 of A'),
    'D*':  dict(u=unit(K.vec({2: 1, -3: sp.sqrt(R(12, 13))})), stab='C5', where='circle |z|^2=12/13 of D'),
    'F*':  dict(u=unit(K.vec({0: 1, 2: sp.I * sp.sqrt(R(5, 3)) / s2, -2: sp.I * sp.sqrt(R(5, 3)) / s2})), stab='D2', where='z=+-i sqrt(5/3) of F'),
    'G*':  dict(u=unit(K.vec({0: 1, 3: sp.I * sp.sqrt(R(20, 23)) / s2, -3: -sp.I * sp.sqrt(R(20, 23)) / s2})), stab='D3', where='z=+-i sqrt(20/23) of G'),
}
# which plane each orbit is an interior point of (for item 5b): name -> (class, chart z)
INTERIOR = {'A*': ('A', R(1, 2)), 'D*': ('D', sp.sqrt(R(12, 13))), 'F*': ('F', sp.I * sp.sqrt(R(5, 3))),
            'G*': ('G', sp.I * sp.sqrt(R(20, 23))), 'xyz': ('C', 1), 'cat': ('E', 1)}
# additional interior occurrences: xyz at z=2/sqrt5 in G; cat at z=sqrt(3/5) in F
INTERIOR_EXTRA = {'xyz@G': ('G', 2 / sp.sqrt(5)), 'cat@F': ('F', sp.sqrt(R(3, 5)))}
