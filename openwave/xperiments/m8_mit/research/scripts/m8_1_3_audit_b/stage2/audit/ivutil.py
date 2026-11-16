"""Interval helpers: exact endpoints of mpmath.iv intervals and outward-rounded decimal strings."""
from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as Fr
from mpmath import mp, iv

getcontext().prec = 400


def ends(x):
    """exact endpoints (mp.mpf) of an iv interval"""
    lo, hi = x._mpi_
    return mp.make_mpf(lo), mp.make_mpf(hi)


def mpf_to_fraction(v):
    sign, man, exp, bc = mp.mpf(v)._mpf_
    if man == 0:
        assert exp == 0, 'special value'
        return Fr(0)
    f = Fr(man) * Fr(2) ** exp if exp >= 0 else Fr(man, 2 ** (-exp))
    return -f if sign else f


def _dec(fr, sig, rounding):
    if fr == 0:
        return '0'
    d = Decimal(fr.numerator) / Decimal(fr.denominator)
    e = d.adjusted()
    q = Decimal(1).scaleb(e - sig + 1)
    return str(d.quantize(q, rounding=rounding))


def outward(x, sig=55):
    """[lower, upper] decimal strings with sig significant digits, rounded outward"""
    lo, hi = ends(x)
    return [_dec(mpf_to_fraction(lo), sig, ROUND_FLOOR), _dec(mpf_to_fraction(hi), sig, ROUND_CEILING)]


def width(x):
    lo, hi = ends(x)
    return hi - lo


def contains_zero(x):
    lo, hi = ends(x)
    return lo <= 0 <= hi


def ivs(s):
    """thin enclosure of a decimal string"""
    return iv.mpf(s)
