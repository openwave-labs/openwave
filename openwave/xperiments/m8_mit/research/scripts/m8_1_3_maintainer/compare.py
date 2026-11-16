"""M8.1.3 adjudication: every scored value of the frozen claims file against the solver rooms' results.json.

Usage: compare.py <claims.json> <copyout dir> [--mutate-claim | --mutate-room]
Each line prints PASS/FAIL with the claim group, the room, the point and both values. The two mutation
modes perturb one claims value and one room value; the run must then show at least one FAIL.
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import sympy as sp
import mpmath as mp
mp.mp.dps = 120

claims_path, co = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else ''
lib.load_claims(claims_path)
C = copy.deepcopy(lib.CLAIMS)
R = {r: json.load(open(os.path.join(co, r, 'results.json'))) for r in ('l6', 'r2', 'cert', 'ca', 'cb', 'rig')}
if mode == '--mutate-claim':
    C['B1']['value']['hexagon']['signature'] = [8, 0, 1]
if mode == '--mutate-room':
    R['r2']['A.4']['line {7/2, -5/2}']['Q'] = '370/247'

out = []; n = {'PASS': 0, 'FAIL': 0}
def rec(gid, room, where, ok, detail):
    t = 'PASS' if ok else 'FAIL'; n[t] += 1
    out.append(f'{t} {gid:3} {room:5} {where:42} {detail}')

def pdf_word_first(s):
    return s.split(';')[0].strip()

def krein_from_word(s):
    """the Krein part of a PDF slow-flow cell: '+', '−', '4+, 1−' -> multiset"""
    if ';' not in s: return None
    k = s.split(';')[1].replace('Krein', '').strip().replace('−', '-')
    if k in ('+', '-'): return k
    parts = [p.strip() for p in k.split(',')]
    ms = []
    for p in parts:
        ms += [p[-1]] * int(p[:-1])
    return sorted(ms)

# ---------------- Block A
A1 = C['A1']['value']
rec('A1', 'l6', 'beta R4', *lib.eq_rational(R['l6']['A.2']['R4']['beta'], A1['R4']))
rec('A1', 'l6', 'beta R5', *lib.eq_rational(R['l6']['A.2']['R5']['beta'], A1['R5']))
rec('A1', 'r2', 'beta R2', *lib.eq_rational(R['r2']['A.2']['R2']['beta'], A1['R2']))
rec('A1', 'l6', 'ratio beta R5/R4 = 21/52 / 28/39', *lib.eq_rational(R['l6']['A.3']['beta_R5/beta_R4'], sp.Rational(A1['R5']) / sp.Rational(A1['R4'])))
rec('A1', 'l6', 'difference beta R4 - R5 (49/156 stated)', *lib.eq_rational(R['l6']['A.3']['beta_R4 - beta_R5'], '49/156'))
rec('A1', '-', 'closed form d(d-r)/(13r) at the room betas', all(lib.eq_rational(b, sp.Rational(d * (d - r), 13 * r))[0] for b, d, r in
    ((R['r2']['A.2']['R2']['beta'], 8, 2), (R['l6']['A.2']['R4']['beta'], 7, 3), (R['l6']['A.2']['R5']['beta'], 7, 4))), 'evaluated, not given to rooms')
A2 = C['A2']['value']
for s in ('R4', 'R5'):
    got = [R['l6']['A.1'][s][k] for k in ('q_6', 'q_4', 'q_2', 'q_0')]
    rec('A2', 'l6', f'(q6,q4,q2,q0) {s}', all(lib.eq_rational(a, b)[0] for a, b in zip(got, A2[s])), f'{got} vs {A2[s]}')
rec('A2', 'r2', 'q7 R2', *lib.eq_rational(R['r2']['A.1']['R2']['q_7'], A2['R2 q7']))
got = [R['r2']['A.1']['R2'][k] for k in ('q_5', 'q_3', 'q_1')]
rec('A2', 'r2', '(q5,q3,q1) R2, script value', all(lib.eq_rational(a, b)[0] for a, b in zip(got, C['A2']['script_value']['R2 (q5, q3, q1)'])), f'{got}')
for pt, v in C['A3']['value'].items():
    g = R['r2']['A.4'][pt]
    rec('A3', 'r2', f'{pt}: critical', g['critical'] in (True, 'True'), str(g['critical']))
    rec('A3', 'r2', f'{pt}: Q', *lib.eq_rational(g['Q'], v['Q']))
    rec('A3', 'r2', f'{pt}: signature', *lib.eq_sig(g['signature'], v['signature']))
for span, val in R['r2']['A.5'].items():
    rec('A4', 'r2', f'Q constant on {span} (identity)', 'on unit vectors' in val and '*' not in val['on unit vectors'], val['on unit vectors'])
A5 = C['A5']['value']
for s in ('R4', 'R5'):
    g = R['l6']['A.6'][s]
    got = [g['sqrt((q-1)/2) percent, rounded up 2 s.f.'] + '%', g['(q-1)/2 percent, rounded up 2 s.f.'] + '%']
    rec('A5', 'l6', f'percents {s}', got == A5[s], f'{got} vs {A5[s]}')
g = R['r2']['A.6']['R2']; got = [g['sqrt((q-1)/2) %'] + '%', g['(q-1)/2 %'] + '%']
rec('A5', 'r2', 'percents R2', got == A5['R2'], f'{got} vs {A5["R2"]}')

# ---------------- Block B, census
def krein_room(entry):
    return sorted(k['Krein'] for k in entry.get('krein', []))
for gid, slot, key in (('B1', 'R4', 'B.1'), ('B2', 'R5', 'B.2')):
    for pt, v in C[gid]['value'].items():
        g = R['l6'][key][slot][pt]
        w = f'{pt} {slot}'
        rec(gid, 'l6', w + ': critical', g['critical'] in (True, 'True'), str(g['critical']))
        if gid == 'B1':
            rec(gid, 'l6', w + ': 924 r6', *lib.eq_rational(g['924 r6hat'], v['924 r6 (Table 2)']))
        rec(gid, 'l6', w + ': signature', *lib.eq_sig(g['signature'], v['signature']))
        if 'k' in v:
            rec(gid, 'l6', w + ': k', *lib.eq_int(g['k'], v['k']))
            rec(gid, 'l6', w + ': r_u', *lib.eq_int(g['r_u'], v['r_u']))
        else:
            rec(gid, 'l6', w + ': k, r_u degenerate -> n0 > 0 (scored with signature)', int(g['signature'][1]) > 0, f"n0={g['signature'][1]}; room k,r_u = {g['k']},{g['r_u']} (C3's)")
        sc = v['script']; kz = [k for k in sc if k.startswith('zero multiplicity')][0]; kp = [k for k in sc if k.startswith('slow polynomial')][0]
        rec(gid, 'l6', w + ': zero multiplicity (script)', *lib.eq_int(g['zero multiplicity'], sc[kz]))
        if 'zero multiplicity of S_u, as (Z) asks (PDF)' in v:
            rec(gid, 'l6', w + ': zero multiplicity as (Z) asks = k + r_u', *lib.eq_int(g['zero multiplicity'], v['zero multiplicity of S_u, as (Z) asks (PDF)']))
        rec(gid, 'l6', w + ': slow polynomial, monic (script)', *lib.eq_poly_monic(g['slow polynomial'], sc[kp]))
        pdfw = v['slow flow (PDF)']
        if pdfw.startswith('not decided'):
            rec(gid, 'l6', w + ': slow flow unscored (degenerate rule)', True, f"room says {g['words']!r}")
        else:
            rec(gid, 'l6', w + ': slow-flow words', g['words'] == pdf_word_first(pdfw), f"{g['words']!r} vs {pdf_word_first(pdfw)!r}")
            kw = krein_from_word(pdfw)
            if kw is not None:
                kr = krein_room(g)
                ok = (kr == sorted(kw)) if isinstance(kw, list) else (set(kr) == {kw})
                rec(gid, 'l6', w + ': Krein signs (PDF)', ok, f'{kr} vs {kw}')
            kscr = [k for k in sc if k.startswith('Krein')]
            if kscr:
                rec(gid, 'l6', w + ': Krein per cluster (script)', krein_room(g) == sorted(sc[kscr[0]]), f'{krein_room(g)} vs {sorted(sc[kscr[0]])}')
for pt in C['B2']['value']:
    a = R['l6']['B.1']['R4'][pt]['slow polynomial']; b = R['l6']['B.2']['R5'][pt]['slow polynomial']
    rec('B2', 'l6', f'{pt}: scaling, R5 monic = R4 monic at 256nu/81', lib.poly_monic(b) == lib.scale_poly(a, sp.Rational(256, 81)), 'the rooms own polynomials')
    rt = R['l6']['B.2']['R5 vs R4 eigenvalue ratios (clause 3)'][pt]
    rec('B2', 'l6', f'{pt}: every paired eigenvalue ratio 9/16', all(p['ratio +xi_R5/+xi_R4 (= ratio of -xi\'s)'] == '9/16' for p in rt['pairs']), f"{len(rt['pairs'])} pairs")

# ---------------- B3
for pt, v in C['B3']['value'].items():
    g = R['r2']['B.3'][pt]
    rec('B3', 'r2', f'{pt}: signature', *lib.eq_sig(g['signature'], v['signature']))
    if 'k' in v:
        rec('B3', 'r2', f'{pt}: k', *lib.eq_int(g['k'], v['k'])); rec('B3', 'r2', f'{pt}: r_u', *lib.eq_int(g['r_u'], v['r_u']))
    else:
        rec('B3', 'r2', f'{pt}: k, r_u degenerate -> n0 > 0', int(g['signature'][1]) > 0, f"n0={g['signature'][1]}")
    sc = v['script']
    rec('B3', 'r2', f'{pt}: zero multiplicity (script)', *lib.eq_int(g['zero multiplicity'], sc['zero multiplicity of S_u']))
    rec('B3', 'r2', f'{pt}: slow polynomial, monic (script)', *lib.eq_poly_monic(g['slow polynomial'], sc['slow polynomial in nu, unit u']))
    atp = v['at the point (PDF)']
    if atp.startswith('not decided'):
        rec('B3', 'r2', f'{pt}: at the point unscored (degenerate rule)', True, f"room says {g['slow-flow words']!r}")
    else:
        word = pdf_word_first(atp)
        ok = g['slow-flow words'] == word or (word in ('elliptic', 'hyperbolic') and g['slow-flow words'].startswith(word))
        rec('B3', 'r2', f'{pt}: words at the point', ok, f"{g['slow-flow words']!r} vs {word!r}")
        if ';' in atp:
            kw = atp.split(';')[1].replace('Krein', '').strip().replace('−', '-')
            ks = {k['Krein sign'] for k in g['Krein signs']}
            rec('B3', 'r2', f'{pt}: Krein at the point', ks == {kw}, f'{sorted(ks)} vs {kw}')
    fam = v['along the family (PDF)']
    if fam is not None:
        mem = g['members']
        if fam == 'hyperbolic':
            ok = all(m['slow-flow words'].startswith('hyperbolic') for m in mem.values())
        else:  # definite: elliptic at all four, every cluster definite
            ok = all(m['slow-flow words'].startswith('elliptic') for m in mem.values()) and all(
                len({k['Krein sign'] for k in m['Krein signs']}) == 1 for m in mem.values())
        rec('B3', 'r2', f'{pt}: along the family = {fam}', ok, '; '.join(f"{k}: {m['slow-flow words']}" for k, m in mem.items()))

# ---------------- B4 and D1
def enc(x): return [str(x[0]), str(x[1])]
for slot in ('R4', 'R5'):
    for o in ('O1', 'O2', 'O3'):
        box = f'{o} {slot}'
        g = R['cert']['B.4'][slot][box]; d = R['cert']['D'][slot][box]
        v4 = C['B4']['value'][o]; v1 = C['D1']['value'][box]
        rec('D1', 'cert', f'{box}: Krawczyk in the given box', d['Krawczyk test on the Section-5 box'] == 'passed', d['Krawczyk test on the Section-5 box'])
        for gid, src in (('D1', d), ('B4', g)):
            rec(gid, 'cert', f'{box}: signature', *lib.eq_sig(src['signature'], (v1 if gid == 'D1' else v4)['signature']))
            rec(gid, 'cert', f'{box}: k', *lib.eq_int(src['k'], v1['k'])); rec(gid, 'cert', f'{box}: r_u', *lib.eq_int(src['r_u'], v1['r_u']))
            rec(gid, 'cert', f'{box}: zero multiplicity', *lib.eq_int(src['zero multiplicity'], v1['zero multiplicity of S_u']))
        signs = d['slow root signs']
        rec('D1', 'cert', f'{box}: slow roots positive, negative', [signs.count('+'), signs.count('-')] == v1['slow roots in nu: positive, negative'] and d['slow roots real and simple'] is True, f"{[signs.count('+'), signs.count('-')]} vs {v1['slow roots in nu: positive, negative']}")
        for gid, e in (('D1', d['924 r6']), ('B4', g['924 r6'])):
            ok, det = lib.inside_digits(enc(e), v4['924 r6 (Table 2)'])
            rec(gid, 'cert', f'{box}: 924 r6 inside Table 2 digits', ok, det)
        sw = C['D1']['script_value']['924 r6 enclosure, the routine\'s, rounded outward to 57 decimals'][box]
        lo, hi = mp.mpf(d['924 r6'][0]), mp.mpf(d['924 r6'][1])
        rec('D1', 'cert', f'{box}: room enclosure inside the script enclosure', mp.mpf(sw[0]) <= lo and hi <= mp.mpf(sw[1]), 'script value')
        wid = mp.mpf(g['924 r6'][1]) - mp.mpf(g['924 r6'][0])
        rec('B4', 'cert', f'{box}: 924 r6 width <= 1e-50 (clause 2)', wid <= mp.mpf('1e-50'), mp.nstr(wid, 3))
        rec('B4', 'cert', f'{box}: slow-flow words', g['slow-flow words'] == v4['slow flow (PDF)'], f"{g['slow-flow words']!r} vs {v4['slow flow (PDF)']!r}")
        if o == 'O2':
            rec('B4', 'cert', f'{box}: contains 28800/121', *lib.contains(enc(g['924 r6']), '28800/121'))
        if o == 'O3':
            rec('B4', 'cert', f'{box}: zero multiplicity as (Z) asks', *lib.eq_int(g['zero multiplicity'], v4['zero multiplicity of S_u, as (Z) asks (PDF)']))
        if slot == 'R5':
            rs = g['eigenvalue ratios R5/R4 (clause 3 pairing)']
            r916 = mp.mpf(9) / 16
            ok = len(rs) == 4 and all(mp.mpf(p['ratio +xi(R5)/+xi(R4) = -xi(R5)/-xi(R4)'][0]) <= r916 <= mp.mpf(p['ratio +xi(R5)/+xi(R4) = -xi(R5)/-xi(R4)'][1]) for p in rs)
            rec('B4', 'cert', f'{o}: scaling, each paired ratio enclosure contains 9/16', ok, f'{len(rs)} pairs')
            # interval overlap reading: each R5 root enclosure meets (81/256)*R4's and no other
            r4 = [[mp.mpf(a), mp.mpf(b)] for a, b in R['cert']['B.4']['R4'][f'{o} R4']['slow roots (nu), sorted']]
            r5 = [[mp.mpf(a), mp.mpf(b)] for a, b in g['slow roots (nu), sorted']]
            f = mp.mpf(81) / 256
            meets = [[(max(a[0], f * b[0]) <= min(a[1], f * b[1])) for b in r4] for a in r5]
            ok = all(sum(row) == 1 and row[i] for i, row in enumerate(meets))
            rec('B4', 'cert', f'{o}: R5 roots meet 81/256 x R4 roots, one each', ok, 'interval overlap')

# ---------------- B5
b5 = R['l6']['B.5']
rec('B5', 'l6', 'HS spin-6 factor = r6, constant 1 (identity)', b5['proportional'] in (True, 'True') and str(b5['constant']) == '1', str(b5['constant']))

# ---------------- C1, C2
for room in ('ca', 'cb'):
    c1 = R[room]['C.1']
    for s in ('R2', 'R4', 'R5'):
        rec('C1', room, f'{s}: s_N', *lib.eq_rational(c1[s]['s_N'], C['C1']['value']['s'][s]))
        lv = c1[s].get('levels where nonzero', c1[s].get('nonzero levels'))
        rec('C1', room, f'{s}: levels with nonzero norm', sorted(int(x) for x in lv) == C['C1']['value']['levels'][s], f'{lv}')
        c2 = R[room]['C.2'][s]
        rec('C2', room, f'{s}: next level', *lib.eq_int(c2['next level'], C['C2']['value']['next level'][s]))
        gk = [k for k in c2 if k.startswith('lambda_1 - lambda_0')][0]
        rec('C2', room, f'{s}: gap', *lib.eq_rational(c2[gk], C['C2']['value']['gap, in units of 1/R^2'][s]))

# ---------------- C3
v = C['C3']['value']["R2's point, the line {5/2, -3/2}"]; g = R['cb']['C.3']['R2']['line {5/2, -3/2}']
rec('C3', 'cb', 'R2 line point: order of Gamma_u', *lib.eq_int(g['|Gamma_u|'], v['order of Gamma_u']))
rec('C3', 'cb', 'R2 line point: dim K\'', *lib.eq_int(g["dim K'"], v["dim K'"]))
rec('C3', 'cb', 'R2 line point: absolutely irreducible', g['absolutely irreducible'] is True and v["Gamma_u on K'"] == 'absolutely irreducible', str(g['absolutely irreducible']))
rec('C3', 'cb', 'R2 line point: n-', *lib.eq_int(g['n_-'], v['n-']))
rec('C3', 'cb', 'R2 line point: gamma_u', *lib.eq_rational(g['gamma_u'].split()[0], v['gamma_u, units R^2']))
v = C['C3']['value']['v1, R4 and R5']
for s in ('R4', 'R5'):
    g = R['cb']['C.3'][s]['v1']
    rec('C3', 'cb', f'v1 {s}: K\' = C v_-3', g["K' is a complex subspace"] is True and g["dim K'"] in (2, '2') and "['-3']" in g["K'"], g["K'"])
    rec('C3', 'cb', f'v1 {s}: weight 4 under the circle', g["K' weight under the circle"] == [4], str(g["K' weight under the circle"]))
    rec('C3', 'cb', f'v1 {s}: gamma_u', *lib.eq_rational(g['gamma_u'].split()[0], v['gamma_u, units R^2'][s]))
    sgn = '-' if sp.Rational(g['gamma_u'].split()[0]) < 0 else '+'
    rec('C3', 'cb', f'v1 {s}: Krein sign of the pair = sign of gamma_u', sgn == v['Krein sign of the pair'][s], f'{sgn} vs {v["Krein sign of the pair"][s]}')
    rec('C3', 'cb', f'v1 {s}: (k, r_u) (clause 5)', [int(g['k']), int(g['r_u'])] == v['k, r_u (p. 18)'], f"{[g['k'], g['r_u']]}")
    zm = int(g["off K': zero multiplicity of S^"])
    roots = [mp.mpf(x) for x in g["off K': slow roots (nu)"]]
    P = lib.poly_monic(g["off K': slow polynomial"])
    simple = sp.Poly(P).sqf_part().degree() == P.degree()
    rec('C3', 'cb', f'v1 {s}: off K\', zero multiplicity 4 = k + r_u', zm == 4 == int(g['k']) + int(g['r_u']), str(zm))
    rec('C3', 'cb', f'v1 {s}: off K\', four simple negative roots', P.degree() == 4 and simple and all(r < 0 for r in roots) and len(sp.real_roots(P)) == 4, f'deg {P.degree()}, simple {simple}')

# ---------------- C4
for slot in ('R2', 'R4', 'R5'):
    for pt, g in R['ca']['C.4'][slot].items():
        rec('C4', 'ca', f'{pt} {slot}: critical, R~0 negative', g['critical'] in (True, 'True') and g['sign'] == '-', f"{g['sign']} {g.get('R0~', g.get('enclosure', ''))}")
sv = C['C4']['script_value']['R~0(u), exact, units V = R = 1']
mapk = {"R2's point, the line {5/2, -3/2}": ('R2', 'line {5/2, -3/2}'), 'coherent v3, R4': ('R4', 'coherent v3'), 'coherent v3, R5': ('R5', 'coherent v3'),
        'octahedron, R4': ('R4', 'octahedron'), 'pentagonal pyramid, R4': ('R4', 'pentagonal pyramid'), 'weight 7/2, R2': ('R2', 'weight 7/2')}
for k, val in sv.items():
    s, pt = mapk[k]
    rec('C4', 'ca', f'{pt} {s}: R~0 exact (script value)', *lib.eq_rational(R['ca']['C.4'][s][pt]['R0~'], val))
rec('C4', 'ca', 'count of tabulated points scored', sum(len(R['ca']['C.4'][s]) for s in ('R2', 'R4', 'R5')) >= 28 + 6, str(sum(len(R['ca']['C.4'][s]) for s in ('R2', 'R4', 'R5'))))

# ---------------- C5
cmap = {'coherent v3, R4': ('R4', 'coherent v3'), 'coherent v3, R5': ('R5', 'coherent v3'), 'octahedron, R4': ('R4', 'octahedron'),
        'pentagonal pyramid, R4': ('R4', 'pentagonal pyramid'), 'weight 7/2, R2': ('R2', 'weight 7/2')}
def cnum(s):
    return complex(sp.N(sp.sympify(str(s).replace('j', '*I') if 'j' in str(s) else str(s)), 30))
for k, v in C['C5']['value'].items():
    s, pt = cmap[k]; g = R['cb']['C.5'][s][pt]
    rec('C5', 'cb', f'{k}: zero multiplicity', *lib.eq_int(g['zero multiplicity'], v['zero multiplicity of S_u']))
    rec('C5', 'cb', f'{k}: k, r_u', [int(g['k']), int(g['r_u'])] == [v['k'], v['r_u']], f"{g['k']},{g['r_u']}")
    for grp in C['C5']['script_value']['c_g per slow group (next_order_check.py E)'][k]:
        sg = complex(grp['s'].replace(' ', '')); cg = complex(grp['c_g'].replace(' ', ''))
        best = min(g['slow groups'], key=lambda x: abs(cnum(x['s_g']) - sg))
        sd = abs(cnum(best['s_g']) - sg) / abs(sg)
        vals = [cnum(c) for c in (best.get('c_g') or best['eigenvalues of C_ab'])]
        rel = max(abs(x - cg) / abs(cg) for x in vals)  # every eigenvalue of a semisimple group must agree
        mult_ok = int(best['multiplicity']) == grp['multiplicity']
        rec('C5', 'cb', f'{k}: c_g at s={grp["s"]}', rel < 1e-9 and sd < 1e-7 and mult_ok, f'rel diff {rel:.1e}, s matched to {sd:.1e}, mult {best["multiplicity"]}')

# ---------------- E1
lg = C['E1']['value']['levels and gap']
names = {0: 'dim1_n0', 1: 'dim2_n1', 2: 'dim3_n2', 3: 'dim4_n3', 4: 'dim5_n4', 5: 'dim6_n5'}
rec('E1', 'rig', 'rigid slots found = n 0..5', set(R['rig']['E.1']) == set(names.values()), str(sorted(R['rig']['E.1'])))
for nn, nm in names.items():
    g = R['rig']['E.1'][nm]; v = lg[str(nn)]
    rec('E1', 'rig', f'n={nn}: levels', [int(g['n']), int(g['next level'])] == v['levels'], f"{[g['n'], g['next level']]}")
    rec('E1', 'rig', f'n={nn}: gap', *lib.eq_int(g['lambda1 - lambda0 (units 1/R^2)'], v['gap, units 1/R^2']))
    ev = R['rig']['E.2']['slots'][nm]['eigenvalues']
    z = ev['0']; ok0 = (z['algebraic'], z['geometric']) == (2 * nn + 2, 2 * nn + 1)
    if nn == 0:
        okw = '2*I*omega' not in ev
    else:
        okw = ev['2*I*omega']['algebraic'] == nn and ev['-2*I*omega']['algebraic'] == nn
    tail = [k for k in ev if 'sqrt' in k]
    okt = len(tail) == 2 and all(ev[k]['algebraic'] == 1 for k in tail) and all('2*c**2*g_4*r**2 + 4*omega**2' in k for k in tail)
    rec('E1', 'rig', f'n={nn}: E0 spectrum (0: 2n+2/2n+1; +-2iw: n; +-i sqrt(4w^2+2g4c^2|v|^2))', ok0 and okw and okt, f"0 {z}, tail {tail}")

print('\n'.join(out))
print(f"\nTOTAL: {n['PASS']} PASS, {n['FAIL']} FAIL" + (f'  [{mode}]' if mode else ''))
