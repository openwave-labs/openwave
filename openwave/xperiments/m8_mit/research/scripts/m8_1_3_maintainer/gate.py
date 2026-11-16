"""Leak gate for M8.1.3's packet: no claims-file value and no withheld term in a packet file.
Usage: gate.py <claims.json> <terms.txt> <private_terms.txt> [--selftest] <files...>"""
import json, re, sys
claims, terms_f, priv_f = sys.argv[1:4]
args = sys.argv[4:]; selftest = '--selftest' in args; files = [a for a in args if a != '--selftest']
d = json.load(open(claims))
vals = []
def walk(x, key=''):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ('matching_rules', 'about', 'serialization', 'what', 'on_record', 'source', 'rules', 'grade', 'status', 'id', 'match', 'scripts', 'pdf', 'certificate_file'): continue
            walk(v, k)
    elif isinstance(x, list):
        for v in x: walk(v, key)
    elif isinstance(x, str): vals.append(x)
    elif isinstance(x, (int, float)) and not isinstance(x, bool): vals.append(str(x))
walk(d)
blob = '\n'.join(vals)
fracs = set(f for f in re.findall(r'\d+/\d+', blob) if f not in ('1/2','3/2','5/2','7/2'))
decs = set(re.findall(r'\d+\.\d{4,}', blob))
ints = set(re.findall(r'(?<![\d.])\d{3,}(?![\d.])', blob))
terms = [t.strip() for f in (terms_f, priv_f) for t in open(f) if t.strip()]
print(f'gate set: {len(vals)} value strings; {len(fracs)} fractions, {len(decs)} decimals (4+ places), {len(ints)} integers of 3+ digits, {len(terms)} terms')
def scan(t):
    hits = set()
    for x in fracs:
        if re.search(r'(?<![\d/])' + re.escape(x) + r'(?![\d/])', t): hits.add(('frac', x))
    for x in decs:
        if x[:12] in t: hits.add(('dec', x[:12]))
    for x in ints:
        if re.search(r'(?<![\d.])' + x + r'(?![\d.])', t): hits.add(('int', x))
    for x in terms:
        if re.search(r'(?<![A-Za-z0-9])' + re.escape(x) + r'(?![A-Za-z0-9])', t): hits.add(('term', x))
    return sorted(hits)
for f in files: print(f.split('/')[-1], scan(open(f, encoding='utf-8').read()))
if selftest:
    base = open(files[0], encoding='utf-8').read(); plant = sorted(fracs, key=len)[-1] + ' ' + sorted(ints, key=len)[-1] + ' ' + terms[0]
    n0, n1 = len(scan(base)), len(scan(base + '\n' + plant))
    ok = n1 >= n0 + 3
    print('selftest:', 'PASS (planted fraction, integer and term flagged)' if ok else 'FAIL'); sys.exit(0 if ok else 1)
