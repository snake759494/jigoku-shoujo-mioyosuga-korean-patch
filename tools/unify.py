"""Consistency pass: identical Japanese lines (>= MINLEN chars, or any line in credit files) get the
majority Korean translation across all files (tie -> earliest file). usage: python tools/unify.py [--dry]"""
import os, glob, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINLEN = 10
CREDITS = {'014', '015', '016', '110', '111', '112', '211', '212', '213', '310', '311', '312', '414'}
def load(p):
    rows = []
    for l in open(p, encoding='utf-8').read().split('\n'):
        if '\t' in l:
            i, t = l.split('\t', 1)
            if i.isdigit(): rows.append((int(i), t))
    return rows
files = sorted(os.path.basename(f)[:3] for f in glob.glob(os.path.join(ROOT, 'translation/ko/*.tsv')))
jp = {n: dict(load(os.path.join(ROOT, 'translation/jp/%s.tsv' % n))) for n in files}
ko = {n: load(os.path.join(ROOT, 'translation/ko/%s.tsv' % n)) for n in files}
votes = collections.defaultdict(collections.Counter); first = {}
for n in files:
    for i, t in ko[n]:
        j = jp[n][i]; votes[j][t] += 1; first.setdefault((j, t), n)
changed = 0
for n in files:
    out = []
    for i, t in ko[n]:
        j = jp[n][i]; c = votes[j]
        if len(c) > 1 and (len(j.replace('\n', '')) >= MINLEN or n in CREDITS):
            best = max(c, key=lambda k: (c[k], -int(first[(j, k)])))
            if best != t: changed += 1; t = best
        out.append('%d\t%s' % (i, t))
    if '--dry' not in sys.argv:
        open(os.path.join(ROOT, 'translation/ko/%s.tsv' % n), 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')
print('changed lines:', changed)
