"""Validate translation/ko/NNN.tsv against jp: ids, trailing newline, width (34 half-cols), line count, charset.
usage: python tools/check_text.py [NNN ...]   (no args = all existing ko files)"""
import sys, os, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM = json.load(open(os.path.join(ROOT, 'translation/charmap.json'), encoding='utf-8'))
MAXW, MAXL = 34, 4

def load(path):
    d = {}
    for ln in open(path, encoding='utf-8').read().split('\n'):
        if not ln.strip(): continue
        i, _, t = ln.partition('\t'); d[int(i)] = t.replace('\\n', '\n')
    return d

def char_ok(c):
    if c in CM or c == '\n' or 0x20 <= ord(c) < 0x7f: return True
    try: b = c.encode('cp932')
    except UnicodeEncodeError: return False
    if len(b) != 2: return False
    v = b[0] << 8 | b[1]
    return v < 0x829f or 0x8397 <= v < 0x889f  # symbols only, no kana/kanji

def width(s): return sum(1 if ord(c) < 0x80 else 2 for c in s)

def check(n):
    jp = load(os.path.join(ROOT, 'translation/jp/%s.tsv' % n))
    kp = os.path.join(ROOT, 'translation/ko/%s.tsv' % n)
    if not os.path.exists(kp): return ['missing file']
    ko = load(kp); errs = []
    for i, j in jp.items():
        if i not in ko: errs.append('%d: missing' % i); continue
        k = ko[i]
        if j.endswith('\n') != k.endswith('\n'): errs.append('%d: trailing \\n mismatch' % i)
        lines = k.rstrip('\n').split('\n')
        jl = len(j.rstrip('\n').split('\n'))
        if len(lines) > max(MAXL, jl): errs.append('%d: %d lines' % (i, len(lines)))
        jw = max(width(x) for x in j.rstrip('\n').split('\n'))
        for x in lines:
            if width(x) > max(MAXW, jw): errs.append('%d: width %d > %d: %s' % (i, width(x), max(MAXW, jw), x))
        bad = sorted(set(c for c in k if not char_ok(c)))
        if bad: errs.append('%d: bad chars %s' % (i, ''.join(bad)))
    for i in ko:
        if i not in jp: errs.append('%d: extra id' % i)
    return errs

if __name__ == '__main__':
    ns = sys.argv[1:] or sorted(os.path.basename(f)[:3] for f in glob.glob(os.path.join(ROOT, 'translation/ko/*.tsv')))
    total = 0
    for n in ns:
        e = check(n); total += len(e)
        print('%s: %d errors' % (n, len(e)))
        for x in e[:40]: print('  ' + x)
    sys.exit(1 if total else 0)
