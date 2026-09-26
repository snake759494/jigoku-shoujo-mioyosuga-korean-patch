"""Dump script strings to translation/jp/NNN.tsv (id<TAB>text, newline written as \\n)."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
from scr import parse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for fn in sorted(glob.glob(os.path.join(ROOT, 'extract/scr/*.bin'))):
    r = parse(open(fn, 'rb').read())
    if not r: continue
    n = os.path.basename(fn)[:3]
    with open(os.path.join(ROOT, 'translation/jp/%s.tsv' % n), 'w', encoding='utf-8', newline='\n') as o:
        for i, s in enumerate(r[2]):
            o.write('%d\t%s\n' % (i, s.decode('cp932').replace('\n', '\\n')))
