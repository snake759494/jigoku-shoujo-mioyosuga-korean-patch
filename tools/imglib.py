"""Access pictures by key 'fid_path_pic' (path = container indices joined by '-', empty allowed)."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from tim2 import pictures, decode, encode_into, walk
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, 'extract/g')
OUT = os.path.join(ROOT, 'translation/img/out')

def key(fid, path, pic): return '%d_%s_%d' % (fid, '-'.join(map(str, path)), pic)
def parse_key(k):
    f, p, i = k.split('_'); return int(f), tuple(int(x) for x in p.split('-')) if p else (), int(i)

def locate(blob, path, pic):
    for p, o, l in walk(blob):
        if tuple(p) == tuple(path):
            return o, list(pictures(blob[o:o + l]))[pic]
    raise KeyError(path)

def load(k, blob=None):
    fid, path, pic = parse_key(k)
    blob = blob or open(os.path.join(G, '%04d.bin' % fid), 'rb').read()
    o, p = locate(blob, path, pic)
    return decode(blob[o:], p)

def apply(fid, blob):
    """apply every translation/img/out/<fid>_*.png to this archive entry; returns new bytes or None"""
    files = sorted(glob.glob(os.path.join(OUT, '%d_*.png' % fid)))
    if not files: return None
    from PIL import Image
    b = bytearray(blob)
    for fn in files:
        _, path, pic = parse_key(os.path.basename(fn)[:-4])
        o, p = locate(bytes(b), path, pic)
        sub = bytearray(b[o:]); encode_into(sub, p, Image.open(fn)); b[o:] = sub
    return bytes(b)
