"""build step: replace PTD000 entries that have outputs in translation/img/out"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
import imglib
S = 2048
def build(arc, ents):
    fids = sorted({int(os.path.basename(f).split('_')[0]) for f in glob.glob(os.path.join(imglib.OUT, '*.png'))})
    repl = {}
    for fid in fids:
        s, z = ents[fid]; new = imglib.apply(fid, arc[s * S:s * S + z])
        if new: assert len(new) == z; repl[fid] = new
    return repl
