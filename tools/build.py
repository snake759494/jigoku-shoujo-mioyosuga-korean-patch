"""Build the Korean ISO: copy the original, then replace SLPM_552.13, DATA/PTDALL.PID, PTD002 (scripts),
PTD003 (fonts) and PTD000 (images, when translation/img/ has outputs).
usage: python tools/build.py [--out NAME.iso]"""
import os, sys, struct, shutil, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
from scr import parse, build as build_scr
from kotext import encode
from inject_elf import inject
from isofs import Iso
import build_font

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'Jigoku Shoujo Mioyosuga (Japan).iso')
S = 2048

def load_tsv(p):
    d = {}
    for ln in open(p, encoding='utf-8').read().split('\n'):
        if ln.strip():
            i, _, t = ln.partition('\t')
            if i.isdigit(): d[int(i)] = t.replace('\\n', '\n')
    return d

class Pid:
    def __init__(self, data):
        self.d = bytearray(data); n = struct.unpack_from('<I', self.d, 0)[0]
        self.counts = struct.unpack_from('<%dI' % n, self.d, 4)
        self.names = [bytes(self.d[0x34 + 16 * i:0x34 + 16 * i + 12]).split(b'\0')[0].decode() for i in range(n)]
    def base(self, arc):
        k = self.names.index(arc); return 0xf4 + 8 * sum(self.counts[:k]), self.counts[k]
    def entries(self, arc):
        b, n = self.base(arc); return [list(struct.unpack_from('<II', self.d, b + 8 * i)) for i in range(n)]
    def set(self, arc, ents):
        b, n = self.base(arc)
        for i, (s, z) in enumerate(ents): struct.pack_into('<II', self.d, b + 8 * i, s, z)

def rebuild_archive(orig, ents, repl):
    """repl: {index: bytes}. Re-pack sequentially (sector aligned), keep entry order."""
    out = bytearray(); new = []
    order = sorted(range(len(ents)), key=lambda i: (ents[i][0], i))
    placed = {}
    for i in order:
        s, z = ents[i]
        if z == 0: new.append((i, [s, 0])); continue
        data = repl.get(i, orig[s * S:s * S + z])
        key = (s, z) if i not in repl else None
        if key and key in placed: new.append((i, [placed[key], len(data)])); continue
        st = len(out) // S; out += data; out += b'\0' * ((-len(out)) % S)
        if key: placed[key] = st
        new.append((i, [st, len(data)]))
    res = [None] * len(ents)
    for i, e in new: res[i] = e
    return bytes(out), res

def scripts(iso, pid):
    arc = iso.read('DATA/PTD002.PTD'); ents = pid.entries('PTD002.PTD'); repl = {}; nko = 0
    for i, (s, z) in enumerate(ents):
        if not z: continue
        d = arc[s * S:s * S + z]; r = parse(d)
        p = os.path.join(ROOT, 'translation/ko/%03d.tsv' % i)
        if not r or not os.path.exists(p): continue
        ko = load_tsv(p); strs = []
        for k, orig in enumerate(r[2]):
            try: strs.append(encode(ko[k]) if k in ko else orig)
            except UnicodeEncodeError as e:
                print('  %03d:%d unencodable %r -> kept original' % (i, k, e.object[e.start])); strs.append(orig)
        repl[i] = build_scr(d, strs); nko += 1
    data, new = rebuild_archive(arc, ents, repl)
    pid.set('PTD002.PTD', new)
    iso.replace('DATA/PTD002.PTD', data)
    print('scripts: %d files translated, PTD002 %d -> %d bytes' % (nko, len(arc), len(data)))

def fonts(iso, pid):
    arc = bytearray(iso.read('DATA/PTD003.PTD')); ents = pid.entries('PTD003.PTD')
    tmp = os.path.join(ROOT, 'extract/font_ko')
    os.makedirs(tmp, exist_ok=True)
    for k, (s, z) in enumerate(ents):
        p = os.path.join(tmp, '%d.t2fp' % k)
        build_font.build(k, p)
        f = open(p, 'rb').read(); assert len(f) == z
        arc[s * S:s * S + z] = f
    iso.replace('DATA/PTD003.PTD', bytes(arc)); print('fonts: ok')

def images(iso, pid):
    try:
        import patch_images
    except ImportError:
        return
    arc = iso.read('DATA/PTD000.PTD'); ents = pid.entries('PTD000.PTD')
    repl = patch_images.build(arc, ents)
    if not repl: return
    data, new = rebuild_archive(arc, ents, repl)
    pid.set('PTD000.PTD', new); iso.replace('DATA/PTD000.PTD', data)
    print('images: %d archive entries replaced' % len(repl))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='Jigoku_Shoujo_Mioyosuga_KR.iso')
    a = ap.parse_args(); out = os.path.join(ROOT, a.out)
    shutil.copyfile(SRC, out); iso = Iso(out)
    pid = Pid(iso.read('DATA/PTDALL.PID'))
    elf, log = inject(iso.read('SLPM_552.13')); iso.replace('SLPM_552.13', elf)
    print('elf: %d relocated' % len(log))
    scripts(iso, pid); fonts(iso, pid); images(iso, pid)
    for p in sorted(glob.glob(os.path.join(ROOT, 'movie/enc/*.PSS'))):   # 자막 입힌 동영상 (tools/movie_sub.py)
        iso.replace('DATA/' + os.path.basename(p), open(p, 'rb').read()); print('movie:', os.path.basename(p))
    iso.replace('DATA/PTDALL.PID', bytes(pid.d)); iso.close()
    print('->', out)

if __name__ == '__main__':
    main()
