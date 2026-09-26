"""Write translation/elf_ko.tsv into the ELF: in place when it fits, otherwise into slack left by other
translated slots, fixing data pointers and lui/addiu pairs. usage: inject(elf_bytes) -> bytes"""
import os, re, struct, sys
sys.path.insert(0, os.path.dirname(__file__))
from kotext import encode
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VA = 0xff000          # vaddr - file offset
TEXT = (0x1000, 0x84a1c)

def slot(d, off):
    e = d.index(b'\0', off)
    while e < len(d) and d[e] == 0: e += 1
    return e - off - 1          # usable bytes excluding one terminating NUL

def refs(d, off):
    va = off + VA; p = struct.pack('<I', va)
    data = [i for i in range(0x84a80, len(d) - 3, 4) if d[i:i + 4] == p]
    hi = (va + 0x8000) >> 16 & 0xffff; lo = va & 0xffff; pairs = []
    for i in range(TEXT[0], TEXT[1], 4):
        w = struct.unpack_from('<I', d, i)[0]
        if w >> 26 == 0xf and w & 0xffff == hi:
            r = (w >> 16) & 31
            for j in range(i + 4, min(i + 120, TEXT[1]), 4):
                w2 = struct.unpack_from('<I', d, j)[0]
                if w2 >> 26 == 0xf and (w2 >> 16) & 31 == r: break   # register reloaded
                if w2 & 0xffff == lo and (w2 >> 21) & 31 == r and w2 >> 26 in (8, 9): pairs.append((i, j)); break
    return data, pairs

def load():
    rows = []
    for ln in open(os.path.join(ROOT, 'translation/elf_ko.tsv'), encoding='utf-8').read().splitlines():
        if not ln.strip(): continue
        o, t = ln.split('\t', 1); rows.append((int(o, 16), t.replace('\\n', '\n')))
    return rows

def inject(d):
    d = bytearray(d); rows = load(); pend = []; free = []
    for off, t in rows:
        b = encode(t); cap = slot(d, off)
        if len(b) <= cap:
            d[off:off + cap + 1] = b + b'\0' * (cap + 1 - len(b))
            if cap - len(b) >= 4: free.append([off + len(b) + 1, cap - len(b) - 1])
        else: pend.append((off, b, cap))
    log = []
    for off, b, cap in pend:
        data, pairs = refs(d, off)
        if not data and not pairs: raise SystemExit('no refs for %X (%d > %d)' % (off, len(b), cap))
        free.sort(key=lambda x: x[1])
        f = next((f for f in free if f[1] >= len(b)), None)
        if not f: raise SystemExit('no free space for %X' % off)
        new = f[0]; d[new:new + len(b) + 1] = b + b'\0'; f[0] += len(b) + 1; f[1] -= len(b) + 1
        va = new + VA
        for i in data: struct.pack_into('<I', d, i, va)
        for i, j in pairs:
            w = struct.unpack_from('<I', d, i)[0]; struct.pack_into('<I', d, i, (w & 0xffff0000) | ((va + 0x8000) >> 16 & 0xffff))
            w = struct.unpack_from('<I', d, j)[0]; struct.pack_into('<I', d, j, (w & 0xffff0000) | (va & 0xffff))
        d[off:off + cap + 1] = b'\0' * (cap + 1)
        log.append('%X -> %X (%d refs)' % (off, new, len(data) + len(pairs)))
    return bytes(d), log

if __name__ == '__main__':
    d = open(os.path.join(ROOT, 'extract/SLPM_552.13'), 'rb').read()
    out, log = inject(d); print('\n'.join(log)); print('relocated', len(log))
