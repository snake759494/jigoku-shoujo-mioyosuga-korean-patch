"""Read the built ISO back: every script string must decode to Hangul (via charmap), ASCII or allowed symbols;
report kana/kanji left in scripts and ELF text area."""
import os, sys, json, struct, re
sys.path.insert(0, os.path.dirname(__file__))
from isofs import Iso
from build import Pid
from scr import parse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = {int(v, 16): k for k, v in json.load(open(os.path.join(ROOT, 'translation/charmap.json'), encoding='utf-8')).items()}
def dec(b):
    out = ''; i = 0; bad = 0
    while i < len(b):
        c = b[i]
        if c < 0x80 or 0xa0 <= c < 0xe0: out += chr(c); i += 1; continue
        v = c << 8 | b[i + 1]; i += 2
        if v in REV: out += REV[v]
        elif v < 0x829f or 0x8397 <= v < 0x889f: out += bytes([v >> 8, v & 255]).decode('cp932', 'replace')
        else: out += '■'; bad += 1
    return out, bad
iso = Iso(os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else 'Jigoku_Shoujo_Mioyosuga_KR.iso'))
pid = Pid(iso.read('DATA/PTDALL.PID')); arc = iso.read('DATA/PTD002.PTD'); total = 0
for i, (s, z) in enumerate(pid.entries('PTD002.PTD')):
    if not z: continue
    r = parse(arc[s * 2048:s * 2048 + z])
    if not r: continue
    for k, st in enumerate(r[2]):
        t, bad = dec(st)
        if bad: total += 1; print('script %03d:%d %s' % (i, k, t[:40]))
elf = iso.read('SLPM_552.13'); nb = 0
for m in re.finditer(rb'(?<=\0)((?:[\x81-\x9f\xe0-\xef][\x40-\x7e\x80-\xfc])+)\0', elf[0xf9000:0x106000]):
    t, bad = dec(m.group(1))
    if bad and len(m.group(1)) >= 4: nb += 1; print('elf %X %s' % (0xf9000 + m.start(), t))
print('script strings with leftovers:', total, ' elf:', nb)
