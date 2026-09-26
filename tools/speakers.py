"""Heuristic speaker per script string: text ops are 61 00 (28|38)+hi lo; 0x28 form carries voice args (char, ...),
0x38 form uses the preceding 61 44 <char> voice op. char ids are 1-based into the ELF name list."""
import sys, os, re, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from scr import parse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ['', '御景ゆずき', '神林明日香', '六道るい', '草尾春人', '篠崎歩弓', '石元蓮', '閻魔あい', '曽根アンナ', '不破龍堂', '山童', 'きくり',
         '吉備津嵩泰', '笠間御咲', '松方智彦', '松方信彦', '松方弥生子', '松方美夜', '草尾雅春', '篠崎兵馬', '村上昭夫', '一目連', '骨女', '輪入道', '？？？', '一同']
def rdint(b, i):
    x = b[i]
    if 0xa8 <= x <= 0xc7: return x - 0xa8, i + 1
    if x in (0xc8, 0xd8): return b[i + 1], i + 2
    if x == 0xd9: return 0x100 + b[i + 1], i + 2
    return None, i + 1
def speakers(d):
    h, offs, strs = parse(d); code = d[h[1]:h[2]]
    sp = {}; last = 0; lastvoice = None
    for m in re.finditer(rb'\x61(\x44|\x00[\x28-\x3f])', code, re.S):
        if m.group(1) == b'\x44':
            v, _ = rdint(code, m.end()); lastvoice = v; continue
        a, b = code[m.end() - 1], code[m.end()]
        idx = ((a & 7) << 8) | b
        if idx >= len(strs): continue
        who = None
        if a & 0xf8 == 0x28:
            who, _ = rdint(code, m.end() + 1)
        elif lastvoice is not None:
            who = lastvoice
        sp.setdefault(idx, who); lastvoice = None
    return sp
if __name__ == '__main__':
    out = {}
    for fn in sorted(glob.glob(os.path.join(ROOT, 'extract/scr/*.bin'))):
        d = open(fn, 'rb').read()
        if not parse(d): continue
        out[os.path.basename(fn)[:3]] = {str(k): v for k, v in speakers(d).items()}
    json.dump(out, open(os.path.join(ROOT, 'translation/speakers.json'), 'w'), indent=0)
