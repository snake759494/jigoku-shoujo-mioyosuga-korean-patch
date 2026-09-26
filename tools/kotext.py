"""Encode Korean text to the game's SJIS-slot encoding (charmap.json)."""
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM = {k: int(v, 16) for k, v in json.load(open(os.path.join(ROOT, 'translation/charmap.json'), encoding='utf-8')).items()}
FW = {'(': '（', ')': '）'}
def encode(s):
    out = bytearray()
    for c in s:
        if c in CM: out += CM[c].to_bytes(2, 'big')
        elif ord(c) < 0x80: out.append(ord(c))
        else: out += c.encode('cp932')
    return bytes(out)
