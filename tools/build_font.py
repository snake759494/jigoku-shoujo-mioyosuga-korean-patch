"""Build Korean T2FP fonts: KS X 1001 2350 Hangul drawn into kanji slots (SJIS 0x889F..), 
font 0 (normal) <- SeoulHangangB, font 1 (bold) <- SeoulHangangEB, 1-bit (0/15) like the original; weights matched by stroke width (area/skeleton)."""
import sys, os, json, struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from t2fp import Font
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE = {0: ('SeoulHangangB.ttf', 21, 100), 1: ('SeoulHangangEB.ttf', 21, 84)}
CY = 10.0          # vertical centre of original kanji ink box (rows 0..19)

def hangul2350():
    out = []
    for hi in range(0xb0, 0xc9):
        for lo in range(0xa1, 0xff):
            out.append(bytes([hi, lo]).decode('euc-kr'))
    return out

def sjis_slots(font):
    """kanji codes usable as Hangul slots, in table order"""
    return [c for c in font.codes if c >= 0x889f and c < 0xeaa5]

def charmap():
    f = Font(os.path.join(ROOT, 'extract/font/0.t2fp'))
    slots = sjis_slots(f); hs = hangul2350()
    assert len(slots) >= len(hs)
    return {h: slots[i] for i, h in enumerate(hs)}

def render(ch, ttf, size, th, w=24, h=24):
    ft = ImageFont.truetype(ttf, size)
    im = Image.new('L', (w * 2, h * 2))
    ImageDraw.Draw(im).text((w, h), ch, font=ft, fill=255, anchor='mm')
    a = np.array(im)
    ys, xs = np.nonzero(a >= th)       # ink box (for centring) from the solid part
    if len(xs) == 0: return np.zeros((h, w), int)
    # centre ink box horizontally in cell, vertically on CY
    cx = (xs.min() + xs.max()) / 2; cy = (ys.min() + ys.max()) / 2
    dx = int(round(w / 2 - 0.5 - cx)) + w; dy = int(round(CY - cy)) + h
    out = np.zeros((h, w), int)
    for y, x in zip(ys, xs):
        X, Y = x + dx - w, y + dy - h
        if 0 <= X < w and 0 <= Y < h: out[Y, X] = 15   # 1-bit like the original font
    return out

def build(k, out_path):
    f = Font(os.path.join(ROOT, 'extract/font/%d.t2fp' % k))
    ttf, size, th = STYLE[k]; ttf = os.path.join(ROOT, ttf)
    for ch, code in charmap().items():
        f.put(f.idx[code], render(ch, ttf, size, th).tolist())
    f.save(out_path)

if __name__ == '__main__':
    os.makedirs(os.path.join(ROOT, 'extract/font_ko'), exist_ok=True)
    cm = charmap()
    json.dump({h: '%04X' % c for h, c in cm.items()}, open(os.path.join(ROOT, 'translation/charmap.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    for k in (0, 1): build(k, os.path.join(ROOT, 'extract/font_ko/%d.t2fp' % k))
    print('ok', len(cm))
