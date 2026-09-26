"""Shared helpers for redrawing Japanese text in images as Korean.
Fonts: SeoulHangang L/M/B/EB (main, brush-like serif feel), NanumSquareNeo a..e (rounded gothic UI)."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = {'L': 'SeoulHangangL.ttf', 'M': 'SeoulHangangM.ttf', 'B': 'SeoulHangangB.ttf', 'EB': 'SeoulHangangEB.ttf',
         'nL': 'NanumSquareNeo-aLt.ttf', 'nR': 'NanumSquareNeo-bRg.ttf', 'nB': 'NanumSquareNeo-cBd.ttf',
         'nEB': 'NanumSquareNeo-dEb.ttf', 'nH': 'NanumSquareNeo-eHv.ttf'}

def font(w='EB', size=20):
    return ImageFont.truetype(os.path.join(ROOT, FONTS[w]), size)

def orig(key):
    return Image.open(os.path.join(ROOT, 'extract/img_orig/%s.png' % key)).convert('RGBA')

def save(key, im):
    p = os.path.join(ROOT, 'translation/img/out/%s.png' % key)
    os.makedirs(os.path.dirname(p), exist_ok=True); im.save(p)

def inpaint(im, mask, radius=4):
    """fill masked pixels (mask: L image or bool array) from surroundings; alpha is inpainted too"""
    a = np.array(im.convert('RGBA')); m = (np.array(mask) > 0).astype(np.uint8) * 255
    rgb = cv2.inpaint(np.ascontiguousarray(a[:, :, :3]), m, radius, cv2.INPAINT_TELEA)
    al = cv2.inpaint(np.ascontiguousarray(a[:, :, 3]), m, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(np.dstack([rgb, al]), 'RGBA')

def mask_where(im, fn, box=None, grow=1):
    """mask of pixels where fn(r,g,b,a arrays) is true, limited to box=(x0,y0,x1,y1), dilated by grow px"""
    a = np.array(im.convert('RGBA')).astype(int)
    m = fn(a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3]).astype(np.uint8)
    if box:
        z = np.zeros_like(m); x0, y0, x1, y1 = box; z[y0:y1, x0:x1] = m[y0:y1, x0:x1]; m = z
    if grow: m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return Image.fromarray(m * 255, 'L')

def text_layer(size, text, fnt, fill=(255, 255, 255, 255), xy=None, anchor='mm', vertical=False, spacing=0,
               stroke=0, stroke_fill=(0, 0, 0, 255), glow=0, glow_fill=(255, 255, 255, 255), shadow=None, scale_x=1.0):
    """render text onto a transparent layer of `size`; glow = blur radius of a halo behind the text;
    shadow = (dx, dy, rgba); scale_x squeezes horizontally (e.g. 0.85) to fit narrow boxes."""
    W, H = size
    big = Image.new("RGBA", (W if scale_x == 1.0 else int(W / scale_x) + 1, H), (0, 0, 0, 0))
    cx, cy = xy if xy else (W / 2, H / 2)
    cx = cx / scale_x

    def draw_on(layer, col, st=0, stf=None):
        d = ImageDraw.Draw(layer)
        if not vertical:
            d.text((cx, cy), text, font=fnt, fill=col, anchor=anchor, stroke_width=st, stroke_fill=stf or col)
        else:
            chars = [c for c in text if c != ' ']
            asc = fnt.size + spacing
            y = cy - asc * len(chars) / 2 + asc / 2
            for c in chars:
                d.text((cx, y), c, font=fnt, fill=col, anchor='mm', stroke_width=st, stroke_fill=stf or col); y += asc
    out = Image.new('RGBA', big.size, (0, 0, 0, 0))
    if glow:
        g = Image.new('RGBA', big.size, (0, 0, 0, 0)); draw_on(g, glow_fill, max(1, glow // 2), glow_fill)
        out = Image.alpha_composite(out, g.filter(ImageFilter.GaussianBlur(glow)))
        out = Image.alpha_composite(out, g.filter(ImageFilter.GaussianBlur(glow / 2)))
    if shadow:
        s = Image.new('RGBA', big.size, (0, 0, 0, 0)); dx, dy, col = shadow
        draw_on(s, col, stroke, col); out = Image.alpha_composite(out, s.transform(s.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy)))
    t = Image.new('RGBA', big.size, (0, 0, 0, 0)); draw_on(t, fill, stroke, stroke_fill)
    out = Image.alpha_composite(out, t)
    if scale_x != 1.0: out = out.resize((W, H), Image.LANCZOS)
    return out

def over(im, layer, clip=None):
    """composite layer over im; clip=(x0,y0,x1,y1) keeps only that region of the layer"""
    if clip:
        m = Image.new('L', layer.size, 0); ImageDraw.Draw(m).rectangle(clip, fill=255)
        layer = Image.composite(layer, Image.new('RGBA', layer.size, (0, 0, 0, 0)), m)
    return Image.alpha_composite(im.convert('RGBA'), layer)

def fit_size(text, w='EB', box=(100, 20), start=40, vertical=False):
    """largest font size whose text fits box (w,h)"""
    for s in range(start, 6, -1):
        f = font(w, s)
        if vertical:
            if f.size * len(text.replace(' ', '')) <= box[1] and s <= box[0]: return s
        else:
            l, t, r, b = f.getbbox(text)
            if r - l <= box[0] and b - t <= box[1]: return s
    return 7

def sheet(keys, path, scale=2):
    """before/after preview: original left, new right, on magenta and dark backgrounds"""
    rows = []
    for k in keys:
        a = orig(k); p = os.path.join(ROOT, 'translation/img/out/%s.png' % k)
        b = Image.open(p).convert('RGBA') if os.path.exists(p) else a
        rows.append((a, b))
    W = max(a.width for a, _ in rows) * 2 + 10; H = sum(a.height + 6 for a, _ in rows)
    c = Image.new('RGBA', (W, H), (90, 20, 90, 255)); y = 0
    for a, b in rows:
        c.alpha_composite(a, (0, y)); c.alpha_composite(b, (a.width + 10, y)); y += a.height + 6
    c = c.resize((W * scale, H * scale), Image.NEAREST) if scale != 1 else c
    c.save(path)
