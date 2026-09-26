"""charsel: character select (candles 623..643, backgrounds 621..724) + speaker name plates 700.
run: python tools/img_specs/charsel.py [preview_dir]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from imgdraw import font, orig, save, inpaint, mask_where, text_layer, over, sheet

PREV = sys.argv[1] if len(sys.argv) > 1 else None

# ---------------------------------------------------------------- 700 name plates
NAMES = {0: '미카게 유즈키', 2: '칸바야시 아스카', 4: '리쿠도 루이', 6: '쿠사오 하루토', 8: '시노사키 아유미',
         10: '이시모토 렌', 12: '엔마 아이', 14: '소네 안나', 16: '후와 류도', 18: '야마와로', 20: '키쿠리',
         22: '키비츠 타카야스', 24: '카사마 미사키', 26: '마츠카타 토모히코', 28: '마츠카타 노부히코',
         30: '마츠카타 야요이코', 32: '마츠카타 미야', 34: '쿠사오 마사하루', 36: '시노사키 효마',
         38: '무라카미 아키오', 40: '이치모쿠렌', 42: '호네온나', 44: '와뉴도', 46: '？？？', 48: '일동'}


def plate(text, W, H, size=24, maxtrack=4, gap=12, x0=3, right=177):
    """black serif text, left aligned, letter-spaced like the original (pitch ~37px for kanji)."""
    f = font('B', size)
    words = text.split(' ')
    adv = [[f.getlength(c) for c in w] for w in words]
    n = sum(len(a) for a in adv)
    base = sum(sum(a) for a in adv) + gap * (len(words) - 1)
    avail = right - x0
    track = max(0, min(maxtrack, (avail - base) / max(1, n - 1)))
    width = base + track * (n - 1)
    sx = min(1.0, avail / width)
    big = Image.new('RGBA', (int(W / sx) + 2, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    x = x0 / sx
    for wi, (w, a) in enumerate(zip(words, adv)):
        for c, ad in zip(w, a):
            d.text((x, H / 2 + 1), c, font=f, fill=(0, 0, 0, 255), anchor='lm')
            x += ad + track
        x += gap
    if sx != 1.0:
        big = big.resize((int(W / sx) + 2, H), Image.LANCZOS)  # keep sharpness then squeeze
        big = big.resize((int(big.width * sx), H), Image.LANCZOS).crop((0, 0, W, H))
    else:
        big = big.crop((0, 0, W, H))
    return big


def do_plates():
    keys = []
    for p, name in NAMES.items():
        k = '700_%d_0' % p
        o = orig(k)
        if name == '？？？':
            im = plate('???', o.width, o.height, size=26, maxtrack=26)
        else:
            im = plate(name, o.width, o.height)
        save(k, im); keys.append(k)
    return keys

# ---------------------------------------------------------------- candles
CANDLES = {623: '미카게 유즈키', 627: '쿠사오 하루토', 631: '시노사키 아유미', 635: '칸바야시 아스카',
           639: '이치모쿠렌', 643: '돌아가기'}


TOP = {623: 95, 627: 88, 631: 95, 635: 100, 639: 98, 643: 105}


def ink_mask(im, box):
    a = np.array(im).astype(np.float32)
    lum = a[:, :, :3].mean(2)
    bg = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), 21).astype(np.float32)
    m = ((lum < bg * 0.72) & (a[:, :, 3] > 200)).astype(np.uint8)
    z = np.zeros_like(m); x0, y0, x1, y1 = box; z[y0:y1, x0:x1] = m[y0:y1, x0:x1]
    z = cv2.dilate(z, np.ones((5, 5), np.uint8))
    return z


def candle_body(im):
    """x range and y range of opaque candle body"""
    al = np.array(im)[:, :, 3] > 200
    cols = np.nonzero(al.sum(0) > al.shape[0] * 0.5)[0]
    return cols.min(), cols.max()


def do_candles():
    keys = []
    for fid, name in CANDLES.items():
        for st in range(3):
            k = '%d_%d_0' % (fid, st)
            o = orig(k); W, H = o.size
            bx0, bx1 = candle_body(o)
            cx = (bx0 + bx1) / 2
            m = ink_mask(o, (int(bx0) + 4, TOP[fid], int(bx1) - 3, H - 12))
            ys = np.nonzero(m.any(1))[0]; ty0, ty1 = ys.min(), ys.max()
            clean = inpaint(o, m * 255, 6)
            a = np.array(clean); a[:, :, 3] = np.array(o)[:, :, 3]; clean = Image.fromarray(a, 'RGBA')
            # ink colour: darkest original ink sample
            oa = np.array(o).astype(int); inkpx = oa[:, :, :3][ink_mask(o, (int(bx0) + 4, TOP[fid], int(bx1) - 3, H - 12)) > 0]
            dark = np.percentile(inkpx.mean(1), 5)
            col = tuple(int(v) for v in np.median(inkpx[inkpx.mean(1) <= dark + 12], 0) * 0.7) + (250,)
            chars = name.replace(' ', '')
            lim0, lim1 = TOP[fid] + 4, H - 16
            span = lim1 - lim0
            gapn = 1 if ' ' in name else 0
            pitch = min(25, span / (len(chars) + 0.4 * gapn))
            size = int(min(pitch * 0.92, 25))
            f = font('EB', size)
            lay = Image.new('RGBA', o.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
            total = pitch * (len(chars) + 0.4 * gapn)
            c0 = min(max((ty0 + ty1) / 2, lim0 + total / 2), lim1 - total / 2)
            y = c0 - total / 2 + pitch / 2
            for wi, w in enumerate(name.split(' ')):
                for c in w:
                    d.text((cx, y), c, font=f, fill=col, anchor='mm'); y += pitch
                y += pitch * 0.4
            lay = lay.filter(ImageFilter.GaussianBlur(0.4))
            # keep text inside candle alpha
            la = np.array(lay); la[:, :, 3] = (la[:, :, 3].astype(int) * (np.array(o)[:, :, 3] > 200)).astype(np.uint8)
            res = over(clean, Image.fromarray(la, 'RGBA'))
            save(k, res); keys.append(k)
    return keys

# ---------------------------------------------------------------- backgrounds
BGS = [621, 625, 629, 633, 637, 641, 724]  # 622/626/630/634/638/642 carry no text


def do_bgs():
    keys = []
    for fid in BGS:
        k = '%d__0' % fid
        o = orig(k)
        a = np.array(o).astype(int); r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        mn = np.minimum(np.minimum(r, g), b); mx = np.maximum(np.maximum(r, g), b)
        m = ((mn > 120) & (mx - mn < 45)).astype(np.uint8)
        z = np.zeros_like(m); z[8:272, 66:132] = m[8:272, 66:132]
        z = cv2.dilate(z, np.ones((5, 5), np.uint8))
        clean = inpaint(o, z * 255, 5)
        f = font('B', 21)
        sh = Image.new('RGBA', o.size, (0, 0, 0, 0)); lay = Image.new('RGBA', o.size, (0, 0, 0, 0))
        for L, col in ((sh, (0, 0, 0, 170)), (lay, (250, 250, 248, 255))):
            d = ImageDraw.Draw(L)
            for cx, y0, text in ((114, 30, '캐릭터를'), (86, 106, '선택해 주세요')):
                y = y0
                for c in text:
                    if c == ' ': y += 9; continue
                    d.text((cx, y), c, font=f, fill=col, anchor='mm', stroke_width=1 if L is sh else 0, stroke_fill=col); y += 25
        res = over(clean, sh.filter(ImageFilter.GaussianBlur(2)))
        res = over(res, lay)
        save(k, res); keys.append(k)
    return keys


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nameplate; kp = nameplate.do_plates(); kc = do_candles(); kb = do_bgs()
    if PREV:
        os.makedirs(PREV, exist_ok=True)
        sheet(kp, os.path.join(PREV, 'plates.png'), 3)
        sheet(kc, os.path.join(PREV, 'candles.png'), 1)
        for k in kb[:2]:
            sheet([k], os.path.join(PREV, 'bg_%s.png' % k), 1)
    print(len(kp) + len(kc) + len(kb), 'images')
