"""options: 660-667, 722, 723, 728-739 (gallery plaques, option panels, bonus buttons, mini-theater titles, UI backgrounds).
run: python tools/img_specs/options.py [preview_dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from imgdraw import font, orig, save, inpaint, mask_where, text_layer, over, fit_size, sheet

# ---------------------------------------------------------------- engine
def efill(im, mask, src=None, P=4, win=60):
    """exemplar (patch copy) fill of masked pixels; keeps periodic hex texture sharp.
    src: bool array of pixels allowed as source (default: everything not masked)."""
    a = np.array(im.convert('RGBA')).astype(np.float32)
    m = np.array(mask) > 0
    H, W = m.shape
    if src is None: src = np.ones_like(m)
    known = ~m
    okc = (known & src).astype(np.uint8)
    # valid source centres: whole patch known & allowed
    k = np.ones((2 * P + 1, 2 * P + 1), np.uint8)
    full = cv2.erode(okc, k, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    full[:P, :] = full[-P:, :] = False; full[:, :P] = full[:, -P:] = False
    cy, cx = np.nonzero(full)
    if len(cy) == 0: return inpaint(im, mask)
    pad = np.pad(a, ((P, P), (P, P), (0, 0)))
    offs = [(dy, dx) for dy in range(-P, P + 1) for dx in range(-P, P + 1)]
    patches = np.stack([a[cy + dy, cx + dx] for dy, dx in offs], 1)  # N,K,4
    kn = known.copy()
    while True:
        todo = ~kn
        if not todo.any(): break
        cnt = cv2.filter2D(kn.astype(np.float32), -1, np.ones((2 * P + 1, 2 * P + 1), np.float32), borderType=cv2.BORDER_CONSTANT)
        cnt[~todo] = -1
        y, x = np.unravel_index(np.argmax(cnt), cnt.shape)
        sel = (np.abs(cy - y) <= win) & (np.abs(cx - x) <= win * 3)
        if not sel.any(): sel = np.ones_like(cy, bool)
        tp = np.stack([pad[y + P + dy, x + P + dx] for dy, dx in offs], 0)  # K,4
        tk = np.array([0 <= y + dy < H and 0 <= x + dx < W and kn[y + dy, x + dx] for dy, dx in offs])
        cand = patches[sel]
        d = ((cand[:, tk, :3] - tp[None, tk, :3]) ** 2).sum((1, 2))
        j = np.argmin(d); best = cand[j]
        for i, (dy, dx) in enumerate(offs):
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and not kn[yy, xx]:
                a[yy, xx] = best[i]; kn[yy, xx] = True
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA')

def wmask(im, box, thr=190, sat=45, grow=2):
    """bright, low-saturation (white/grey) text pixels in box, grown to cover the dark outline"""
    return mask_where(im, lambda r, g, b, a: (np.minimum(np.minimum(r, g), b) >= thr) &
                      ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) < sat) & (a > 100), box, grow)

def mor(*ms):
    out = np.zeros_like(np.array(ms[0]))
    for m in ms: out = np.maximum(out, np.array(m))
    return Image.fromarray(out, 'L')

def boxsrc(shape, box):
    s = np.zeros(shape, bool); x0, y0, x1, y1 = box; s[y0:y1, x0:x1] = True; return s

def put(im, text, c, size, w='B', fill=(255, 255, 255, 255), stroke=1, sf=(20, 20, 20, 255), sx=1.0, glow=0, gf=None, shadow=None):
    L = text_layer(im.size, text, font(w, size), fill=fill, xy=c, stroke=stroke, stroke_fill=sf, scale_x=sx,
                   glow=glow, glow_fill=gf or (255, 255, 255, 255), shadow=shadow)
    return over(im, L)

def fitx(text, w, size, maxw):
    """horizontal squeeze factor so text fits maxw at given size"""
    l, t, r, b = font(w, size).getbbox(text)
    return min(1.0, maxw / max(1, r - l))

# hex tile pattern of the option panels: lattice (28,0),(14,8) -> cell centres (ox+14i, oy+8j), i+j even
def cell_index(H, W, ox, oy):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # nearest lattice point among candidates around (x,y)
    i0 = np.floor((xx - ox) / 14); j0 = np.floor((yy - oy) / 8)
    bi = np.zeros((H, W), np.int32); bj = np.zeros((H, W), np.int32); bd = np.full((H, W), 1e9, np.float32)
    for di in (-1, 0, 1, 2):
        for dj in (-1, 0, 1, 2):
            i = i0 + di; j = j0 + dj
            ok = ((i + j) % 2 == 0)
            # hex metric: scale so Voronoi of lattice is regular (cells 28 wide, rows 16 -> y distances * 28/(16*sqrt3)*... )
            d = ((xx - (ox + 14 * i)) ** 2 + ((yy - (oy + 8 * j)) * 1.0) ** 2)
            d = np.where(ok, d, 1e9)
            upd = d < bd
            bd = np.where(upd, d, bd); bi = np.where(upd, i, bi).astype(np.int32); bj = np.where(upd, j, bj).astype(np.int32)
    return bi, bj

def hex_origin(a, ok):
    """find lattice origin via 180-degree point symmetry of each cell"""
    H, W = ok.shape; g = a[:, :, :3].astype(np.float32)
    best = (1e9, 0, 0)
    for oy in range(0, 16):
        for ox in range(0, 28):
            if ((ox // 14) + (oy // 8)) % 2: continue  # (ox,oy) and (ox+14,oy+8) are the same lattice
            bi, bj = cell_index(H, W, ox, oy)
            cx = ox + 14 * bi; cy = oy + 8 * bj
            yy, xx = np.mgrid[0:H, 0:W]
            mx = 2 * cx - xx; my = 2 * cy - yy
            v = ok & (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
            mxv = np.clip(mx, 0, W - 1); myv = np.clip(my, 0, H - 1)
            v &= ok[myv, mxv]
            if v.sum() < 100: continue
            s = np.abs(g - g[myv, mxv]).sum(2)[v].mean()
            if s < best[0]: best = (s, ox, oy)
    return best[1], best[2]

def hexfill(im, mask, region, origin=None, maxs=8, protect=None):
    """replace every hex cell touching `mask` by a whole clean cell copied along the lattice.
    region: bool array where the hex pattern lives (sources must lie fully inside, clean)."""
    a = np.array(im.convert('RGBA')); H, W = region.shape
    m = np.array(mask) > 0
    ok = region & ~m
    if origin is None: origin = hex_origin(a, ok)
    ox, oy = origin
    bi, bj = cell_index(H, W, ox, oy)
    cid = (bi + 100) * 1000 + (bj + 100)
    ids = np.unique(cid[m & region])
    out = a.copy()
    frame = protect if protect is not None else np.zeros_like(region)
    used = {}
    for c in ids:
        i, j = c // 1000 - 100, c % 1000 - 100
        ys, xs = np.nonzero(cid == c)
        sel = region[ys, xs] & ~frame[ys, xs]
        ys, xs = ys[sel], xs[sel]
        if len(ys) == 0: continue
        keep = ok[ys, xs]
        bestd, best = 1e18, None
        for dj in range(-maxs, maxs + 1):
            for di in range(-3 * maxs, 3 * maxs + 1):
                if (di + dj) % 2 or (di == 0 and dj == 0): continue
                sy, sx = ys + 8 * dj, xs + 14 * di
                if sy.min() < 0 or sx.min() < 0 or sy.max() >= H or sx.max() >= W: continue
                if not ok[sy, sx].all(): continue
                if keep.sum() >= 4: d = np.abs(a[sy, sx, :3].astype(int) - a[ys, xs, :3].astype(int))[keep].mean()
                else: d = 0
                d += 0.3 * (abs(di) + 2 * abs(dj)) + 15 * used.get((i + di, j + dj), 0)
                if d < bestd: bestd, best = d, (di, dj)
        if best is None: continue
        di, dj = best; used[(i + di, j + dj)] = used.get((i + di, j + dj), 0) + 1
        out[ys, xs] = a[ys + 8 * dj, xs + 14 * di]
    return Image.fromarray(out, 'RGBA'), origin

PV = sys.argv[1] if len(sys.argv) > 1 else None
def preview(keys, name):
    if PV: os.makedirs(PV, exist_ok=True); sheet(keys, os.path.join(PV, name))

# ---------------------------------------------------------------- 661 option menu items
M661 = ['게임 설정', '스킵 모드 설정', '사운드 설정', '음성 전환', '조작 방법', '설정 초기화']
MENU_SIZE = 22
def menu_text(im, text, cy, col, cx=176, size=MENU_SIZE):
    sx = fitx(text, 'B', size, 215)
    return put(im, text, (cx, cy), size, 'B', fill=col, stroke=2, sf=(8, 8, 8, 255), sx=sx)

def do661():
    # clean band of the selected state from 661_11 (its text does not touch the flame ornament)
    don = orig('661_11_0'); W, H = don.size
    reg = boxsrc((H, W), (22, 17, 340, 44)) & ~boxsrc((H, W), (60, 8, 128, 52))
    dclean, _ = hexfill(don, wmask(don, (126, 8, 240, 52), thr=190, grow=2), reg)
    band = boxsrc((H, W), (40, 17, 322, 44))
    for n, t in enumerate(M661):
        for st in range(3):
            k = '661_%d_0' % (n * 3 + st); im = orig(k); W, H = im.size
            if st == 2:
                a = np.array(im); a[band] = np.array(dclean)[band]; e = Image.fromarray(a, 'RGBA')
                cy = 31; col = (255, 255, 255, 255)
            else:
                thr = 190 if st == 0 else 120
                bx = (20, 17, 340, 44) if st == 0 else (18, 11, 334, 38)
                e, _ = hexfill(im, wmask(im, (40, 6, W - 40, H - 6), thr=thr, grow=2), boxsrc((H, W), bx))
                cy = 31 if st == 0 else 24.5; col = (255, 255, 255, 255) if st == 0 else (190, 190, 190, 255)
            save(k, menu_text(e, t, cy, col))
    preview(['661_%d_0' % i for i in range(18)], '661.png')

# ---------------------------------------------------------------- 662-665 slider / toggle panels
def panel(im, erase, texts, protect=(), frame_box=None, thr=150, w='B', col=(255, 255, 255, 255)):
    """erase: boxes whose white text is removed; texts: (text, (cx,cy), size[, maxw]); protect: boxes left untouched"""
    W, H = im.size; a = np.array(im).astype(int)
    m = mor(*[wmask(im, b, thr=thr, grow=2) for b in erase])
    pm = np.zeros((H, W), bool)
    for b in protect: pm |= boxsrc((H, W), b)
    mm = (np.array(m) > 0) & ~pm
    fb = frame_box or (30, 6, W - 30, H - 6)
    region = boxsrc((H, W), fb) & ~pm & ~((a[:, :, :3].max(2) < 30) & ~mm)
    e, _ = hexfill(im, Image.fromarray(mm.astype(np.uint8) * 255), region)
    rest = (np.array(e)[:, :, :3].astype(int) == a[:, :, :3]).all(2) & mm   # cells that could not be replaced
    if rest.any(): e = efill(e, Image.fromarray(rest.astype(np.uint8) * 255), src=region & ~mm)
    for t in texts:
        text, c, size = t[:3]; maxw = t[3] if len(t) > 3 else 999
        e = put(e, text, c, size, w, fill=col, stroke=2, sf=(8, 8, 8, 255), sx=fitx(text, w, size, maxw))
    return e

def do662():
    T, L = 19, 18
    save('662_0_0', panel(orig('662_0_0'), [(100, 8, 400, 70)], [('스킵 모드', (248, 25), T), ('읽은 문장', (143, 56), L, 70), ('강제', (249, 56), L), ('자동', (352, 56), L)]))
    save('662_1_0', panel(orig('662_1_0'), [(100, 8, 400, 38), (45, 38, 84, 70), (416, 38, 455, 70)], [('메시지 창', (248, 23), T), ('옅게', (66, 53), 16, 28), ('짙게', (432, 53), 16, 28)], protect=[(84, 38, 416, 67)]))
    save('662_2_0', panel(orig('662_2_0'), [(100, 8, 400, 38), (45, 38, 84, 70), (416, 38, 455, 70)], [('글자 표시 속도', (248, 22), T), ('느림', (65, 54), 16, 28), ('빠름', (431, 54), 16, 28)], protect=[(84, 38, 416, 67)]))
    save('662_3_0', panel(orig('662_3_0'), [(200, 8, 300, 40)], [('진동', (248, 26), T)], protect=[(100, 42, 400, 80)]))
    for n, t in enumerate(['읽은 문장 모드', '강제 모드', '자동 모드']):
        k = '663_%d_0' % n
        save(k, panel(orig(k), [(100, 12, 400, 48), (45, 50, 84, 82), (416, 50, 455, 82), (150, 80, 350, 112)],
                      [(t, (253, 29), T), ('느림', (65, 66), 16, 28), ('빠름', (432, 66), 16, 28), ('해제', (194, 97), L), ('계속', (310, 97), L)],
                      protect=[(84, 50, 416, 79)]))
    for n, t in enumerate([None, '효과음', '보이스']):
        k = '664_%d_0' % n
        er = [(45, 44, 84, 76), (416, 44, 460, 76)] + ([(150, 10, 350, 42)] if t else [])
        tx = [('작게', (66, 59), 16, 28), ('크게', (436, 58), 16, 30)] + ([(t, (250, 27), T)] if t else [])
        save(k, panel(orig(k), er, tx, protect=[(84, 44, 416, 73)]))
    preview(['662_%d_0' % i for i in range(4)] + ['663_%d_0' % i for i in range(3)] + ['664_%d_0' % i for i in range(3)], '662.png')

N665 = ['미카게 유즈키', '칸바야시 아스카', '쿠사오 하루토', '시노사키 아유미', '엔마 아이', '이치모쿠렌', '호네온나', '와뉴도', '야마와로', '키쿠리', '기타']
def text_boxes(im, box, thr=150, gap=10):
    m = np.array(wmask(im, box, thr=thr, grow=0)) > 0
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None

def do665():
    for n, t in enumerate(N665):
        k = '665_%d_0' % n; im = orig(k); W, H = im.size
        nb = text_boxes(im, (15, 4, 150, H - 4)); yb = text_boxes(im, (155, 4, 212, H - 4)); mb = text_boxes(im, (212, 4, 270, H - 4))
        cy = 22
        e = panel(im, [(15, 4, 270, H - 4)], [(t, (max(74, (nb[0] + nb[2]) / 2), cy), 18, 104), ('켬', ((yb[0] + yb[2]) / 2, cy), 20), ('끔', ((mb[0] + mb[2]) / 2, cy), 20)],
                  frame_box=(10, 4, 272, H - 4))
        save(k, e)
    preview(['665_%d_0' % i for i in range(11)], '665.png')

def do666():
    for n, t in enumerate(['CG 감상', 'MOVIE 감상']):
        for st in range(3):
            k = '666_%d_0' % (n * 3 + st); im = orig(k); W, H = im.size
            thr = 120 if st == 1 else 190
            m = wmask(im, (55, 6, 222, 42), thr=thr, sat=70 if st == 1 else 45, grow=3)
            src = boxsrc((H, W), (30, 8, 246, 41))
            e = efill(im, m, src=src, P=3)
            col = (175, 178, 215, 255) if st == 1 else (255, 255, 255, 255)
            gl = (40, 40, 90, 255) if st == 1 else (255, 255, 255, 200)
            e = put(e, t, (138, 25), 22, 'EB', fill=col, stroke=2, sf=(30, 30, 40, 255), sx=fitx(t, 'EB', 22, 150), glow=2 if st != 1 else 0, gf=gl)
            save(k, e)
    preview(['666_%d_0' % i for i in range(6)], '666.png')

# ---------------------------------------------------------------- 667 mini-theater titles
T667 = [('1', '소꿉친구랑 하는 모든 것'), ('2', '공포! 히메우마 마을의 괴이!'), ('3', '쿠킹 파이트! 레디이이, 고오오!'),
        ('4', '지옥 마작'), ('5', '헬 걸 스토리'), (None, None), (None, '이어하기')]
G667 = [((15, 110, 175), (15, 178, 207)), ((10, 95, 148), (10, 150, 173)), ((15, 176, 146), (15, 208, 135))]
OUT667 = [(255, 255, 255, 255), (205, 211, 212, 255), (255, 255, 255, 255)]

def heart_layer(im):
    a = np.array(im); r, g = a[..., 0].astype(int), a[..., 1].astype(int)
    pink = (r > g + 60) & (a[..., 3] > 0)
    if not pink.any(): return None
    # fill the digit hole: closed contour of the heart, inpaint the non-pink inside
    filled = cv2.morphologyEx(pink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    full = np.zeros_like(filled); cv2.drawContours(full, cs, -1, 1, -1)
    hole = (full > 0) & ~pink
    out = np.zeros_like(a); out[..., :3] = np.where(hole[..., None], np.array([255, 27, 124], np.uint8), a[..., :3]); out[..., 3] = np.where(full > 0, 255, 0)
    ring = cv2.dilate(full, np.ones((3, 3), np.uint8)) > 0
    L = Image.fromarray(out, 'RGBA')
    wr = np.zeros_like(a); wr[ring] = (255, 255, 255, 255)
    return Image.alpha_composite(Image.fromarray(wr, 'RGBA'), L)

def grad_text(size, text, fnt, xy, g0, g1, outline, sx=1.0, anchor='lm'):
    W, H = size
    m = text_layer(size, text, fnt, fill=(255, 255, 255, 255), xy=xy, anchor=anchor, scale_x=sx)
    o = text_layer(size, text, fnt, fill=outline, xy=xy, anchor=anchor, stroke=2, stroke_fill=outline, scale_x=sx)
    t = np.linspace(0, 1, H)[:, None, None]; t = np.clip((t * H - 8) / 15, 0, 1)
    col = (np.array(g0)[None, None] * (1 - t) + np.array(g1)[None, None] * t) * np.ones((H, W, 1))
    ma = np.array(m)[..., 3]
    f = np.dstack([col, ma]).astype(np.uint8)
    return Image.alpha_composite(o, Image.fromarray(f, 'RGBA'))

def do667():
    f = font('nEB', 16); cy = 16
    for n, (num, title) in enumerate(T667):
        for st in range(3):
            k = '667_%d_0' % (n * 3 + st); im = orig(k); W, H = im.size
            if title is None: save(k, im); continue   # ？？？？ has no Japanese
            base = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            h = heart_layer(im)
            g0, g1 = G667[st]; ol = OUT667[st]
            if num:
                if h: base = Image.alpha_composite(base, h)
                hx = 29 if n != 2 else 35
                if h is not None:
                    xs = np.nonzero(np.array(h)[..., 3].any(0))[0]; hx = (xs.min() + xs.max()) / 2
                base = Image.alpha_composite(base, grad_text((W, H), '제', f, (hx - 28, cy), g0, g1, ol))
                base = Image.alpha_composite(base, grad_text((W, H), num, font('nH', 17), (hx, cy), g0, g1, ol, anchor='mm'))
                base = Image.alpha_composite(base, grad_text((W, H), '화', f, (hx + 12, cy), g0, g1, ol))
                x0 = hx + 43
            else:
                if h: base = Image.alpha_composite(base, h)
                x0 = 4
            l, t_, r, b = f.getbbox(title); sx = min(1.0, (W - 2 - x0) / (r - l))
            base = Image.alpha_composite(base, grad_text((W, H), title, f, (x0, cy), g0, g1, ol, sx=sx))
            save(k, base)
    preview(['667_%d_0' % i for i in range(21)], '667.png')

# ---------------------------------------------------------------- 660 gallery plaques
N660 = ['유즈키', '아스카', '하루토', '아유미', '토모히코', '미사키', '야요이코', '무라카미', '노부히코', '마사하루', '키비츠']
def do660():
    for n, t in enumerate(N660):
        for st in range(2):
            k = '660_%d_0' % (n * 2 + st); im = orig(k); a = np.array(im).astype(int)
            plain = np.array(orig('660_%d_0' % (n * 2 + 1)))[..., 3] > 200   # odd = plaque alone
            inner = cv2.erode(plain.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            lum = a[..., :3].mean(2)
            txt = inner & ((lum < 145) | ((a[..., 1] > 160) & (a[..., 2] > 105)))
            m = cv2.dilate(txt.astype(np.uint8), np.ones((5, 5), np.uint8)) & inner.astype(np.uint8)
            e = inpaint(im, Image.fromarray(m * 255), 5)
            e = e.filter(ImageFilter.SMOOTH) if False else e
            ys = np.nonzero(inner.any(1))[0]; xs = np.nonzero(inner.any(0))[0]
            cx = (xs.min() + xs.max()) / 2 + 0.5; cy = (ys.min() + ys.max()) / 2 - 1
            L = len(t); size = {2: 18, 3: 17, 4: 14}[L]
            e = over(e, text_layer(im.size, t, font('EB', size), fill=(40, 22, 12, 255), xy=(cx, cy), vertical=True,
                                   stroke=1, stroke_fill=(232, 210, 165, 210)))
            save(k, e)
    preview(['660_%d_0' % i for i in range(22)], '660.png')

# ---------------------------------------------------------------- backgrounds 722, 723, 728-739
def colfill(im, mask):
    """fill masked runs by linear interpolation along each column (vertical-stripe plaques)"""
    a = np.array(im).astype(np.float32); m = np.array(mask) > 0; H, W = m.shape
    for x in range(W):
        col = m[:, x]
        if not col.any(): continue
        y = 0
        while y < H:
            if col[y]:
                y1 = y
                while y1 < H and col[y1]: y1 += 1
                top = a[y - 1, x] if y > 0 else a[y1, x]; bot = a[y1, x] if y1 < H else top
                for t, yy in enumerate(range(y, y1)):
                    f = (t + 1) / (y1 - y + 1); a[yy, x] = top * (1 - f) + bot * f
                y = y1
            else: y += 1
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA')

def paste_sprite(bg, key, gain=0.75):
    """re-apply a translated sprite onto a background that contains it dimmed by `gain`"""
    B = np.array(bg).astype(np.float32); T0 = np.array(orig(key)).astype(np.float32)
    T1 = np.array(Image.open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'translation/img/out/%s.png' % key)).convert('RGBA')).astype(np.float32)
    r = cv2.matchTemplate(np.ascontiguousarray(B[..., :3]), np.ascontiguousarray(T0[..., :3] * gain), cv2.TM_SQDIFF)
    _, _, (x, y), _ = cv2.minMaxLoc(r); h, w = T0.shape[:2]
    Bc = B[y:y + h, x:x + w]
    match = (np.abs(Bc[..., :3] - T0[..., :3] * gain) < 2.5).all(2)
    changed = (np.abs(T1 - T0).sum(2) > 0)
    ch = cv2.dilate(changed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    sel = match | ch
    Bc[sel, :3] = T1[sel, :3] * gain
    B[y:y + h, x:x + w] = Bc
    return Image.fromarray(np.clip(B, 0, 255).astype(np.uint8), 'RGBA')

def title(im, box, lines, how='col', thr=190, w='EB', gain=1.0):
    """erase white title text in box (col = stripe interpolation, ex = exemplar) and draw lines [(text, (cx,cy), size)]"""
    m = wmask(im, box, thr=int(thr * gain), grow=2)
    if how == 'col': e = colfill(im, m)
    else:
        W, H = im.size; x0, y0, x1, y1 = box
        e = efill(im, m, src=boxsrc((H, W), (max(0, x0 - 30), max(0, y0 - 12), min(W, x1 + 30), min(H, y1 + 12))), P=3)
    g = int(255 * gain)
    for t, c, size in lines:
        e = put(e, t, c, size, w, fill=(g, g, g, 255), stroke=2, sf=(10, 10, 10, 255))
    return e

def back(im, box, x0, gain=1.0):
    """'戻る' next to the x button -> '돌아가기' (left aligned at x0)"""
    m = wmask(im, box, thr=int(170 * gain), grow=2)
    W, H = im.size; bx0, by0, bx1, by1 = box
    e = efill(im, m, src=boxsrc((H, W), (bx0 - 6, by0 - 10, min(W, bx1 + 30), min(H, by1 + 10))) & ~(np.array(m) > 0), P=3)
    g = int(255 * gain)
    sx = min(1.0, (W - 3 - x0) / (font('B', 15).getlength('돌아가기') + 2))
    L = text_layer(im.size, '돌아가기', font('B', 15), fill=(g, g, g, 255), xy=(x0, (by0 + by1) / 2), anchor='lm', stroke=1, stroke_fill=(20, 20, 30, 255), scale_x=sx)
    return over(e, L)

def dobg():
    B1 = (580, 405, 626, 430)            # 戻る bottom right (723, 735-739)
    im = orig('722__0'); im = title(im, (505, 52, 575, 92), [('갤러리', (541, 73), 22)]); im = back(im, (70, 400, 112, 422), 74); save('722__0', im)
    im = orig('723__0')
    for n in range(6): im = paste_sprite(im, '661_%d_0' % (n * 3))
    im = title(im, (15, 30, 155, 64), [('옵션', (84, 47), 24)], gain=0.75 if False else 1.0)
    im = back(im, B1, 585); save('723__0', im)
    im = orig('728__0'); save('728__0', title(im, (270, 18, 368, 50), [('세이브', (319, 34), 22)]))
    im = orig('729__0'); save('729__0', title(im, (274, 17, 366, 49), [('로드', (320, 33), 22)]))
    im = orig('730__0'); im = title(im, (492, 19, 603, 53), [('CG 감상', (547, 36), 22)]); im = back(im, (403, 27, 442, 48), 406); save('730__0', im)
    im = orig('731__0'); im = title(im, (495, 47, 580, 76), [('감상', (537, 61), 22)]); im = back(im, (403, 27, 442, 48), 406); save('731__0', im)
    for k, t, bx in [('732', '시노사키 아유미', (260, 14, 380, 50)), ('733', '칸바야시 아스카', (244, 14, 400, 50)), ('734', '이치모쿠렌', (274, 14, 372, 50))]:
        im = orig(k + '__0'); im = title(im, bx, [(t, (322, 33), 21)], how='ex', w='B'); im = back(im, (585, 19, 626, 42), 586); save(k + '__0', im)
    for k, pan, lines, bx in [('735', ['662_%d_0' % i for i in range(4)], [('게임 설정', (84, 47), 22)], (14, 30, 153, 65)),
                             ('736', ['663_%d_0' % i for i in range(3)], [('스킵 모드', (86, 49), 19), ('설정', (86, 73), 22)], (12, 32, 160, 88)),
                             ('737', ['664_%d_0' % i for i in range(3)], [('사운드 설정', (84, 46), 22)], (9, 28, 159, 63)),
                             ('738', ['665_%d_0' % i for i in range(11)], [('음성 전환', (90, 48), 22)], (33, 30, 147, 67)),
                             ('739', [], [('조작 방법', (87, 48), 22)], (29, 30, 145, 66))]:
        im = orig(k + '__0')
        for p in pan: im = paste_sprite(im, p)
        im = title(im, bx, lines); im = back(im, B1, 585); save(k + '__0', im)
    if PV:
        ks = ['722', '723', '728', '729', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739']
        for i in range(0, 14, 2): sheet([k + '__0' for k in ks[i:i + 2]], os.path.join(PV, 'bg%d.png' % i), scale=1)

if __name__ == '__main__':
    todo = sys.argv[2:] or ['660', '661', '662', '665', '666', '667', 'bg']
    for t in todo: globals()['do' + t]()
