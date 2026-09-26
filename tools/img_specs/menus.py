"""menus: 603, 620, 649, 650-659, 701, 702, 750, 567, 553 image Korean redraw.
run: python tools/img_specs/menus.py [group ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imgdraw import *

PREV = 'extract/preview/menus'
os.makedirs(PREV, exist_ok=True)


def gfont(size, which='B'):
    """brush-lettered items: SeoulHangang (B default, EB for heavy) from the work folder"""
    return font(which, size)


def A(im):
    return np.array(im.convert('RGBA')).astype(int)


def fit(text, box, fn, start=60):
    for s in range(start, 6, -1):
        f = fn(s); l, t, r, b = f.getbbox(text)
        if r - l <= box[0] and b - t <= box[1]: return f
    return fn(7)


def ink_bbox(layer):
    a = np.array(layer)[:, :, 3]; ys, xs = np.nonzero(a > 40)
    return xs.min(), ys.min(), xs.max(), ys.max()


def keys(prefix):
    import glob
    return sorted((os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, 'extract/img_orig/%s_*.png' % prefix))),
                  key=lambda k: [int(x) if x.isdigit() else x for x in k.replace('-', '_').split('_')])


# ---------------------------------------------------------------- 603 in-game menu buttons
def g603():
    # text per pair, drawing box (x0,y0,x1,y1)
    spec = {0: ('불러오기', (70, 18, 158, 58)), 2: ('저장', (8, 18, 88, 60)), 4: ('설정', (13, 18, 93, 56)),
            6: ('타이틀', (5, 22, 92, 60)), 8: ('뒤로', (16, 16, 94, 58))}
    for i in range(10):
        k = '603_%d_0' % i; im = orig(k); txt, box = spec[i - i % 2]
        x0, y0, x1, y1 = box
        m = mask_where(im, lambda r, g, b, a: (np.minimum(np.minimum(r, g), b) > 140) & (a > 100),
                       box=(x0 - 2, y0 - 4, x1 + 2, y1 + 4), grow=3)
        im = inpaint(im, m, 5)
        sx = 0.72 if len(txt) >= 4 else 1.0
        f = fit(txt, ((x1 - x0 - 4) / sx, y1 - y0 - 4), gfont, 40)
        lay = text_layer(im.size, txt, f, xy=((x0 + x1) / 2, (y0 + y1) / 2), stroke=2, scale_x=sx,
                         stroke_fill=(0, 0, 0, 255), shadow=(2, 2, (0, 0, 0, 200)))
        # keep the text inside the panel (alpha of original)
        save(k, over(im, lay))
    sheet(keys('603'), os.path.join(PREV, 's603.png'), 3)


# ---------------------------------------------------------------- 620 title menu
T620 = ['처음부터'] * 3 + ['이어하기'] * 3 + ['갤러리'] * 3 + ['설정'] * 3 + ['보너스'] * 6


def bbox(m):
    ys, xs = np.nonzero(m); return xs.min(), ys.min(), xs.max(), ys.max()


def orb620():
    """clean red orb (with its halo) from all orb-bearing frames: per-pixel brightest sample removes black text"""
    crops = []
    for i in range(18):
        if i % 3 == 1: continue
        im = orig('620_%d_0' % i); a = A(im)
        x0, y0, _, _ = bbox((a[:, :, 0] > 150) & (a[:, :, 1] < 90) & (a[:, :, 3] > 100))
        crops.append(A(im.crop((x0 - 9, y0 - 8, x0 + 27, y0 + 30))))
    st = np.stack(crops)
    # exclude black-text and gold-text samples: score = brightness, penalise yellow/orange (r>g>b with b low and g high)
    r, g, b = st[..., 0], st[..., 1], st[..., 2]
    gold = (r > 150) & (g > 90) & (b < 80)
    score = (r + g + b) * st[..., 3] / 255 - gold * 2000
    idx = score.argmax(0)
    out = np.take_along_axis(st, idx[None, ..., None].repeat(4, -1), 0)[0].astype(float)
    h, w = out.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - 19, yy - 18)  # feather the rectangular crop edge into the halo
    out[..., 3] *= np.clip((17 - d) / 6, 0, 1)
    return Image.fromarray(out.astype(np.uint8), 'RGBA'), 9  # 9 = offset of red left edge inside crop


def g620():
    orb, ox = orb620()
    for i in range(18):
        k = '620_%d_0' % i; st = i % 3; txt = T620[i]
        W, H = 256, 48
        f = gfont(24)
        sp = ' '.join(txt)
        lay_t = text_layer((W, H), sp, f, xy=(W / 2 + 4, H / 2 + 1), fill=(10, 8, 8, 255))
        x0, y0, x1, y1 = ink_bbox(lay_t)
        glow = text_layer((W, H), sp, f, xy=(W / 2 + 4, H / 2 + 1), fill=(255, 255, 255, 0), stroke=5,
                          stroke_fill=(255, 255, 255, 255))
        glow = glow.filter(ImageFilter.GaussianBlur(5))
        ga = np.array(glow); ga[..., 3] = np.clip(ga[..., 3].astype(int) * 1.5, 0, 255); ga[..., :3] = 255
        im = Image.fromarray(ga, 'RGBA')
        if st != 1:
            o = Image.new('RGBA', (W, H), (0, 0, 0, 0)); ox0 = x0 - 13 - ox
            o.alpha_composite(orb, (ox0, 4)); im = Image.alpha_composite(im, o)
        if st == 2:
            ta = np.array(lay_t)[..., 3]
            xs = np.linspace(0, 1, W)[None, :].repeat(H, 0)
            t = np.clip((xs * W - x0) / max(1, x1 - x0), 0, 1)
            c0, c1 = np.array([235, 215, 0]), np.array([205, 85, 15])
            rgb = c0[None, None] * (1 - t[..., None]) + c1[None, None] * t[..., None]
            ol = text_layer((W, H), sp, f, xy=(W / 2 + 4, H / 2 + 1), fill=(110, 50, 0, 255), stroke=1,
                            stroke_fill=(110, 50, 0, 230))
            im = Image.alpha_composite(im, ol)
            ta = np.array(text_layer((W, H), sp, f, xy=(W / 2 + 4, H / 2 + 1)))[..., 3]
            g = Image.fromarray(np.dstack([rgb, ta]).astype(np.uint8), 'RGBA')
            im = Image.alpha_composite(im, g)
        else:
            im = Image.alpha_composite(im, lay_t)
        save(k, im)
    sheet(keys('620'), os.path.join(PREV, 's620.png'), 2)


# ---------------------------------------------------------------- helpers for grid-textured buttons
def periodic_fill(im, mask, P=4, maxk=30):
    """replace masked pixels by the nearest unmasked pixel offset by multiples of P (keeps a P-periodic dot grid)"""
    a = np.array(im.convert('RGBA')); m = np.array(mask) > 0; out = a.copy(); H, W = m.shape
    offs = sorted({(dx * P, dy * P) for dx in range(-maxk, maxk + 1) for dy in range(-maxk, maxk + 1) if dx or dy},
                  key=lambda o: o[0] ** 2 + 20 * o[1] ** 2)
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        for dx, dy in offs:
            xx, yy = x + dx, y + dy
            if 0 <= xx < W and 0 <= yy < H and not m[yy, xx]:
                out[y, x] = a[yy, xx]; break
    return Image.fromarray(out, 'RGBA')


def pct_color(im, mask, dark=True, q=15):
    a = A(im); m = np.array(mask) > 0; px = a[m][:, :3]; lum = px.sum(1)
    sel = px[lum <= np.percentile(lum, q)] if dark else px[lum >= np.percentile(lum, 100 - q)]
    return tuple(int(v) for v in sel.mean(0)) + (255,)


# ---------------------------------------------------------------- 649 / 750 site buttons
def small_bg(h):
    """rebuild the dotted interior of the small site buttons: per row & x-phase(4) median of non-text pixels"""
    st = np.stack([A(orig('649_%d_0' % i)) for i in (3, 4, 6)])
    c = st[0].copy(); H, W = c.shape[:2]
    for y in range(H):
        for ph in range(4):
            xs = [x for x in range(9, 54) if x % 4 == ph]
            v = st[:, y, xs].reshape(-1, 4); ok = v[:, 1] > 185
            if ok.sum() >= 3:
                med = np.median(v[ok], 0)
                for x in xs:
                    c[y, x] = med
            # interior columns 5..8 and 54..57: same row/phase from nearest
    return_cols = None
    if h == 37: c = np.concatenate([c[:18], c[17:]], 0)
    return c.astype(np.uint8)

def g649():
    T = {0: '전송', 1: '전송', 2: '진행', 3: '뒤로', 4: '삭제', 5: '초기화', 6: '결정'}
    jobs = [('649_%d_0' % i, T[i], i == 1) for i in range(7)] + [('750_1-0_0', '전송', False), ('750_1-1_0', '전송', True)]
    cleaned = {}
    for k, txt, lit in jobs:
        im = orig(k); a = A(im); ys, xs = np.nonzero(a[:, :, 1] > 150)
        bx = (xs.min() + 4, ys.min() + 3, xs.max() - 3, ys.max() - 2)
        big = im.width > 100
        if big:
            ty = 21 if k.startswith('649') else 25
            tb = (36, ty - 3, 124, ty + 39)
            cx, cy = 80, ty + 18
            if lit:
                uk = k.replace('_1_0', '_0_0').replace('1-1', '1-0')
                u0 = A(orig(uk)); l0 = A(orig(k)); keep = np.ones(u0.shape[:2], bool); keep[tb[1]:tb[3], tb[0]:tb[2]] = False
                bc = A(cleaned[uk]).astype(float)
                for c in range(3):  # map unlit->lit background per channel (fit outside the text zone)
                    for lo, hi in ((0, 90), (90, 256)):
                        sel = keep & (u0[..., c] >= lo) & (u0[..., c] < hi)
                        if sel.sum() > 20:
                            pa = np.polyfit(u0[..., c][sel], l0[..., c][sel], 1)
                            z = (bc[..., c] >= lo) & (bc[..., c] < hi); bc[..., c][z] = np.polyval(pa, bc[..., c][z])
                base = Image.fromarray(np.clip(bc, 0, 255).astype(np.uint8), 'RGBA')
                f = font('EB', 26); sp = ' '.join(txt)
                halo = text_layer(im.size, sp, f, xy=(cx, cy), fill=(255, 255, 255, 90), stroke=4,
                                  stroke_fill=(255, 255, 255, 90)).filter(ImageFilter.GaussianBlur(5))
                lay = text_layer(im.size, sp, f, xy=(cx, cy), fill=(250, 255, 255, 255))
                save(k, over(over(base, halo), lay)); continue
            m = mask_where(im, lambda r, g, b, al: g < 190, box=tb, grow=2)
            col = pct_color(im, m, dark=True)
            im = periodic_fill(im, m); cleaned[k] = im
            f = font('EB', 26); sp = ' '.join(txt)
            lay = text_layer(im.size, sp, f, xy=(cx, cy), fill=col[:3] + (235,)).filter(ImageFilter.GaussianBlur(0.7))
        else:
            m = mask_where(im, lambda r, g, b, al: g < 185, box=bx, grow=1)
            col = pct_color(im, m, dark=True)
            im = Image.fromarray(small_bg(im.height), 'RGBA')
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2 + 1
            sp = ' '.join(txt) if len(txt) == 2 else txt
            f = fit(sp, (bx[2] - bx[0] - 5, bx[3] - bx[1] - 3), lambda z: font('B', z), 22)
            lay = text_layer(im.size, sp, f, xy=(cx, cy), fill=col)
        save(k, over(im, lay))
    sheet([j[0] for j in jobs], os.path.join(PREV, 's649.png'), 3)


# ---------------------------------------------------------------- 652/653 yes/no glowing boxes
def g652():
    base = {}
    cur = None
    # cursor sprite from 652_1 (grey/white arrow + black outline)
    a = A(orig('652_1_0')); b = A(orig('652_0_0'))
    sat = a[..., :3].max(2) - a[..., :3].min(2)
    cb = (sat < 30) & (a[..., :3].sum(2) > 200)
    cm = cb | ((a[..., 1] < 32) & (cv2.dilate(cb.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
    cm[:, :45] = False; cm[:, 90:] = False; cm[:10] = False; cm[52:] = False
    cm &= (np.abs(a - b)[..., :3].sum(2) > 25) | (a[..., 1] < 32)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cm.astype(np.uint8), connectivity=8)
    cm = lab == (1 + stats[1:, cv2.CC_STAT_AREA].argmax())
    for fid, txt in (('652', '예'), ('653', '아니요')):
        for st in range(3):
            k = '%s_%d_0' % (fid, st); im = orig(k); a = A(im)
            bx = (12, 11, 117, 48)
            tm = (a[..., 1] > 75)
            tmask = np.zeros_like(tm); tmask[bx[1]:bx[3], bx[0]:bx[2]] = tm[bx[1]:bx[3], bx[0]:bx[2]]
            tmask = cv2.dilate(tmask.astype(np.uint8), np.ones((5, 5), np.uint8))
            col = [(120, 205, 215, 255), (205, 250, 250, 255), (95, 165, 175, 255)][st]
            clean = inpaint(im, Image.fromarray(tmask * 255), 5)
            f = gfont(34 if len(txt) < 3 else 30)
            lay = text_layer(im.size, txt, f, xy=(64, 31), fill=col, glow=3, glow_fill=col[:3] + (120,))
            out = over(clean, lay)
            if st:
                o = A(out); o[cm] = a[cm] if fid == '652' else A(orig('652_%d_0' % st))[cm]; out = Image.fromarray(o.astype(np.uint8), 'RGBA')
            save(k, out)
    sheet(keys('652') + keys('653'), os.path.join(PREV, 's652.png'), 3)


# ---------------------------------------------------------------- 654/655 hex plank yes/no
def g654():
    for fid, txt in (('654', '예'), ('655', '아니요')):
        for st in range(2):
            k = '%s_%d_0' % (fid, st); im = orig(k)
            m = mask_where(im, lambda r, g, b, a: (((np.minimum(np.minimum(r, g), b) > 150)) |
                                                   ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) < 28) & (r + g + b > 240))) & (a > 100),
                           box=(40, 8, 130, 40), grow=2)
            im = inpaint(im, m, 6)
            f = gfont(28 if len(txt) == 1 else 24)
            sp = txt if len(txt) > 1 else txt
            lay = text_layer(im.size, sp, f, xy=(84, 24), stroke=1, stroke_fill=(0, 0, 0, 255),
                             shadow=(1, 1, (0, 0, 0, 200)))
            save(k, over(im, lay))
    sheet(keys('654') + keys('655'), os.path.join(PREV, 's654.png'), 3)


# ---------------------------------------------------------------- scratchy (scratched brush) text
def scratch_mask(size, text, fnt, xy, seed=0, angle=0, sx=1.0):
    """alpha mask (float 0..1) of text with ragged hair-line scratches, rendered at 3x"""
    S = 3; W, H = size
    f = ImageFont.truetype(fnt.path, fnt.size * S, index=getattr(fnt, 'index', 0))
    L = Image.new('L', (int(W * S / sx) + 1, H * S), 0)
    ImageDraw.Draw(L).text((xy[0] * S / sx, xy[1] * S), text, font=f, fill=255, anchor='mm')
    if sx != 1.0: L = L.resize((W * S, H * S), Image.LANCZOS)
    if angle: L = L.rotate(angle, resample=Image.BICUBIC, center=(xy[0] * S, xy[1] * S))
    m = np.array(L).astype(np.float32) / 255
    rng = np.random.default_rng(seed)
    n = rng.random(m.shape).astype(np.float32)
    k = np.zeros((41, 41), np.float32); cv2.line(k, (12, 40), (28, 0), 1, 1); k /= k.sum()   # slanted streaks
    st = cv2.filter2D(n, -1, k); st = (st - st.mean()) / (st.std() + 1e-6)
    ms = cv2.GaussianBlur(m, (0, 0), 1.0)
    tip = np.clip(cv2.filter2D(m, -1, k) * 3, 0, 1)     # strokes drag out along the streak direction
    base = np.maximum(ms, tip * 0.55)
    out = np.clip(base * 2.4 - 1.0 + 0.22 * st, 0, 1) * (base > 0.12)
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    return np.clip(out * 1.4, 0, 1)


def paint(mask, rgb):
    a = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([np.full(mask.shape + (3,), rgb, np.uint8), a]), 'RGBA')


# ---------------------------------------------------------------- 650/651 hell choice
def g650():
    for fid, txt in (('650', '예'), ('651', '아니요')):
        for st in range(3):
            k = '%s_%d_0' % (fid, st); im = orig(k); a = A(im)
            W, H = im.size; cx, cy = W / 2 + 2, H / 2 - 4
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            if st == 0:
                m = (np.minimum(np.minimum(r, g), b) > 60) & (a[..., 3] > 60)
            else:
                m = (r > 110) & (r > g * 2.2) & (a[..., 3] > 60)
                if st == 1:
                    m &= r > 170
            m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8))
            im = inpaint(im, m * 255, 4)
            f = gfont(58 if len(txt) == 1 else 46)
            if st == 0:
                im = over(im, paint(scratch_mask(im.size, txt, f, (cx, cy), seed=1, sx=0.8), (250, 250, 250)))
            elif st == 1:
                im = over(im, paint(scratch_mask(im.size, txt, f, (cx, cy), seed=2, sx=0.8), (225, 25, 25)))
            else:
                rng = np.random.default_rng(7 if fid == '650' else 8)
                n_ = 7 if len(txt) == 1 else 5
                for j in range(n_):
                    size = int(rng.integers(34, 56)) if len(txt) == 1 else int(rng.integers(28, 40))
                    spread = 55 if len(txt) == 1 else 30
                    x = cx + rng.uniform(-spread, spread); y = cy + rng.uniform(-16, 16)
                    col = (int(rng.integers(200, 250)), 15, 15)
                    alpha = 1.0 if j > 1 else 0.55
                    lay = paint(scratch_mask(im.size, txt, gfont(size), (x, y), seed=10 + j,
                                             angle=rng.uniform(-10, 10), sx=rng.uniform(0.6, 0.9)) * alpha, col)
                    im = over(im, lay)
            save(k, im)
    sheet(keys('650') + keys('651'), os.path.join(PREV, 's650.png'), 2)


# ---------------------------------------------------------------- 553 "send to hell?"
def g553():
    k = '553__0'; im = Image.new('RGBA', (320, 86), (0, 0, 0, 0))
    txt = '지옥에 보내겠습니까?'
    m = scratch_mask(im.size, txt, gfont(34), (160, 42), seed=3, sx=0.82)
    glow = cv2.GaussianBlur(m, (0, 0), 3) * 0.9
    im = over(im, paint(np.clip(glow * 1.2, 0, 0.6), (200, 210, 255)))
    im = over(im, paint(m, (255, 255, 255)))
    save(k, im)
    sheet([k], os.path.join(PREV, 's553.png'), 3)


# ---------------------------------------------------------------- 567 XP error dialog
def g567():
    k = '567__0'; im = orig(k); a = A(im)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = np.zeros(r.shape, bool)
    m[8:26, 14:95] = (r > 120)[8:26, 14:95]                       # title (white on blue)
    m[52:96, 130:262] = (r + g + b < 640)[52:96, 130:262]          # grey message
    m[50:76, 150:240] |= ((r > 150) & (r - g > 25))[50:76, 150:240]   # red heading
    m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8))
    im = periodic_fill(im, m * 255, P=2, maxk=60)
    f1 = font('nB', 13); f2 = font('nB', 15); f3 = font('nEB', 15)
    im = over(im, text_layer(im.size, '오류', f1, xy=(18, 17), anchor='lm', fill=(255, 255, 255, 255),
                             shadow=(1, 1, (10, 40, 140, 200))))
    im = over(im, text_layer(im.size, '오류!', f3, xy=(196, 64), fill=(250, 60, 60, 255)))
    im = over(im, text_layer(im.size, '받을 수 없습니다', f2, xy=(196, 85), fill=(60, 60, 60, 255)))
    save(k, im)
    sheet([k], os.path.join(PREV, 's567.png'), 3)


# ---------------------------------------------------------------- 656-659 gallery labels, 702/701 skip labels
def outlined(size, txt, fnt, xy, fill, outline=(40, 30, 0, 255), sw=1, sx=1.0, anchor='mm'):
    return text_layer(size, txt, fnt, xy=xy, fill=fill, stroke=sw, stroke_fill=outline, scale_x=sx, anchor=anchor)


def clear_alpha(im, box):
    a = np.array(im); x0, y0, x1, y1 = box; a[y0:y1, x0:x1] = 0; return Image.fromarray(a, 'RGBA')


def yellow_grad(layer, top=(255, 255, 120), bot=(235, 200, 0)):
    a = np.array(layer).astype(float); H = a.shape[0]
    t = np.linspace(0, 1, H)[:, None, None]
    col = np.array(top)[None, None] * (1 - t) + np.array(bot)[None, None] * t
    ink = (a[..., :3].sum(-1) > 400)[..., None]
    a[..., :3] = np.where(ink, col, a[..., :3]); return Image.fromarray(a.astype(np.uint8), 'RGBA')


def g656():
    Y = (250, 240, 30, 255)
    for k, txt in (('656__0', '음성 재생'), ('657__0', '되감기')):
        im = clear_alpha(orig(k), (0, 29, 88, 64))
        f = fit(txt, (80, 18), lambda z: gfont(z), 20)
        im = over(im, outlined(im.size, txt, f, (44, 41), Y, outline=(60, 40, 0, 255), sw=1))
        save(k, im)
    for k, txt in (('658__0', 'CG 표시'), ('659__0', '장면 재생')):
        im = orig(k)
        x0 = 36 if k.startswith('658') else 33
        im = clear_alpha(im, (x0, 0, 120, 29))
        f = fit(txt, (120 - x0 - 4, 16), lambda z: gfont(z), 18)
        im = over(im, outlined(im.size, txt, f, (x0 + 2, 15), (255, 255, 255, 255), outline=(20, 20, 20, 255),
                               sw=1, anchor='lm'))
        save(k, im)
    T = ['읽음', '강제', '자동']
    for i in range(3):
        k = '702_%d_0' % i; im = Image.new('RGBA', (56, 27), (0, 0, 0, 0))
        im = over(im, yellow_grad(outlined(im.size, T[i], gfont(19), (28, 14), (255, 255, 255, 255),
                                           outline=(70, 50, 0, 255))))
        save(k, im)
    sheet(['656__0', '657__0', '658__0', '659__0', '702_0_0', '702_1_0', '702_2_0'], os.path.join(PREV, 's656.png'), 4)


def g701():
    T = ['읽음'] * 3 + ['강제'] * 3 + ['자동'] * 3
    for i in range(9):
        k = '701_%d_0' % i; im = orig(k); a = A(im)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        yel = (r > 170) & (g > 170) & (b < 150) & (g > r * 0.8)
        m = np.zeros(yel.shape, bool); m[12:42, 25:103] = yel[12:42, 25:103]
        m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8))
        im = inpaint(im, m * 255, 5)
        lay = yellow_grad(outlined(im.size, T[i], gfont(22), (64, 29), (255, 255, 255, 255), outline=(90, 60, 0, 255)),
                          top=(255, 255, 150), bot=(230, 210, 0))
        save(k, over(im, lay))
    sheet(keys('701'), os.path.join(PREV, 's701.png'), 3)


GROUPS = {'603': g603, '620': g620, '649': g649, '652': g652, '654': g654, '650': g650, '553': g553,
          '567': g567, '656': g656, '701': g701}

if __name__ == '__main__':
    for g in (sys.argv[1:] or GROUPS):
        GROUPS[g]()
