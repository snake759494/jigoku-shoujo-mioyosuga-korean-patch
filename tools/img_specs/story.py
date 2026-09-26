"""story: chapter cards 500-506, hell-target name cards 450-458, hell-correspondence site 362/372/374/725,
fake desktop 726, 媛馬村 seal 400/727, To Hell logo 740.  run: python tools/img_specs/story.py"""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from imgdraw import *

PREV = 'extract/preview/story'


def rough(layer, seed=0, amt=0.55, streak=(9, 1.2), holes=0.10):
    """make a text layer look like a scratchy brush: jagged edges + horizontal dry-brush streak gaps"""
    rng = np.random.default_rng(seed)
    a = np.array(layer).astype(float); A = a[:, :, 3] / 255.0
    H, W = A.shape
    n1 = cv2.GaussianBlur(rng.random((H, W)), (0, 0), 1.2)
    n1 = (n1 - n1.mean()) / (n1.std() + 1e-6)
    n2 = cv2.GaussianBlur(rng.random((H, W)), (0, 0), sigmaX=streak[0], sigmaY=streak[1])
    n2 = (n2 - n2.mean()) / (n2.std() + 1e-6)
    Ab = cv2.GaussianBlur(A, (0, 0), 0.8)
    v = (Ab - 0.5) * 3.0 + amt * 0.25 * n1 + 0.5
    v = np.clip(v, 0, 1)
    v = np.where(n2 > (2.2 - holes * 10), v * 0.15, v)   # dry-brush gaps
    a[:, :, 3] = np.clip(v, 0, 1) * 255
    return Image.fromarray(a.astype(np.uint8), 'RGBA')


def bleed(layer, col, r=2, alpha=0.6):
    """soft ink bleed behind a layer"""
    a = np.array(layer); m = a[:, :, 3].astype(float)
    m = cv2.GaussianBlur(m, (0, 0), r) * alpha
    out = np.zeros_like(a); out[:, :, :3] = col; out[:, :, 3] = np.clip(m, 0, 255)
    return Image.alpha_composite(Image.fromarray(out, 'RGBA'), layer)


# ---------------- chapter cards (red scratchy brush on black) ----------------
CHAP = {'500__0': ('프롤로그', (162, 162, 494, 284)), '501__0': ('1일째', (262, 165, 402, 272)),
        '502__0': ('2일째', (250, 180, 402, 265)), '503__0': ('3일째', (250, 180, 402, 290)),
        '504__0': ('4일째', (247, 160, 402, 292)), '505__0': ('5일째', (242, 152, 402, 276)),
        '506__0': ('에필로그', (180, 176, 493, 264))}


def chapter(key, text, box):
    im = orig(key)
    a = np.array(im); m = a[:, :, 0] > 25
    m = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    a[m] = (0, 0, 0, 255); im = Image.fromarray(a, 'RGBA')
    cx = (box[0] + box[2]) / 2; cy = 223   # all cards share the same vertical centre
    s = fit_size(text, 'EB', (box[2] - box[0] + 10, 95), start=92)
    s = min(s, 80)
    t = text_layer(im.size, text, font('EB', s), fill=(230, 0, 4, 255), xy=(cx, cy), stroke=1,
                   stroke_fill=(230, 0, 4, 255))
    t = rough(t, seed=int(key[:3]), amt=0.8, streak=(10, 0.9), holes=0.09)
    t = bleed(t, (120, 0, 0), r=2.5, alpha=0.7)
    save(key, over(im, t))


# ---------------- hell-target name cards (white scratch brush over portrait) ----------------
NAMES = {'450__0': '칸바야시 아스카', '451__0': '쿠사오 하루토', '452__0': '시노사키 아유미',
         '453__0': '키비츠 타카야스', '454__0': '카사마 미사키', '455__0': '마츠카타 토모히코',
         '456__0': '마츠카타 노부히코', '457__0': '마츠카타 미야', '458__0': '쿠사오 마사하루'}


def namecard(key, text):
    im = orig(key)
    a = np.array(im).astype(int)
    lum = a[:, :, :3].mean(2); sat = a[:, :, :3].max(2) - a[:, :, :3].min(2)
    bg = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 6)
    m = ((lum > 125) & (sat < 80)) | ((lum - bg > 25) & (sat < 70))
    m[:25] = 0; m[172:] = 0; m[:, :60] = 0; m[:, 585:] = 0
    m = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8))
    im = inpaint(im, Image.fromarray(m * 255), 7)
    # darken the repaired band a little so the new white letters read like the original
    s = fit_size(text, 'EB', (500, 70), start=66)
    t = text_layer(im.size, text, font('EB', s), fill=(245, 245, 240, 255), xy=(320, 100))
    t = rough(t, seed=int(key[:3]), amt=0.9, streak=(12, 0.8), holes=0.12)
    sh = text_layer(im.size, text, font('EB', s), fill=(30, 0, 0, 170), xy=(322, 102), stroke=2,
                    stroke_fill=(30, 0, 0, 170)).filter(ImageFilter.GaussianBlur(2.5))
    save(key, over(over(im, sh), t))


# ---------------- hell correspondence site ----------------
def vfill(im, box, pad=3):
    """rebuild box by vertical linear interpolation between the rows just above/below (smooth gradients)"""
    a = np.array(im).astype(float); x0, y0, x1, y1 = box
    top = a[y0 - pad:y0, x0:x1].mean(0); bot = a[y1:y1 + pad, x0:x1].mean(0)
    for y in range(y0, y1):
        t = (y - y0 + 0.5) / (y1 - y0); a[y, x0:x1] = top * (1 - t) + bot * t
    return Image.fromarray(a.astype(np.uint8), 'RGBA')


def site_heading(im):
    im = vfill(im, (84, 24, 580, 72))
    t = text_layer(im.size, '당신의 원한, 풀어드립니다.', font('B', 30), fill=(215, 232, 232, 255), xy=(330, 48),
                   glow=3, glow_fill=(120, 220, 230, 150))
    return over(im, t)


def button(im, box, text, size=15, spacing_scale=1.0):
    """box = inner face of a cyan dotted button; dark teal letters"""
    x0, y0, x1, y1 = box
    a = np.array(im).astype(int); lum = a[:, :, :3].mean(2)
    m = np.zeros(lum.shape, np.uint8)
    sub = lum[y0:y1, x0:x1]; m[y0:y1, x0:x1] = sub < np.median(sub) - 25
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    im = inpaint(im, Image.fromarray(m * 255), 2)
    # restore the dotted screen texture on the repaired pixels
    b = np.array(im).astype(int); yy, xx = np.nonzero(m)
    dots = ((yy + xx) % 2 == 0)
    b[yy[dots], xx[dots], :3] = np.clip(b[yy[dots], xx[dots], :3] - 18, 0, 255)
    im = Image.fromarray(b.astype(np.uint8), 'RGBA')
    t = text_layer(im.size, text, font('B', size), fill=(10, 50, 65, 255), xy=((x0 + x1) / 2, (y0 + y1) / 2 + 1))
    return over(im, t)


def site(key):
    im = site_heading(orig(key))
    if key != '362__0':
        im = button(im, (270, 168, 376, 208), '전 송', 21)
    if key == '725__0':
        for i, tx in enumerate(['진 행', '뒤 로', '삭 제', '초기화', '결 정']):
            y = [245, 281, 317, 354, 390][i]
            im = button(im, (511, y + 4, 561, y + 28), tx, 16 if len(tx) < 3 else 14)
    save(key, im)


# ---------------- fake desktop 726 ----------------
def desk():
    im = orig('726__0')
    a = np.array(im).astype(int)
    for box in [(368, 84, 432, 107), (370, 173, 442, 199)]:
        x0, y0, x1, y1 = box
        m = np.zeros(a.shape[:2], np.uint8); sub = a[y0:y1, x0:x1, :3].min(2)
        m[y0:y1, x0:x1] = sub > 150
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        im = inpaint(im, Image.fromarray(m * 255), 3); a = np.array(im).astype(int)
    f = font('B', 16)
    im = over(im, text_layer(im.size, '대상자', f, fill=(245, 245, 245, 255), xy=(400, 96)))
    im = over(im, text_layer(im.size, '죽일래?', f, fill=(245, 245, 245, 255), xy=(405, 186)))
    # taskbar labels
    a = np.array(im).astype(int)
    def fillbox(box, col):
        nonlocal a
        x0, y0, x1, y1 = box; a[y0:y1, x0:x1, :3] = col
    # start button: blue gradient -> rebuild by vertical interpolation over text span
    im = Image.fromarray(a.astype(np.uint8), 'RGBA')
    im = vfill(im, (20, 431, 78, 445), 1)
    im = over(im, text_layer(im.size, '시작', font('nB', 12), fill=(255, 255, 255, 255), xy=(46, 438),
                             stroke=1, stroke_fill=(20, 40, 120, 255)))
    im = vfill(im, (116, 431, 194, 445), 1)
    im = over(im, text_layer(im.size, '지옥 보내기 시…', font('nB', 11), fill=(30, 20, 0, 255), xy=(154, 438)))
    # IME indicator あ 般 -> 가 한
    a = np.array(im).astype(int); m = np.zeros(a.shape[:2], np.uint8)
    m[429:447, 462:516] = (a[429:447, 462:516, :3].min(2) > 200) | (a[429:447, 462:516, 2] > 140)
    im = inpaint(im, Image.fromarray(cv2.dilate(m, np.ones((5, 5), np.uint8)) * 255), 3)
    im = over(im, text_layer(im.size, '가 한', font('nB', 13), fill=(255, 255, 255, 255), xy=(488, 438)))
    save('726__0', im)


# ---------------- 媛馬村 seal 400 / 727 ----------------
def seal(im, box, col):
    x0, y0, x1, y1 = box
    a = np.array(im).astype(int)
    m = np.zeros(a.shape[:2], np.uint8)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    red = (r > 120) & (r - g > 70) & (r - b > 70)
    m[y0 - 3:y1 + 3, x0 - 3:x1 + 3] = red[y0 - 3:y1 + 3, x0 - 3:x1 + 3]
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    im = inpaint(im, Image.fromarray(m * 255), 5)
    W, H = x1 - x0, y1 - y0
    lay = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    lw = max(2, W // 18)
    d.rounded_rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), radius=W // 2 - 1, outline=col, width=lw)
    s = int(W * 0.40)
    f = font('EB', s)
    rows = ['히메', '우마', '마을']
    step = (H - W * 0.55) / 3
    for i, t in enumerate(rows):
        d.text(((x0 + x1) / 2, y0 + W * 0.3 + step * (i + 0.5)), t, font=f, fill=col, anchor='mm')
    return over(im, lay)


def seal_img(key, box):
    im = seal(orig(key), box, (250, 0, 0, 255))
    if key == '727__0':
        # vertical 「場所を選」 + row 「択して」 -> 장/소/를/선 + 택해 줘
        a = np.array(im).astype(int)
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        txt = (r > 170) & (b < 90) & ((r - b) > 110)
        m = np.zeros(a.shape[:2], np.uint8)
        m[292:434, 20:52] = txt[292:434, 20:52]; m[402:434, 20:200] = txt[402:434, 20:200]
        m = cv2.dilate(m, np.ones((7, 7), np.uint8))
        im = inpaint(im, Image.fromarray(m * 255), 4)
        f = font('EB', 19)
        lay = Image.new('RGBA', im.size, (0, 0, 0, 0))
        for i, c in enumerate('장소를'):
            lay = over(lay, text_layer(im.size, c, f, fill=(240, 30, 0, 255), xy=(36, 314 + i * 32), stroke=1,
                                       stroke_fill=(250, 220, 0, 255)))
        lay = over(lay, text_layer(im.size, '선택해 주세요', f, fill=(240, 30, 0, 255), xy=(26, 420), anchor='lm',
                                   stroke=1, stroke_fill=(250, 220, 0, 255)))
        im = over(im, lay)
    save(key, im)


# ---------------- To Hell logo 740 ----------------
def tohell():
    im = orig('740__0')
    a = np.array(im).astype(int)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    # ribbon katakana: dark navy on yellow ribbon
    m = np.zeros(a.shape[:2], np.uint8)
    rb = (b > 90) & (r < 110) & (g < 110)
    m[146:172, 262:392] = rb[146:172, 262:392]
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    im = inpaint(im, Image.fromarray(m * 255), 3)
    im = over(im, text_layer(im.size, '투  헬', font('nEB', 18), fill=(20, 20, 140, 255), xy=(322, 159)))
    # subtitle ～地獄少女物語～ : rebuild area by inpainting everything that is not background
    x0, y0, x1, y1 = 182, 186, 470, 220
    a = np.array(im).astype(int); r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    sub = ((r > 200) & (g > 60) & (b < 120)) | ((b > 80) & (r < 90) & (g < 90)) | ((r > 200) & (g > 180) & (b < 200) & (b > 120))
    m = np.zeros(a.shape[:2], np.uint8); m[y0:y1, x0:x1] = sub[y0:y1, x0:x1]
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    im = inpaint(im, Image.fromarray(m * 255), 5)
    txt = '~지옥소녀 이야기~'
    f = font('nH', 27)
    base = text_layer(im.size, txt, f, fill=(255, 255, 255, 255), xy=(326, 203), stroke=3, stroke_fill=(20, 20, 120, 255))
    # vertical gradient fill (pink->orange->yellow like original)
    grad = np.zeros((im.size[1], im.size[0], 4), np.uint8)
    for y in range(im.size[1]):
        t = np.clip((y - 192) / 22, 0, 1)
        grad[y, :, :3] = (np.array([255, 120, 20]) * (1 - t) + np.array([255, 235, 60]) * t).astype(np.uint8)
    grad[:, :, 3] = 255
    inner = text_layer(im.size, txt, f, fill=(255, 255, 255, 255), xy=(326, 203))
    ga = Image.fromarray(grad, 'RGBA'); ga.putalpha(inner.getchannel('A'))
    outline = text_layer(im.size, txt, f, fill=(20, 20, 120, 255), xy=(326, 203), stroke=3, stroke_fill=(20, 20, 120, 255))
    im = over(over(im, outline), ga)
    save('740__0', im)


if __name__ == '__main__':
    for k, (t, b) in CHAP.items(): chapter(k, t, b)
    for k, t in NAMES.items(): namecard(k, t)
    for k in ['362__0', '372__0', '374__0', '725__0']: site(k)
    desk()
    seal_img('400__0', (19, 19, 85, 148))
    seal_img('727__0', (22, 67, 71, 164))
    tohell()
    os.makedirs(PREV, exist_ok=True)
    sheet(list(CHAP), os.path.join(PREV, 's_chap.png'), 1)
    sheet(list(NAMES)[:5], os.path.join(PREV, 's_name1.png'), 1)
    sheet(list(NAMES)[5:], os.path.join(PREV, 's_name2.png'), 1)
    sheet(['362__0', '372__0', '725__0'], os.path.join(PREV, 's_site.png'), 1)
    sheet(['726__0', '400__0', '727__0', '740__0'], os.path.join(PREV, 's_misc.png'), 1)
