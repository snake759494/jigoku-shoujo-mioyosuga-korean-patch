"""700: speaker name plate (even index, black letters) + tile backing (odd index, one tile per letter),
50..59: alternate flower-tile backings for names 0..9. The game draws the plate over its tile image, so both
must agree: one syllable centred in each tile. Max 5 tiles (texture 180px), longer names use the given name."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from PIL import Image, ImageDraw
from imgdraw import font, orig, save
import imglib

NAMES = ['유즈키', '아스카', '리쿠도루이', '하루토', '아유미', '이시모토렌', '엔마아이', '소네안나', '후와류도',
         '야마와로', '키쿠리', '키비츠', '미사키', '토모히코', '노부히코', '야요이코', '미야', '마사하루', '효마',
         '무라카미', '이치모쿠렌', '호네온나', '와뉴도', '???', '일동']
TILES = [(0, 32), (35, 68), (71, 103), (107, 139), (143, 175)]
SIZE = 24

def backing(src_key, n):
    """keep the first n tiles of the 5-tile original, fill the rest with its key colour (read at the right edge)"""
    a = np.array(imglib.load(src_key).convert('RGBA'))
    kc = a[15, 178, :3].copy()
    x0 = TILES[n - 1][1] if n < 5 else 180
    a[:, x0:, :3] = kc
    return Image.fromarray(a, 'RGBA')

def plate(text, h):
    im = Image.new('RGBA', (180, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); f = font('B', SIZE)
    for (a, b), c in zip(TILES, text):
        d.text(((a + b) / 2, h / 2), c, font=f, fill=(0, 0, 0, 255), anchor='mm')
    return im

def do_plates():
    keys = []
    full = imglib.load('700_1_0'); flower = imglib.load('700_50_0')
    for i, name in enumerate(NAMES):
        n = len(name); assert n <= 5, name
        ok = orig('700_%d_0' % (2 * i)); save('700_%d_0' % (2 * i), plate(name, ok.height)); keys.append('700_%d_0' % (2 * i))
        save('700_%d_0' % (2 * i + 1), backing('700_1_0', n))
        if i < 10: save('700_%d_0' % (50 + i), backing('700_50_0', n))
    return keys
