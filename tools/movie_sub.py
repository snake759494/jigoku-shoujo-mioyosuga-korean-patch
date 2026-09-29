"""동영상(DATA/*.PSS)에 한국어 자막을 입힌다 (블랙 한글패치 방식).

python tools/movie_sub.py [이름 ...]   입력 movie/orig/<이름>.PSS + movie/subs/<이름>.tsv -> movie/enc/<이름>.PSS

원본: MPEG-PS, 비디오 720x480 29.97fps 인터레이스(상위 필드 먼저), GOP 15장, VBV 1,835,008비트, 음성 SShd PCM.
방법: 자막이 걸린 GOP 구간만 디코드 -> 자막 합성 -> 닫힌 GOP로 재인코딩. 구간 바이트가 원본 이하가 되는 가장 고운 q를 쓰고
      남는 바이트는 0으로 채워(시작 코드 앞 0 채움은 규격상 허용) 비디오 ES 길이를 원본과 똑같이 맞춘 뒤,
      PES 패킷 자리에 그대로 되돌려 쓴다. 팩 구조·음성·파일 크기는 원본과 같다.
자막: 게임 대사와 같은 서울한강 B, 대사 글자 크기(한글 높이 약 20px, 화면 640폭 기준), 흰 글자 + 검은 외곽선, 화면 아래 가운데.
     영상은 720폭이 화면 640폭으로 표시되므로 640폭에서 그린 뒤 가로로 720/640 늘려 합성한다.
tsv: 시작초 <TAB> 끝초 <TAB> 한국어 (줄바꿈은 \\n), # 줄은 주석
"""
import os, sys, subprocess
import numpy as np
import av
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
sys.path.insert(0, os.path.dirname(__file__))
import pss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, 'movie')
FONT = os.path.join(ROOT, 'SeoulHangangB.ttf')
SIZE = 22            # 한글 높이 약 20px = 게임 대사 글리프와 같음
STROKE = 2
LINE_H = 26
BOTTOM = 480 - 40    # 마지막 줄 아래 끝 (TV 안전 영역 안)
W, H = 720, 480
FPS = 30000 / 1001
GOP = 15
Q_STEPS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 24, 28]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_font = ImageFont.truetype(FONT, SIZE)


def render(text):
    im = Image.new('RGBA', (640, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    lines = text.split('\\n'); y = BOTTOM - LINE_H * len(lines)
    for ln in lines:
        d.text((320, y + LINE_H / 2), ln, font=_font, anchor='mm', fill=(255, 255, 255, 255),
               stroke_width=STROKE, stroke_fill=(0, 0, 0, 255))
        y += LINE_H
    return im.resize((W, H), Image.LANCZOS)


def load_subs(name):
    subs = []
    for ln in open(os.path.join(M, 'subs', name + '.tsv'), encoding='utf8').read().splitlines():
        if ln.strip() and not ln.startswith('#'):
            s, e, t = ln.split('\t', 2); subs.append((float(s), float(e), t))
    return subs


def split_gops(es):
    starts, i = [], 0
    while True:
        i = es.find(b'\x00\x00\x01\xb3', i)
        if i < 0: break
        starts.append(i); i += 4
    end = len(es) - 4 if es.endswith(b'\x00\x00\x01\xb7') else len(es)
    return [(s, (starts[k + 1] if k + 1 < len(starts) else end)) for k, s in enumerate(starts)]


def gop_frames(es, gops):
    return [es.count(b'\x00\x00\x01\x00', a, b) for a, b in gops]


def encode(frames, q, maxrate):
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', '%dx%d' % (W, H), '-r', '30000/1001', '-i', '-',
           '-c:v', 'mpeg2video', '-pix_fmt', 'yuv420p', '-qscale:v', str(q), '-qmin', '1', '-qmax', '28',
           '-g', str(GOP), '-bf', '2', '-flags', '+ildct+ilme', '-top', '1', '-sc_threshold', '1000000000',
           '-alternate_scan', '1', '-intra_vlc', '1', '-non_linear_quant', '1', '-dc', '9',
           '-maxrate', str(maxrate), '-bufsize', '1835008', '-aspect', '4:3', '-seq_disp_ext', 'never',
           '-f', 'mpeg2video', '-']
    r = subprocess.run(cmd, input=b''.join(f.tobytes() for f in frames), capture_output=True)
    if r.returncode: raise RuntimeError(r.stderr.decode())
    es = r.stdout
    return es[:-4] if es.endswith(b'\x00\x00\x01\xb7') else es


def bitrate(es):
    i = es.find(b'\x00\x00\x01\xb3'); h = es[i + 4:i + 12]
    return (h[4] << 10 | h[5] << 2 | h[6] >> 6) * 400


def pic_starts(es):
    r, i = [], 0
    while True:
        i = es.find(b'\x00\x00\x01\x00', i)
        if i < 0: return r
        r.append(i); i += 4


def align(orig, enc, lat=0):
    """Pad `enc` with zeros so every coded picture starts at the same byte offset as in `orig`
    (and the total length is equal). The PS2 stream player feeds video/audio in mux order, so picture
    data must arrive neither later nor far earlier than the original. A picture may run at most `lat`
    bytes past its original end (re-aligned at the next picture). None if that is exceeded."""
    po, pn = pic_starts(orig), pic_starts(enc)
    if len(po) != len(pn): return None
    bo, bn = po + [len(orig)], pn + [len(enc)]
    out = bytearray(enc[:pn[0]])
    if len(out) > po[0] + lat: return None
    out += b'\x00' * max(0, po[0] - len(out))
    for k in range(len(pn)):
        out += enc[bn[k]:bn[k + 1]]
        if len(out) > bo[k + 1] + (lat if k + 1 < len(pn) else 0): return None
        out += b'\x00' * max(0, bo[k + 1] - len(out))
    return bytes(out)


def process(name):
    src = open(os.path.join(M, 'orig', name + '.PSS'), 'rb').read()
    es = pss.video_es(src); gops = split_gops(es); nf = gop_frames(es, gops)
    first = np.cumsum([0] + nf); total = int(first[-1])
    sub_at = [None] * total
    for s, e, t in load_subs(name):
        for fi in range(int(round(s * FPS)), min(total, int(round(e * FPS)))): sub_at[fi] = t
    hit = [any(sub_at[first[g]:first[g + 1]]) for g in range(len(gops))]
    runs, g = [], 0
    while g < len(gops):
        if hit[g]:
            h = g
            while h + 1 < len(gops) and hit[h + 1]: h += 1
            h = min(h + 1, len(gops) - 1)   # next GOP is open (leading B refs our last P): re-encode it too
            runs.append((g, h)); g = h + 1
        else: g += 1
    # decode every frame (subtitles composited) - each run is encoded from the start of the stream so that
    # ffmpeg's open-GOP layout (first GOP 13 pictures, then IBBPBBPBBPBBPBB) matches the original exactly
    last = min(total, int(first[min(len(gops), runs[-1][1] + 3)])) if runs else 0
    frames, cache = [], {}
    tmp = os.path.join(M, 'enc', name + '.m2v'); os.makedirs(os.path.dirname(tmp), exist_ok=True)
    open(tmp, 'wb').write(es)
    c = av.open(tmp)
    for i, fr in enumerate(c.decode(video=0)):
        if i >= last: break
        img = fr.to_ndarray(format='rgb24'); t = sub_at[i]
        if t:
            if t not in cache: cache[t] = render(t)
            base = Image.fromarray(img).convert('RGBA'); base.alpha_composite(cache[t])
            img = np.asarray(base.convert('RGB'))
        frames.append(img)
    c.close(); os.remove(tmp)
    out = bytearray(); prev = 0; qs = []
    for a, b in runs:
        out += es[prev:gops[a][0]]
        budget = gops[b][1] - gops[a][0]
        orig = es[gops[a][0]:gops[b][1]]
        n_in = min(len(frames), int(first[min(len(gops), b + 3)]))
        aligned = None; encs = {}
        for lat in (0, 16384, 65536):
            for q in Q_STEPS:
                if q not in encs:
                    full = encode(frames[:n_in], q, bitrate(es)); ge = split_gops(full)
                    assert gop_frames(full, ge)[a:b + 1] == nf[a:b + 1], (name, a, b, 'gop layout differs')
                    encs[q] = full[ge[a][0]:ge[b][1]]
                enc = encs[q]
                if len(enc) <= budget: aligned = align(orig, enc, lat)
                if aligned is not None: break
            if aligned is not None: break
        assert aligned is not None, (name, a, b)
        assert len(aligned) == budget
        out += aligned; prev = gops[b][1]; qs.append((q, lat))
    out += es[prev:]
    assert len(out) == len(es)
    dst = pss.put_video_es(src, bytes(out))
    assert len(dst) == len(src)
    open(os.path.join(M, 'enc', name + '.PSS'), 'wb').write(dst)
    print(name, 'runs', len(runs), 'q', qs, flush=True)


def preview(name, t, path):
    tmp = path + '.m2v'; open(tmp, 'wb').write(pss.video_es(open(os.path.join(M, 'enc', name + '.PSS'), 'rb').read()))
    c = av.open(tmp)
    for i, fr in enumerate(c.decode(video=0)):
        if i == int(t * FPS): fr.to_image().resize((640, 480)).save(path); break
    c.close(); os.remove(tmp)


if __name__ == '__main__':
    for n in sys.argv[1:] or ['MOVIE_02', 'MOVIE_04']: process(n)
