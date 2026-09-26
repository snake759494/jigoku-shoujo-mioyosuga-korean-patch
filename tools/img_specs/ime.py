"""751~756: 지옥통신 이름 입력란 IME 타이핑 애니메이션 -> 한글 2벌식 조합 과정으로 재작성.
python tools/img_specs/ime.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from imgdraw import *
import numpy as np

PREV = 'extract/preview/ime'
FNT = font('nR', 25)
X0, BASE = 17, 41          # text start x, baseline y
INK = (22, 58, 66)

# (committed text, composing text) per keystroke state
def S(*items):
    out = []
    for it in items:
        c, p = it.split('|') if '|' in it else (it, '')
        out.append((c, p))
    return out

JOBS = {
    # fid: (frames, leading empty frames, trailing final frames, states)
    751: (20, 1, 3, S('|ㅁ', '|마', '|맟', '마|츠', '마|츸', '마츠|카', '마츠|캍', '마츠카|타',
                      '마츠카타 |', '마츠카타 |ㅁ', '마츠카타 |미', '마츠카타 |밍', '마츠카타 미|야')),
    752: (23, 2, 3, S('|ㅋ', '|카', '|칸', '칸|ㅂ', '칸|바', '칸|방', '칸바|야', '칸바|얏', '칸바야|시',
                      '칸바야시 |', '칸바야시 |ㅇ', '칸바야시 |아', '칸바야시 |앗', '칸바야시 아|스',
                      '칸바야시 아|슼', '칸바야시 아스|카')),
    753: (17, 1, 3, S('|ㅋ', '|쿠', '|쿳', '쿠|사', '쿠|상', '쿠사|오', '쿠사오 |', '쿠사오 |ㅎ',
                      '쿠사오 |하', '쿠사오 |할', '쿠사오 하|루', '쿠사오 하|룥', '쿠사오 하루|토')),
    754: (22, 1, 3, S('|ㅅ', '|시', '|신', '시|노', '시|놋', '시노|사', '시노|삭', '시노사|키',
                      '시노사키 |', '시노사키 |ㅇ', '시노사키 |아', '시노사키 |앙', '시노사키 아|유',
                      '시노사키 아|윰', '시노사키 아유|미')),
    755: (22, 1, 3, S('|ㅁ', '|마', '|맟', '마|츠', '마|츸', '마츠|카', '마츠|캍', '마츠카|타',
                      '마츠카타 |', '마츠카타 |ㄴ', '마츠카타 |노', '마츠카타 |놉', '마츠카타 노|부',
                      '마츠카타 노|붛', '마츠카타 노부|히', '마츠카타 노부|힠', '마츠카타 노부히|코')),
    756: (20, 1, 3, S('|ㅋ', '|쿠', '|쿳', '쿠|사', '쿠|상', '쿠사|오', '쿠사오 |', '쿠사오 |ㅁ',
                      '쿠사오 |마', '쿠사오 |맛', '쿠사오 마|사', '쿠사오 마|샇', '쿠사오 마사|하',
                      '쿠사오 마사|할', '쿠사오 마사하|루')),
}

def fit(states, m):
    """stretch the keystroke states to m frames; extra frames = pause right after the surname (space)."""
    states = list(states); assert len(states) <= m
    sp = next(i for i, (c, p) in enumerate(states) if c.endswith(' ') and not p)
    while len(states) < m:
        states.insert(sp, states[sp - 1])   # linger on finished surname first
        if len(states) < m: states.insert(sp + 2, states[sp + 1])
    return states

def darken(im, mask, strength):
    """multiply background toward INK where mask(0..1) -- keeps the screen texture like the original"""
    a = np.array(im).astype(float); k = (mask * strength)[:, :, None]
    a[:, :, :3] = a[:, :, :3] * (1 - k) + np.array(INK) * k
    return Image.fromarray(a.clip(0, 255).astype(np.uint8), 'RGBA')

def frame(bg, committed, comp, cursor):
    W, H = bg.size
    t = Image.new('L', (W, H), 0); d = ImageDraw.Draw(t)
    full = committed + comp
    d.text((X0, BASE), full, font=FNT, fill=255, anchor='ls')
    im = darken(bg, np.array(t) / 255.0, 0.9)
    xs = X0 + FNT.getlength(committed); xe = X0 + FNT.getlength(full)
    m = np.zeros((H, W))
    if comp:   # dotted composition underline, as in the original
        for x in range(int(round(xs)), int(round(xe))):
            if (x // 2) % 2 == 0: m[48:50, x] = 1
    if cursor:
        cx = int(round(xe)) + 1
        m[14:48, cx:cx + 2] = 1; m[14:48, cx - 1] = np.maximum(m[14:48, cx - 1], .15); m[14:48, cx + 2] = .15
    return darken(im, m, 0.45)

def main():
    os.makedirs(PREV, exist_ok=True)
    bg = orig('752_1-0_0')       # empty field, no cursor
    for fid, (n, lead, tail, states) in JOBS.items():
        mid = fit(states, n - lead - tail)
        final = (states[-1][0] + states[-1][1], '')
        seq = [('', '')] * lead + mid + [final] * tail
        keys = []
        for i, (c, p) in enumerate(seq):
            k = '%d_1-%d_0' % (fid, i)
            cur = not (i == n - 1 or (fid == 752 and i == 0))
            save(k, frame(bg, c, p, cur)); keys.append(k)
        sheet(keys, PREV + 'cmp_%d.png' % fid, scale=1)

if __name__ == '__main__':
    main()
