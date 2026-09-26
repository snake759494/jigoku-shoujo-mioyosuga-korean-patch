"""TIM2 decode/encode (single picture per file supported for writing) and PTD000 container walker."""
import struct
from PIL import Image

def _unswz(clut):
    # CSM1 256-colour CLUT: swap blocks 8..15 with 16..23 in every 32
    out=list(clut)
    for i in range(len(clut)):
        j=i
        if (i & 0x18)==0x08: j=i+8
        elif (i & 0x18)==0x10: j=i-8
        out[i]=clut[j]
    return out

def _col(b,o,fmt):
    if fmt==3: r,g,bb,a=b[o:o+4]; return (r,g,bb,min(255,a*2))
    if fmt==2: r,g,bb=b[o:o+3]; return (r,g,bb,255)
    v=b[o]|b[o+1]<<8
    return ((v&31)<<3,(v>>5&31)<<3,(v>>10&31)<<3,255 if v&0x8000 else 0)

def pictures(b):
    """yield dict per picture in a TIM2 blob"""
    assert b[:4]==b'TIM2'
    n=struct.unpack_from('<H',b,6)[0]; o=16 if b[5]==0 else 128
    for _ in range(n):
        tot,cs,isz,hs,cc,pf,mip,ct,it,w,h=struct.unpack_from('<IIIHHBBBBHH',b,o)
        yield dict(off=o,total=tot,clut_size=cs,img_size=isz,hdr=hs,colors=cc,clut_type=ct,img_type=it,w=w,h=h,
                   img_off=o+hs,clut_off=o+hs+isz)
        o+=tot

def decode(b,p=None):
    p=p or next(pictures(b))
    w,h,it=p['w'],p['h'],p['img_type']; io=p['img_off']
    if it in (1,2,3):
        bpp={1:2,2:3,3:4}[it]
        px=[_col(b,io+i*bpp,it) for i in range(w*h)]
    else:
        ct=p['clut_type']&0x3f; cb={1:2,2:3,3:4}[ct]
        clut=[_col(b,p['clut_off']+i*cb,ct) for i in range(p['colors'])]
        if it==5 and not (p['clut_type']&0x80) and len(clut)>=256: clut=_unswz(clut[:256])+clut[256:]
        if it==5: idx=b[io:io+w*h]
        else: idx=[(b[io+i//2]>>(4*(i&1)))&15 for i in range(w*h)]
        px=[clut[i] for i in idx]
    im=Image.new('RGBA',(w,h)); im.putdata(px); return im

def walk(b,path=()):
    """yield (path, offset, length) of each TIM2 inside nested PTD000 containers"""
    if b[:4]==b'TIM2': yield path,0,len(b); return
    if len(b)<8: return
    n=struct.unpack_from('<I',b,0)[0]
    if not 0<n<4096 or 4+4*(n+1)>len(b): return
    offs=struct.unpack_from('<%dI'%(n+1),b,4)
    if offs[-1]!=len(b) or any(offs[i]>offs[i+1] for i in range(n)) or offs[0]<4+4*(n+1): return
    for i in range(n):
        for p,o,l in walk(b[offs[i]:offs[i+1]],path+(i,)):
            yield p,offs[i]+o,l

def _enc_col(c, fmt):
    r, g, b, a = c
    if fmt == 3: return bytes((r, g, b, (a + 1) // 2))
    if fmt == 2: return bytes((r, g, b))
    v = (r >> 3) | (g >> 3) << 5 | (b >> 3) << 10 | (0x8000 if a >= 128 else 0)
    return bytes((v & 255, v >> 8))

def encode_into(b, p, im):
    """write RGBA image `im` into TIM2 blob `b` (bytearray) picture `p` in its original format"""
    w, h, it = p['w'], p['h'], p['img_type']; io = p['img_off']
    im = im.convert('RGBA'); assert im.size == (w, h), (im.size, w, h)
    px = list(im.getdata())
    if it in (1, 2, 3):
        bpp = {1: 2, 2: 3, 3: 4}[it]
        b[io:io + w * h * bpp] = b''.join(_enc_col(c, it) for c in px)
        return
    # paletted: requantize. CLUT index 0 is the game's transparent key (e.g. pure green), so keep the original
    # index-0 colour at index 0 and map every pixel of that exact colour to it.
    ncol = 256 if it == 5 else 16
    ct = p['clut_type'] & 0x3f; cb = {1: 2, 2: 3, 3: 4}[ct]
    swz = it == 5 and not (p['clut_type'] & 0x80)
    old = [_col(b, p['clut_off'] + i * cb, ct) for i in range(ncol)]
    if swz: old = _unswz(old)
    key = old[0]
    iskey = [c[:3] == key[:3] for c in px]
    rest = Image.new('RGBA', (w, h)); rest.putdata([c if not k else px[0] for c, k in zip(px, iskey)])
    q = rest.quantize(colors=ncol - 1, method=Image.Quantize.FASTOCTREE)
    qp = q.getpalette('RGBA'); pal = [key] + [tuple(qp[i * 4:i * 4 + 4]) for i in range(ncol - 1)]
    idx = [0 if k else v + 1 for v, k in zip(q.getdata(), iskey)]
    clut = _unswz(pal) if swz else pal  # swizzle is its own inverse
    b[p['clut_off']:p['clut_off'] + ncol * cb] = b''.join(_enc_col(c, ct) for c in clut)
    if it == 5: b[io:io + w * h] = bytes(idx)
    else: b[io:io + w * h // 2] = bytes(idx[i] | idx[i + 1] << 4 for i in range(0, w * h, 2))
