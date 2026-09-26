"""T2FP font: 'T2FP' u16 hdrsize(0x20) u16 ? u8 13 u8 28 u8 w u8 h u32 0x20 (code table) u32 clut_off u32 glyph_off;
code table = sorted u16 char codes (SJIS as big-endian value, <0x100 single byte); glyph i = 4bpp w*h at glyph_off+i*w*h/2 (low nibble = left pixel)."""
import struct
class Font:
    def __init__(s,path):
        s.d=bytearray(open(path,'rb').read())
        s.w,s.h=s.d[10],s.d[11]
        s.clut,s.goff=struct.unpack_from('<II',s.d,0x10)
        s.n=(s.clut-0x20)//2
        s.codes=list(struct.unpack_from('<%dH'%s.n,s.d,0x20)); s.idx={c:i for i,c in enumerate(s.codes)}
        s.gs=s.w*s.h//2
    def get(s,i):
        o=s.goff+i*s.gs; out=[]
        for y in range(s.h):
            row=[]
            for x in range(s.w):
                b=s.d[o+y*s.w//2+x//2]; row.append((b>>4) if x&1 else (b&15))
            out.append(row)
        return out
    def put(s,i,rows):
        o=s.goff+i*s.gs
        for y in range(s.h):
            for x in range(0,s.w,2):
                s.d[o+y*s.w//2+x//2]=(rows[y][x]&15)|((rows[y][x+1]&15)<<4)
    def save(s,path): open(path,'wb').write(s.d)
