"""PTD002 script: header u16[8] = magic 0x10, code_off, tbl_off, txt_off, 0x1ff, code_len, n_str, txt_len.
Text = n_str u16 offsets (relative to txt_off) -> NUL-terminated SJIS strings."""
import struct
def parse(d):
    h=list(struct.unpack_from('<8H',d,0))
    if h[0]!=0x10: return None
    tbl,txt,n=h[2],h[3],h[6]
    offs=struct.unpack_from('<%dH'%n,d,tbl)
    strs=[]
    for o in offs:
        e=d.index(b'\0',txt+o); strs.append(d[txt+o:e])
    return h,offs,strs
def build(d,strs):
    h,offs,_=parse(d)
    txt=b''; no=[]; seen={}
    for s in strs:
        if s in seen: no.append(seen[s]); continue
        seen[s]=len(txt); no.append(len(txt)); txt+=s+b'\0'
    assert len(txt)<0x10000, 'text section overflow %d'%len(txt)
    out=bytearray(d[:h[3]]); struct.pack_into('<%dH'%len(no),out,h[2],*no)
    out+=txt; h[7]=len(txt); struct.pack_into('<8H',out,0,*h)
    return bytes(out)
