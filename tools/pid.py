"""PTDALL.PID index: 12 archive counts, names, then per-archive (start_sector, size) entries."""
import struct, pycdlib
ISO='Jigoku Shoujo Mioyosuga (Japan).iso'
def load(path='extract/PTDALL.PID'):
    d=open(path,'rb').read()
    n=struct.unpack_from('<I',d,0)[0]
    counts=struct.unpack_from('<%dI'%n,d,4)
    names=[d[0x34+16*i:0x34+16*i+12].split(b'\0')[0].decode() for i in range(n)]
    off=0xf4; arcs=[]
    for c,nm in zip(counts,names):
        ents=[struct.unpack_from('<II',d,off+8*i) for i in range(c)]
        arcs.append((nm,ents,off)); off+=8*c
    return arcs

def entries(arc):
    """yield (index, bytes) for archive name like 'PTD002.PTD', reading from the ISO."""
    import pycdlib
    iso=pycdlib.PyCdlib(); iso.open(ISO)
    lba=iso.get_record(iso_path='/DATA/'+arc+';1').extent_location(); iso.close()
    f=open(ISO,'rb')
    for nm,ents,_ in load():
        if nm!=arc: continue
        for i,(s,z) in enumerate(ents):
            if z: f.seek((lba+s)*2048); yield i,f.read(z)
