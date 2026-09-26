"""PSS (MPEG-PS) helpers: iterate PES packets, extract video ES / SShd-SSbd PCM audio, write video ES back in place."""
import struct
import numpy as np

def packets(d):
    """yield (sid, payload_start, payload_end) for each PES packet"""
    i = 0
    while i < len(d) - 4:
        if d[i:i + 3] != b'\0\0\1': i += 1; continue
        sid = d[i + 3]
        if sid == 0xba: i += 14 + (d[i + 13] & 7); continue
        if sid == 0xb9: break
        ln = struct.unpack('>H', d[i + 4:i + 6])[0]
        if sid in (0xe0, 0xbd):
            hl = d[i + 8]; ps = i + 9 + hl
            if sid == 0xbd: ps += 4           # substream id + 3 bytes
            yield sid, ps, i + 6 + ln
        i += 6 + ln

def video_es(d):
    return b''.join(d[a:b] for s, a, b in packets(d) if s == 0xe0)

def put_video_es(d, es):
    d = bytearray(d); o = 0
    for s, a, b in packets(bytes(d)):
        if s == 0xe0: d[a:b] = es[o:o + b - a]; o += b - a
    assert o == len(es), (o, len(es))
    return bytes(d)

def audio(d):
    """-> (int16 array [n, ch], rate)"""
    raw = b''.join(d[a:b] for s, a, b in packets(d) if s == 0xbd)
    assert raw[:4] == b'SShd'
    hl = struct.unpack_from('<I', raw, 4)[0]
    typ, rate, ch, inter = struct.unpack_from('<IIII', raw, 8)
    body = raw[8 + hl:]
    assert body[:4] == b'SSbd'
    pcm = body[8:]
    blk = inter * ch; n = len(pcm) // blk
    a = np.frombuffer(pcm[:n * blk], '<i2').reshape(n, ch, inter // 2)
    return a.transpose(0, 2, 1).reshape(-1, ch), rate
