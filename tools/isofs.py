"""Minimal in-place ISO9660 file replacement: overwrite a file's extent, or append it at the end of the image
when it grows, then fix the directory record (extent/size, both endians) and the PVD volume size."""
import os, struct, shutil

SECTOR = 2048

class Iso:
    def __init__(self, path):
        self.f = open(path, 'r+b')
        self.f.seek(16 * SECTOR); pvd = self.f.read(SECTOR)
        assert pvd[1:6] == b'CD001'
        self.root = pvd[156:156 + 34]

    def _dir(self, lba, size):
        self.f.seek(lba * SECTOR); data = self.f.read(size); o = 0
        while o < len(data):
            n = data[o]
            if n == 0: o = (o // SECTOR + 1) * SECTOR; continue
            yield lba * SECTOR + o, data[o:o + n]; o += n

    def find(self, path):
        """return (record file offset, extent lba, size)"""
        rec = self.root
        for part in path.strip('/').split('/'):
            lba, size = struct.unpack_from('<I', rec, 2)[0], struct.unpack_from('<I', rec, 10)[0]
            for off, r in self._dir(lba, size):
                name = r[33:33 + r[32]].decode('ascii', 'replace').split(';')[0]
                if name.upper() == part.upper(): rec = r; recoff = off; break
            else: raise FileNotFoundError(path)
        return recoff, struct.unpack_from('<I', rec, 2)[0], struct.unpack_from('<I', rec, 10)[0]

    def read(self, path):
        _, lba, size = self.find(path); self.f.seek(lba * SECTOR); return self.f.read(size)

    def replace(self, path, data, capacity=None):
        """capacity = sectors available at the current extent (default: original size rounded up)"""
        recoff, lba, size = self.find(path)
        cap = capacity if capacity is not None else (size + SECTOR - 1) // SECTOR
        if (len(data) + SECTOR - 1) // SECTOR > cap:
            self.f.seek(0, 2); end = self.f.tell(); lba = (end + SECTOR - 1) // SECTOR
        self.f.seek(lba * SECTOR); self.f.write(data)
        pad = (-len(data)) % SECTOR
        self.f.write(b'\0' * pad)
        self.f.seek(recoff + 2); self.f.write(struct.pack('<I', lba) + struct.pack('>I', lba))
        self.f.write(struct.pack('<I', len(data)) + struct.pack('>I', len(data)))
        self.f.seek(0, 2); total = self.f.tell() // SECTOR
        self.f.seek(16 * SECTOR + 80); self.f.write(struct.pack('<I', total) + struct.pack('>I', total))
        return lba

    def close(self): self.f.close()
