"""Find translatable SJIS strings in the ELF data area; slot = bytes up to next non-NUL byte (in-place capacity)."""
import re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELF = os.path.join(ROOT, 'extract/SLPM_552.13')
PAT = re.compile(rb'(?<=\0)((?:[\x81-\x9f\xe0-\xef][\x40-\x7e\x80-\xfc]|[\x20-\x7e\n%])+)\0')
def scan(d):
    out = []
    for m in PAT.finditer(d, 0xf9000):
        s = m.group(1)
        if m.start() >= 0x106000: break
        try: t = s.decode('cp932')
        except UnicodeDecodeError: continue
        if not re.search('[ぁ-んァ-ヶ一-龥々＿？]', t): continue
        if len(s) == 2 and m.start() > 0x105000 and False: pass
        e = m.end()
        while e < len(d) and d[e] == 0: e += 1
        out.append((m.start(), e - m.start() - 1, t))
    return out
if __name__ == '__main__':
    d = open(ELF, 'rb').read()
    with open(os.path.join(ROOT, 'translation/elf_jp.tsv'), 'w', encoding='utf-8', newline='\n') as o:
        for off, cap, t in scan(d):
            o.write('%X\t%d\t%s\n' % (off, cap, t.replace('\n', '\\n')))
