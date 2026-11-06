#!/usr/bin/env python3
# Audit VGPR/SGPR/LDS per kernel from HIP-compiled ELF objects (amdgpu code object v4+).
# No ROCm tooling needed: parses ELF notes, decodes the msgpack kernel metadata.
import struct, sys, os, glob, re

# minimal msgpack decoder
class Dec:
    def __init__(self, d, i):
        self.d, self.i = d, i
    def __call__(self):
        d, i = self.d, self.i
        c = d[i]
        self.i += 1
        if c < 0x80: return c
        if c >= 0xe0: return c - 256
        if 0x80 <= c <= 0x8f: return self.map(c & 0xf)
        if 0x90 <= c <= 0x9f: return self.arr(c & 0xf)
        if 0xa0 <= c <= 0xbf: return self.str(c & 0x1f, str)
        if c == 0xc0: return None
        if c == 0xc2: return False
        if c == 0xc3: return True
        if c == 0xc4: return self.str(1, bytes)
        if c == 0xc5: return self.l(2, bytes)
        if c == 0xc6: return self.l(4, bytes)
        if c == 0xca: self.i += 4; return struct.unpack('>f', d[self.i-4:self.i])[0]
        if c == 0xcb: self.i += 8; return struct.unpack('>d', d[self.i-8:self.i])[0]
        if c == 0xcc: self.i += 1; return d[self.i-1]
        if c == 0xcd: self.i += 2; return int.from_bytes(d[self.i-2:self.i],'big')
        if c == 0xce: self.i += 4; return int.from_bytes(d[self.i-4:self.i],'big')
        if c == 0xcf: self.i += 8; return int.from_bytes(d[self.i-8:self.i],'big')
        if c == 0xd0: self.i += 1; return d[self.i-1] - 256
        if c == 0xd1: self.i += 2; n = int.from_bytes(d[self.i-2:self.i],'big'); return n-65536 if n >= 32768 else n
        if c == 0xd2: self.i += 4; n = int.from_bytes(d[self.i-4:self.i],'big'); return n-2**32 if n >= 2**31 else n
        if c == 0xd3: self.i += 8; n = int.from_bytes(d[self.i-8:self.i],'big'); return n-2**64 if n >= 2**63 else n
        if 0xd4 <= c <= 0xd8: n = [1,2,4,8,16][c-0xd4]; self.i += 2*n; return None  # fixext, skip
        if 0xd9 <= c <= 0xdb: return self.l([1,2,4][c-0xd9], str)
        if 0xdc <= c <= 0xdd: return self.arr(int.from_bytes(d[self.i:self.i+[2,4][c-0xdc]],'big')) if False else self.arr(self.u([2,4][c-0xdc]))
        if 0xde <= c <= 0xdf: return self.map(self.u([2,4][c-0xde]))
        if c == 0xd8: return self.ext(1)[0]
        self.i = i; raise ValueError('unsupported msgpack byte %02x at %d' % (c, i))
    def u(self, n):
        v = int.from_bytes(self.d[self.i:self.i+n],'big'); self.i += n; return v
    def l(self, n, t):
        m = self.u(n); v = self.d[self.i:self.i+m]; self.i += m
        return v.decode('utf-8', 'replace') if t is str else v
    def str(self, n, t):
        v = self.d[self.i:self.i+n]; self.i += n
        return v.decode('utf-8', 'replace') if t is str else v
    def ext(self, n):
        self.i += 1; m = self.u(n); self.i += m; return None
    def arr(self, n):
        return [self() for _ in range(n)]
    def map(self, n):
        out = {}
        for _ in range(n):
            k = self(); out[k] = self()
        return out

def note_sections(path):
    d = open(path, 'rb').read()
    if d[:4] != b'\x7fELF': raise ValueError('not ELF: ' + path)
    is64 = d[4] == 2
    if is64:
        shoff, = struct.unpack_from('<Q', d, 0x28)
        shent, shnum = struct.unpack_from('<HH', d, 0x3a)
    else:
        shoff, = struct.unpack_from('<I', d, 0x20)
        shent, shnum = struct.unpack_from('<HH', d, 0x2e)
    hdr = []
    strtab_off, strtab_size = None, 0
    for n in range(shnum):
        o = shoff + n*shent
        if is64:
            nameoff, typ, off, size = struct.unpack_from('<IIQQ', d, o)
        else:
            nameoff, typ = struct.unpack_from('<II', d, o)
            off, size = struct.unpack_from('<II', d, o+16)
        hdr.append((nameoff, typ, off, size))
        if typ == 3:
            strtab_off, strtab_size = off, size
    namesec = d[strtab_off:strtab_off+strtab_size] if strtab_off is not None else b''
    out = []
    for nameoff, typ, off, size in hdr:
        if typ != 7: continue
        blob = d[off:off+size]
        pos = 0
        while pos + 12 <= len(blob):
            namesz, descsz, ntype = struct.unpack_from('<III', blob, pos)
            nm = blob[pos+12:pos+12+namesz].rstrip(b'\0')
            desc = blob[pos+12+(namesz+3)//4*4 : pos+12+(namesz+3)//4*4+descsz]
            pad = 8 if (namesz % 8 == 0 and descsz % 8 != 0) else 4
            pos = pos+12+(namesz+pad-1)//pad*pad+(descsz+pad-1)//pad*pad
            if pos > len(blob): break
            out.append((nm.decode('ascii','replace'), ntype, desc))
    return out

def elf_note_walk(d, base):
    r = lambda fmt, off: struct.unpack_from(fmt, d, base+off)[0]
    magic = d[base:base+4]
    if magic != b'\x7fELF' or d[base+4] != 2 or r('H', 0x12) != 224: return None  # 64-bit amdgcn only
    shoff = r('Q', 0x28); shent = r('H', 0x3a); shnum = r('H', 0x3c)
    if shoff == 0 or shent != 64 or shoff + shent*shnum > len(d) - base: return None
    hdr = []
    strtab_off = None
    for n in range(shnum):
        o = shoff + n*shent
        nameoff, typ = struct.unpack_from('<II', d, base+o)
        soff, size = struct.unpack_from('<QQ', d, base+o+24)
        hdr.append((nameoff, typ, soff, size))
        if typ == 3: strtab_off = soff
    out = []
    for nameoff, typ, soff, size in hdr:
        if typ != 7: continue
        blob = d[base+soff:base+soff+size]
        pos = 0
        while pos + 12 <= len(blob):
            namesz, descsz, ntype = struct.unpack_from('<III', blob, pos)
            end = pos + 12 + ((namesz+3)//4)*4 + ((descsz+3)//4)*4
            if end > len(blob) + 8: break
            desc = blob[pos+12+((namesz+3)//4)*4 : pos+12+((namesz+3)//4)*4+descsz]
            pos = end
            out.append((ntype, desc))
    return out

def kernels_from(obj):
    d = open(obj, 'rb').read()
    res, seen = [], set()
    starts = [0]
    i = 0
    while True:
        i = d.find(b'\x7fELF', i+1)
        if i == -1: break
        starts.append(i)
    for s in starts:
        notes = elf_note_walk(d, s)
        if not notes: continue
        for ntype, desc in notes:
            if ntype != 32: continue  # amdgpu metadata note
            try:
                doc = Dec(desc, 0)()
            except Exception:
                continue
            ks = doc.get('amdhsa.kernels') if isinstance(doc, dict) else None
            if not ks: continue
            for k in ks:
                if not isinstance(k, dict): continue
                sym = k.get('.symbol') or k.get('symbol') or '?'
                row = dict(
                    sym=sym,
                    vgpr=k.get('.vgpr_count',-1), vspill=k.get('.vgpr_spill_count',-1),
                    sgpr=k.get('.sgpr_count',-1), sspill=k.get('.sgpr_spill_count',-1),
                    lds=k.get('.group_segment_fixed_size',-1),
                )
                key = (row['sym'], row['vgpr'], row['sgpr'], row['lds'])
                if key in seen: continue
                seen.add(key)
                res.append(row)
    return res

def waves(vgpr, sgpr):
    wv = min(10, 1024 // vgpr) if vgpr and vgpr > 0 else None
    ws = 800 // sgpr if sgpr and sgpr > 0 else None
    opts = [x for x in (wv, ws) if x]
    return min(opts) if opts else None

def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else 'build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/*.o'
    filt = sys.argv[2].lower() if len(sys.argv) > 2 else ''
    files = sorted(glob.glob(pat))
    if not files: print('no objects match', pat); return
    print('%-10s %-78s %5s %5s %5s %5s %7s %5s' % ('OBJ', 'SYMBOL', 'VGPR', 'SP', 'SGPR', 'SP', 'LDS', 'W/S'))
    seen = set()
    for f in files:
        ks = kernels_from(f)
        if not ks:
            print('%-10s (no kernel metadata found)' % os.path.basename(f.replace('.cu.o','').replace('.cu','')))
            continue
        oname = re.sub(r'\.(cu|cpp)\.o$', '', os.path.basename(f))
        for k in ks:
            s = k['sym']
            s = re.sub(r'^_Z.*(mul_|flash_|rms_|quantize|gcn_)?', '', s) if False else s
            dem = s
            w = waves(k['vgpr'], k['sgpr'])
            row = (oname, dem, k['vgpr'], k['vspill'], k['sgpr'], k['sspill'], k['lds'], w)
            if row in seen: continue
            seen.add(row)
            if filt and filt not in dem.lower(): continue
            print('%-10s %-78s %5s %5s %5s %5s %7s %5s' % row)

if __name__ == '__main__':
    main()
