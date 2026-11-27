#!/usr/bin/env python3
"""Static MMQ config sweep for gfx906: compile one instance TU per cell and
read VGPR / SGPR / spill from the amdgpu code-object metadata (amdgcn-audit.py).
No GPU, no timing. LDS is computed from the same formula the launcher uses.

The table in ggml/src/ggml-cuda/mmq-config-vega.cuh is patched per cell and
restored when the run ends. See bench/mmq/RESULTS.md for the findings."""

import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = '/home/otis/Dokumente/Build/llama.cpp-MI50'
CFG  = ROOT + '/ggml/src/ggml-cuda/mmq-config-vega.cuh'
OBJ  = '/tmp/sweep/obj'
CSV  = ROOT + '/bench/mmq/static-sweep.csv'
TYPES = ['Q4_0', 'Q8_0', 'Q4_K', 'Q5_K', 'Q6_K']

spec = importlib.util.spec_from_file_location('audit', ROOT + '/amdgcn-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

# ggml_type enum value -> name
ENUM = {}
for m in re.finditer(r'GGML_TYPE_(\w+)\s*=\s*(\d+),', open(ROOT + '/ggml/include/ggml.h').read()):
    ENUM[int(m.group(2))] = m.group(1)

# ---------------------------------------------------------------- compile cmds
CC = {}
for e in json.load(open(ROOT + '/build/compile_commands.json')):
    m = re.search(r'mmq-instance-(\w+)\.cu$', e['file'])
    if m:
        CC[m.group(1)] = (e['directory'], e['command'].split())

def compile_tu(typ, out):
    d, argv = CC[typ.lower()]
    argv = list(argv)
    i = argv.index('-o')
    argv[i + 1] = out
    r = subprocess.run(argv, cwd=d, capture_output=True, text=True)
    return r.returncode, r.stderr[-4000:]

# ---------------------------------------------------------------- config patch
CASE_RE = re.compile(
    r'CASE\((GGML_TYPE_\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*'
    r'(GGML_CUDA_MMQ_SRAM_LAYOUT_\w+),\s*(\w+),\s*(true|false),\s*(true|false)\);')

def patch(nthreads, occ, I, k_vram):
    src = open(CFG).read()
    n = 0
    def rep(m):
        nonlocal n
        typ, nt, oc, ii, j, layout, kv, sk, fb = m.groups()
        if typ not in ['GGML_TYPE_' + t for t in TYPES]:
            return m.group(0)
        n += 1
        kv_new = 'MMQ_ITER_K' if k_vram == 256 else str(k_vram)
        return 'CASE(%-19s %3d, %d, %3d, %3d, %s, %-8s, %5s, %s);' % (
            typ + ',', nthreads, occ, I, int(j), layout, kv_new, sk, fb)
    open(CFG, 'w').write(CASE_RE.sub(rep, src))
    return n

# ---------------------------------------------------------------- LDS model
QIK = 32
def tile_x_sizes(typ, I):
    T = QIK
    if typ in ('Q4_0',):
        return I*T + I, I*T//4 + I//4, 0
    if typ in ('Q4_1',):
        return I*T + I, I*T//4 + I//4, 0
    if typ in ('Q8_0', 'Q1_0', 'Q2_0', 'Q5_0', 'IQ2_XXS', 'IQ3_XXS', 'IQ3_S', 'IQ1_S', 'IQ4_XS', 'IQ4_NL'):
        return I*T*2 + I, I*T*2//8 + I//4, 0
    if typ in ('IQ2_XS', 'IQ2_S', 'NVFP4'):
        return I*T*2 + I, I*T*4//8 + I//2, 0
    if typ in ('Q5_1', 'Q8_1', 'MXFP4'):
        return I*T*2 + I, I*T*2//8 + I//4, 0
    if typ == 'Q2_K':
        return I*T*2 + I, I*T + I, 0
    if typ == 'Q3_K':
        return I*T*2 + I, I, I*T//8 + I//8
    if typ == 'Q4_K':
        return I*T + I, I*T//32, I*T//8 + I//8
    if typ == 'Q5_K':
        return I*T*2 + I, I*T//32 + I//32, I*T//8 + I//8
    if typ == 'Q6_K':
        return I*T*2 + I, I*T//32 + I//32, I*T//8 + I//8
    raise KeyError(typ)

def lds_bytes(typ, I, J, nthreads):
    qs, dm, sc = tile_x_sizes(typ, I)
    nbs_ids = J*4
    nbs_x   = (qs + dm + sc) * 4
    nbs_y   = J * 144
    pad     = nthreads*4
    return nbs_ids + nbs_x + ((nbs_y + pad - 1)//pad)*pad

# ---------------------------------------------------------------- audit
SYM_RE = re.compile(r'mul_mat_qIL9ggml_type(\d+)ELi(\d+)ELb(\d)EL9ggml_prec(\d+)ELb(\d)EE')

def audit_obj(path):
    rows = {}
    for k in audit.kernels_from(path):
        m = SYM_RE.search(k['sym'])
        if not m:
            continue
        tv, J, fb, prec, fus = (int(x) for x in m.groups())
        typ = ENUM.get(tv, str(tv))
        key = (typ, J, fb, prec, fus)
        # keep the worst metadata seen for a given key
        if key not in rows or k['vgpr'] > rows[key]['vgpr']:
            rows[key] = dict(typ=typ, J=J, fb=fb, prec=prec, fus=fus,
                             vgpr=k['vgpr'], vspill=k['vspill'],
                             sgpr=k['sgpr'], sspill=k['sspill'])
    return rows

# ---------------------------------------------------------------- cells
CELLS = []
for I in (64, 128, 256):
    for nt in (128, 256, 512):
        for occ in (1, 2):
            CELLS.append(dict(I=I, nthreads=nt, occ=occ, k_vram=256))
CELLS.append(dict(I=128, nthreads=256, occ=1, k_vram=512))
CELLS.append(dict(I=64,  nthreads=128, occ=2, k_vram=256))
CELLS.append(dict(I=192, nthreads=256, occ=1, k_vram=256))

def main():
    os.makedirs(OBJ, exist_ok=True)
    orig = open(CFG).read()
    out = []
    try:
        for ci, c in enumerate(CELLS):
            tag = 'I%d_n%d_o%d_k%d' % (c['I'], c['nthreads'], c['occ'], c['k_vram'])
            n = patch(c['nthreads'], c['occ'], c['I'], c['k_vram'])
            jobs = []
            with ThreadPoolExecutor(max_workers=len(TYPES)) as ex:
                for t in TYPES:
                    p = '%s/%s_%s.o' % (OBJ, t, tag)
                    jobs.append((t, p, ex.submit(compile_tu, t, p)))
                for t, p, f in jobs:
                    rc, err = f.result()
                    if rc != 0:
                        print('COMPILE FAIL %s %s: %s' % (tag, t, err.splitlines()[-1] if err else ''))
                        out.append(dict(cell=tag, typ=t, J=-1, fb=-1, status='compile-fail'))
                        continue
                    for key, r in audit_obj(p).items():
                        typ, J, fb, prec, fus = key
                        if fus != 0:
                            continue
                        r['cell'] = tag
                        r['nthreads'] = c['nthreads']
                        r['occ'] = c['occ']
                        r['I'] = c['I']
                        r['k_vram'] = c['k_vram']
                        r['lds'] = lds_bytes(typ, c['I'], J, c['nthreads'])
                        r['status'] = 'ok'
                        out.append(r)
            print('[%2d/%2d] %-16s rows=%d' % (ci + 1, len(CELLS), tag, sum(1 for r in out if r['cell'] == tag)), flush=True)
    finally:
        open(CFG, 'w').write(orig)
    keys = ['cell', 'typ', 'J', 'fb', 'nthreads', 'occ', 'I', 'k_vram', 'prec',
            'vgpr', 'vspill', 'sgpr', 'sspill', 'lds', 'status']
    with open(CSV, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in out:
            f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    print('wrote', CSV, len(out), 'rows')

if __name__ == '__main__':
    main()
