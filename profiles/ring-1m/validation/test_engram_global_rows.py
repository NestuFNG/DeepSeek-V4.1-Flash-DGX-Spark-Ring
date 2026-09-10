"""Exercise the actual patched disk reader and rank lookup on full synthetic files."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import struct
import tempfile
import types

import torch

source = Path('/patches/engram.py').read_text()
tree = ast.parse(source)
disk = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DiskEngramTable')
parallel = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ParallelEngramEmbedding')
lookup = next(n for n in parallel.body if isinstance(n, ast.FunctionDef) and n.name == '_disk_lookup')
ns = {'torch': torch, '_kai_os': os, '_kai_json': json, '_kai_struct': struct,
      '_KaiPool': ThreadPoolExecutor, 'logger': logging.getLogger('engram-test')}
exec(compile(ast.Module(body=[disk, lookup], type_ignores=[]), '<actual-patch>', 'exec'), ns)

def write_table(root, layer, values, scales):
    names = [f'layers.{layer}.engram.embed.weight', f'layers.{layer}.engram.embed.scale']
    payloads = [values.view(torch.uint8).numpy().tobytes(), scales.numpy().tobytes()]
    header = {}; off = 0
    for name, tensor, payload, dtype in zip(names, [values, scales], payloads, ['F8_E4M3', 'U8']):
        header[name] = {'dtype': dtype, 'shape': list(tensor.shape), 'data_offsets': [off, off+len(payload)]}
        off += len(payload)
    encoded = json.dumps(header).encode(); encoded += b' ' * ((-len(encoded)) % 8)
    file = f'layer-{layer}.safetensors'
    (root/file).write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(payloads))
    return {name: file for name in names}

torch.manual_seed(31871)
with tempfile.TemporaryDirectory(prefix='engram-rank-check-') as tmp:
    root = Path(tmp)
    sizes = tuple(17 + 2*i for i in range(23))  # Uneven heads, and one padded TP4 head.
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    rows = sum(sizes); dim = 256
    tables = {}; index = {}
    for layer in (1, 14):
        weights = (torch.rand(rows, dim, device='cpu')*4-2).to(torch.float8_e4m3fn)
        scales = torch.randint(124, 130, (rows, 8), dtype=torch.uint8, device='cpu')
        tables[layer] = (weights, scales)
        index.update(write_table(root, layer, weights, scales))
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
    ids = torch.tensor([[start + min(t, size-1) for start, size in zip(starts, sizes)]
                        for t in (0, 1, 5, 50)], dtype=torch.int64, device='cpu')
    count = 0
    for layer, (weights, scales) in tables.items():
        table = ns['DiskEngramTable'](str(root), layer, dim, 32)
        ref = (weights.float().reshape(rows,8,32) * torch.exp2(scales.float()-127)[:,:,None]).reshape(rows,dim).bfloat16()
        for rank in range(4):
            first = rank*6; last = min(first+6, len(sizes))
            obj = types.SimpleNamespace(part_n_hash_cols=6, head_start=first,
                    n_hash_cols=len(sizes), vocab_start_idx=sum(sizes[:first]),
                    vocab_end_idx=sum(sizes[:first+6]), disk=table, dim=dim)
            for target in ['cpu', 'cuda']:
                expected = torch.zeros(4,6,dim,device='cpu',dtype=torch.bfloat16)
                expected[:,:last-first] = ref[ids[:,first:last]]
                with torch.device(target):
                    output = torch.empty(4,6,dim,dtype=torch.bfloat16,device=target)
                    ns['_disk_lookup'](obj, ids.to(target), output)
                assert torch.equal(output.cpu(), expected), (layer,rank,target)
                count += 1
        table.pool.shutdown()
        os.close(table.w_fd); os.close(table.s_fd)
    print(json.dumps({'engram_global_row_test': 'passed', 'cases': count,
                      'layers': [1,14], 'ranks': 4, 'devices': ['cpu','cuda'],
                      'full_files': True, 'uneven_and_padded_heads': True}), flush=True)
