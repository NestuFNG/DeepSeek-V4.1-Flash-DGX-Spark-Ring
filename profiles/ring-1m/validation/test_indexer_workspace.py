from pathlib import Path
import ast, json
p=(Path(__file__).resolve().parents[1]/'runtime/memory-patches/indexer.py').read_text()
# Exercise the real buffer/chunk functions without importing the GPU backend.
tree=ast.parse(p); names={'get_max_prefill_buffer_size','split_indexer_prefill_chunks'}
body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
env={};exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'indexer-isolated','exec'),env)
from types import SimpleNamespace as NS
class Scalar(int):
    def item(self):return int(self)
L=1048576;records=[]
for concurrent in (1,6,48):
 cfg=NS(model_config=NS(max_model_len=L,hf_config=NS(model_type='deepseek_v41')),scheduler_config=NS(max_num_seqs=concurrent))
 size=env['get_max_prefill_buffer_size'](cfg); assert size==L*min(40,concurrent)
 for ratio in (1,2):
  seq=[Scalar(L//ratio-i*64) for i in range(min(6,concurrent))];q=[Scalar(512+i*16) for i in range(len(seq))]
  chunks=env['split_indexer_prefill_chunks'](seq,q,size//ratio,512*1024*1024)
  coverage={i:[] for i in range(len(q))}
  for rs,qs in chunks:
   assert sum(seq[rs])<=size//ratio
   assert sum(seq[rs])*(qs.stop-qs.start)*4<=512*1024*1024
   off=0
   for i in range(rs.start,rs.stop):
    a=max(qs.start,off);b=min(qs.stop,off+q[i]);coverage[i].extend(range(a-off,b-off));off+=q[i]
  assert all(sorted(coverage[i])==list(range(q[i])) for i in coverage)
  records.append({'concurrency':concurrent,'ratio':ratio,'workspace_rows':size//ratio,'chunks':len(chunks)})
print(json.dumps({'passed':True,'cases':records},indent=2))
