"""Test full-file row addressing, two-layer staging and changing graph replays."""
import json,os,struct,tempfile,types
from pathlib import Path
import torch
from vllm.models.deepseek_v4_1.common.engram import DiskEngramTable,ParallelEngramEmbedding,EngramDiskStager

def write_table(root,layer,w,s):
    names=[f'layers.{layer}.engram.embed.weight',f'layers.{layer}.engram.embed.scale']
    blobs=[w.view(torch.uint8).numpy().tobytes(),s.numpy().tobytes()]
    header={}; offset=0
    for name,tensor,data,dtype in zip(names,[w,s],blobs,['F8_E4M3','U8']):
        header[name]=dict(dtype=dtype,shape=list(tensor.shape),data_offsets=[offset,offset+len(data)])
        offset+=len(data)
    h=json.dumps(header).encode();h+=b' '*((-len(h))%8)
    file=f'layer-{layer}.safetensors'
    (root/file).write_bytes(struct.pack('<Q',len(h))+h+b''.join(blobs))
    return {name:file for name in names}

torch.manual_seed(31871);torch.set_num_threads(2)
cases=0
with tempfile.TemporaryDirectory(prefix='engram-graphs-') as tmp:
    root=Path(tmp);sizes=tuple(17+2*i for i in range(23));starts=[sum(sizes[:i]) for i in range(23)]
    rows=sum(sizes);dim=256;index={};tables={};references={}
    for layer in (1,14):
        w=(torch.rand(rows,dim,device='cpu')*4-2).to(torch.float8_e4m3fn)
        s=torch.randint(124,130,(rows,8),dtype=torch.uint8,device='cpu')
        index.update(write_table(root,layer,w,s))
        references[layer]=(w.float().reshape(rows,8,32)*torch.exp2(s.float()-127)[:,:,None]).reshape(rows,dim).bfloat16()
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
    for layer in (1,14):tables[layer]=DiskEngramTable(str(root),layer,dim,32)
    for rank in range(4):
        first=rank*6;last=min(first+6,23);engrams=[]
        for li,layer in enumerate((1,14)):
            emb=types.SimpleNamespace(part_n_hash_cols=6,head_start=first,n_hash_cols=23,
                  vocab_start_idx=sum(sizes[:first]),vocab_end_idx=sum(sizes[:first+6]),disk=tables[layer],dim=dim)
            emb.disk_rel_owned=types.MethodType(ParallelEngramEmbedding.disk_rel_owned,emb)
            engrams.append(types.SimpleNamespace(layer_hash_index=li,embed_tokens=emb,
                          staged_rows=torch.zeros(8,6,dim,device='cuda',dtype=torch.bfloat16)))
        class HashFixture:
            multipliers=torch.empty(2,1,device='cpu')
            def ensure_cache(self):return True
            def __call__(self,ids,*args):
                starts_gpu=torch.tensor(starts,device=ids.device,dtype=torch.int32)
                sizes_gpu=torch.tensor(sizes,device=ids.device,dtype=torch.int32)
                return torch.stack([starts_gpu+(ids[:,None]+li)%sizes_gpu for li in range(2)],dim=1)
        stager=EngramDiskStager(HashFixture(),engrams)
        pos=torch.arange(4,device='cuda',dtype=torch.int64)
        qsl=torch.tensor([0,4],device='cuda',dtype=torch.int32)
        lookback=torch.full((1,3),-1,device='cuda',dtype=torch.int32)
        graph=torch.cuda.CUDAGraph()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):result=engrams[0].staged_rows[:4]+engrams[1].staged_rows[:4]
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):result=engrams[0].staged_rows[:4]+engrams[1].staged_rows[:4]
        prior=None
        for shift in (0,7,1,15):
            ids=torch.tensor([shift,shift,shift+3,shift+9],device='cuda',dtype=torch.int32)
            expected_sum=torch.zeros(4,6,dim,device='cpu',dtype=torch.bfloat16)
            with torch.device('cuda'):
                assert stager.stage(ids,pos,qsl,lookback,4)==4
            for li,layer in enumerate((1,14)):
                expected=torch.zeros(4,6,dim,device='cpu',dtype=torch.bfloat16)
                cpu_ids=torch.tensor([[starts[h]+(i+li)%sizes[h] for h in range(23)]
                                     for i in [shift,shift,shift+3,shift+9]],device='cpu')
                expected[:,:last-first]=references[layer][cpu_ids[:,first:last]]
                assert torch.equal(engrams[li].staged_rows[:4].cpu(),expected),(rank,layer,shift,'stager')
                for device in ('cpu','cuda'):
                    with torch.device('cuda'):
                        actual=torch.empty(4,6,dim,device=device,dtype=torch.bfloat16)
                        ParallelEngramEmbedding._disk_lookup(engrams[li].embed_tokens,cpu_ids.to(device),actual)
                    assert torch.equal(actual.cpu(),expected),(rank,layer,shift,device)
                    cases+=1
                expected_sum+=expected
            graph.replay();torch.cuda.synchronize()
            assert torch.equal(result.cpu(),expected_sum),(rank,shift,'replay')
            if prior is not None:assert not torch.equal(prior,result.cpu()),'stale graph rows'
            prior=result.cpu().clone();cases+=1
    for table in tables.values():os.close(table.w_fd);os.close(table.s_fd)
print(json.dumps(dict(passed=True,cases=cases,full_global_rows=True,padded_head=True,
                     changing_replays=True,two_layers=True,cpu_alloc_under_cuda_default=True)),flush=True)
