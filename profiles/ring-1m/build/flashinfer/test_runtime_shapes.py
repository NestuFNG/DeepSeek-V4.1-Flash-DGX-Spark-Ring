"""Verify public attention API for this deployment against gathered FP32 attention."""
import json, sys, time
import torch
sys.path.insert(0, '/opt/fi-src/tests/attention')
from test_sparse_mla_sm120 import quantize_kv_dsv4, dequantize_kv_dsv4
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config

torch.manual_seed(10911)
assert has_flashinfer_sparse_mla_sm120_config(16, 1152)
device='cuda'
workspace=torch.empty(128*1024*1024,dtype=torch.uint8,device=device)
results=[]
for tokens, capacity, extra_capacity in [(1,128,0),(16,1152,0),(65,1152,0),(256,1152,0),
                                        (1,128,512),(16,128,512),(65,128,512),(256,128,512)]:
    q=torch.randn(tokens,16,512,device=device,dtype=torch.bfloat16)*0.2
    sink=torch.randn(16,device=device,dtype=torch.float32)
    kw={}
    values=[]; masks=[]
    for seg, cap in enumerate([capacity]+([extra_capacity] if extra_capacity else [])):
        kv=torch.randn(32,64,1,512,device=device,dtype=torch.bfloat16)*0.2
        packed=quantize_kv_dsv4(kv)
        dequant=dequantize_kv_dsv4(packed).reshape(-1,512).float()
        # vLLM aligns physical blocks; rows within each packed block remain tight.
        padded=torch.empty_strided(packed.shape,(37888,584,584,1),device=device,dtype=torch.uint8)
        padded.copy_(packed)
        idx=torch.randint(0,2048,(tokens,cap),device=device,dtype=torch.int32)
        lens=torch.randint(1,cap+1,(tokens,),device=device,dtype=torch.int32)
        vals=dequant[idx.long()]
        mask=torch.arange(cap,device=device)[None,:] < lens[:,None]
        values.append(vals);masks.append(mask)
        if seg==0:kw.update(swa_kv_cache=padded,sparse_indices=idx,swa_topk_lens=lens)
        else:kw.update(compressed_kv_cache=padded,extra_sparse_indices=idx,extra_sparse_topk_lens=lens)
    vals=torch.cat(values,dim=1)
    mask=torch.cat(masks,dim=1)
    scores=torch.einsum('thd,tkd->thk',q.float(),vals)*(512**-0.5)
    scores.masked_fill_(~mask[:,None,:],float('-inf'))
    denom=torch.logaddexp(torch.logsumexp(scores,dim=-1),sink[None,:])
    reference=torch.einsum('thk,tkd->thd',torch.exp(scores-denom[:,:,None]),vals)
    out=torch.empty_like(q)
    start=time.monotonic()
    returned=trtllm_batch_decode_sparse_mla_dsv4(query=q,workspace_buffer=workspace,
        out=out,bmm1_scale=512**-0.5,sinks=sink,kv_layout='NHD',**kw)
    torch.cuda.synchronize()
    assert returned.data_ptr()==out.data_ptr()
    assert torch.isfinite(out).all()
    error=(out.float()-reference).abs().max().item()
    relative_l2=((out.float()-reference).norm()/reference.norm()).item()
    cosine=torch.nn.functional.cosine_similarity(out.float().flatten(),reference.flatten(),dim=0).item()
    # FlashInfer's own FP8 DSV4 test uses atol=rtol=0.05 against BF16/FP32.
    # Also constrain overall error so a permissive absolute tolerance cannot
    # hide a systematically wrong low-magnitude attention result.
    torch.testing.assert_close(out.float(),reference,atol=0.01,rtol=0.05)
    assert relative_l2 < 0.03 and cosine > 0.999
    row=dict(tokens=tokens,main_topk=capacity,extra_topk=extra_capacity,max_abs_error=error,
             relative_l2=relative_l2,cosine=cosine,
             elapsed_seconds=time.monotonic()-start)
    results.append(row);print(json.dumps(row),flush=True)
print(json.dumps({'passed':True,'cases':len(results),'results':results}),flush=True)
