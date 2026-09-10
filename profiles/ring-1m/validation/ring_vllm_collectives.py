"""Exercise the same PyNccl path used by vLLM tensor parallelism."""
import datetime,json,os,time
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

rank=int(os.environ['RANK']);world=4
torch.cuda.set_device(0)
dist.init_process_group('gloo',init_method='env://',rank=rank,world_size=world,
                        timeout=datetime.timedelta(seconds=90))
comm=PyNcclCommunicator(dist.group.WORLD,device=torch.device('cuda:0'))
assert not comm.disabled
print(json.dumps({'rank':rank,'nccl_runtime_version':comm.nccl.ncclGetVersion()}),flush=True)
for count in [1024,262144,4194304]:
 x=torch.full((count,),rank+1.,device='cuda',dtype=torch.float32)
 y=comm.all_reduce(x);torch.cuda.synchronize();assert bool((y==10).all())
 gathered=torch.empty((world*count,),device='cuda',dtype=x.dtype)
 comm.all_gather(gathered,x);torch.cuda.synchronize()
 for i in range(world):assert bool((gathered[i*count:(i+1)*count]==i+1).all())
 source=torch.full((world*count,),rank+1.,device='cuda',dtype=x.dtype)
 scattered=torch.empty_like(x);comm.reduce_scatter(scattered,source)
 torch.cuda.synchronize();assert bool((scattered==10).all())
 broadcast=torch.full_like(x,73 if rank==0 else 0)
 comm.broadcast(broadcast,src=0);torch.cuda.synchronize();assert bool((broadcast==73).all())
 del y,gathered,source,scattered,broadcast
 out=torch.empty_like(x)
 for _ in range(3):comm.all_reduce(x,out)
 torch.cuda.synchronize();t=time.perf_counter()
 for _ in range(10):comm.all_reduce(x,out)
 torch.cuda.synchronize()
 print(json.dumps({'rank':rank,'input_bytes':count*4,'pynccl_correct':True,
                   'all_reduce_ms':(time.perf_counter()-t)*100}),flush=True)
dist.barrier();print(json.dumps({'rank':rank,'vllm_collectives_passed':True}),flush=True)
dist.destroy_process_group()
