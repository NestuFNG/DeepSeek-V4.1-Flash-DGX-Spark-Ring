"""Use the upstream coding prompt, but preserve ON/max and temperature 1.

This is not a reproduction of the upstream OFF/temperature-0 benchmark.
Run after long validation on an otherwise idle instance.
"""
from validate_max_1m import stream, met, monitor, save
import concurrent.futures, threading, time

PROMPT='Write a Python function merge_intervals(intervals) that merges overlapping intervals and returns them sorted. Include a one-line docstring and two example calls.'
def run():
    results=[]
    for n in (1,6):
        samples=[];stop=threading.Event();t=threading.Thread(target=monitor,args=(stop,samples),daemon=True);t.start()
        before=met();barrier=threading.Barrier(n);start=time.monotonic()
        def one(i):
            barrier.wait()
            return stream('upstream_prompt_max_c%d_s%d'%(n,i),[{'role':'user','content':PROMPT}])
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:cases=list(pool.map(one,range(n)))
        finally:stop.set();t.join(15)
        elapsed=time.monotonic()-start;after=met()
        row={'concurrency':n,'prompt':PROMPT,'cases':cases,'elapsed_seconds':elapsed,
             'aggregate_output_tps':sum(c['usage']['completion_tokens'] for c in cases)/elapsed,
             'metrics_samples':samples,'metrics_delta':{k:after.get(k,0)-v for k,v in before.items()},
             'passed':all(c['finish_reason']=='stop' and c['reasoning'] and c['response'] for c in cases)}
        results.append(row);save('public-prompt-max.json',{'cases':results,'scope':'Same coding prompt as upstream; ON/max and temperature 1, not equivalent to upstream OFF benchmark.'})
        assert row['passed']
        print({'concurrency':n,'aggregate_tps':row['aggregate_output_tps'],'completed':True},flush=True)
if __name__=='__main__':run()
