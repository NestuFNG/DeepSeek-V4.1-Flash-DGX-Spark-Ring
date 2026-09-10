"""Separate complete-batch throughput from measured full-concurrency intervals."""
import json, sys
from pathlib import Path

def analyze(c):
    records=[x for x in c['metrics_samples'] if 'unix' in x and 'vllm:num_requests_running' in x]
    durations={}
    for a,b in zip(records,records[1:]):
        n=int(a['vllm:num_requests_running']);durations[n]=durations.get(n,0)+b['unix']-a['unix']
    full=[x for x in records if x['vllm:num_requests_running']==6]
    # This batch drains monotonically after all requests have begun. Do not apply
    # first-to-last logic to a replenished queue with gaps in full concurrency.
    window=None
    if len(full)>=2:
        a,b=full[0],full[-1]
        between=[x for x in records if a['unix']<=x['unix']<=b['unix']]
        if all(x['vllm:num_requests_running']==6 for x in between):
            dt=b['unix']-a['unix'];tokens=b['vllm:generation_tokens_total']-a['vllm:generation_tokens_total']
            drafts=b['vllm:spec_decode_num_drafts_total']-a['vllm:spec_decode_num_drafts_total']
            accepted=b['vllm:spec_decode_num_accepted_tokens_total']-a['vllm:spec_decode_num_accepted_tokens_total']
            window={'seconds':dt,'tokens':tokens,'output_tps':tokens/dt,'accepted_plus_bonus_per_request_step':1+accepted/drafts if drafts else None,
                    'approx_batch_step_ms':dt/(drafts/6)*1000 if drafts else None}
    delta=c['metrics_delta'];drafts=delta.get('vllm:spec_decode_num_drafts_total',0)
    output=sum(x['usage']['completion_tokens'] for x in c['cases'])
    reasoning=sum(x['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for x in c['cases'])
    return {'batch_seconds':c['elapsed_seconds'],'batch_output_tokens':output,'reasoning_tokens':reasoning,'reasoning_share':reasoning/output,
            'whole_batch_output_tps':output/c['elapsed_seconds'],'approx_seconds_by_active_count':durations,
            'continuous_six_active_window':window,'accepted_plus_bonus_per_request_step':1+delta.get('vllm:spec_decode_num_accepted_tokens_total',0)/drafts if drafts else None,
            'preemptions':delta.get('vllm:num_preemptions_total'),
            'measurement_note':'1-second server counter samples. Full-window throughput includes thinking tokens. Step-time estimate is not a GPU profiler measurement.'}

if __name__=='__main__':
    print(json.dumps(analyze(json.loads(Path(sys.argv[1]).read_text())),indent=2))
