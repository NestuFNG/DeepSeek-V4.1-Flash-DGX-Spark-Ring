"""Recompute the published B12 snapshot; no network or inference required."""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root / 'long-agent-8x500k-b12.json').read_text())
samples = json.loads((root / 'long-agent-8x500k-b12-samples.json').read_text())
for name, phase in data['phases'].items():
    output = sum(r['completion_tokens'] for r in phase['per_request'])
    assert output == phase['completion_tokens']
    whole = output / phase['elapsed_seconds']
    assert math.isclose(whole, phase['batch_output_tps'], rel_tol=1e-10)
    item = samples[name]
    rows = [dict(zip(item['columns'], values)) for values in item['rows']]
    seconds = tokens = 0
    for a, b in zip(rows, rows[1:]):
        # Reject prefill, tool-turn transitions, changing request identities,
        # and any interval where a request has not produced its first token.
        if (a['running'] == b['running'] == 8
                and len(a['active_request_ids']) == 8
                and a['active_request_ids'] == b['active_request_ids']
                and a['active_request_ids'] == a['generating_request_ids']
                and b['active_request_ids'] == b['generating_request_ids']):
            seconds += b['elapsed_seconds'] - a['elapsed_seconds']
            tokens += b['generation_tokens'] - a['generation_tokens']
    window = phase['sampled_all_eight_decoding']
    assert math.isclose(seconds, window['seconds'], abs_tol=1e-6)
    assert tokens == window['tokens']
    print(json.dumps({
        'phase': name, 'passed': phase['passed'],
        'whole_batch_output_tps': whole,
        'all_eight_decode_seconds': seconds,
        'all_eight_decode_tokens': tokens,
        'all_eight_decode_tps': tokens / seconds if seconds else None,
        'completed_requests': sum(r['finish_reason'] != 'length'
                                  for r in phase['per_request']),
        'requests': len(phase['per_request']),
    }))
