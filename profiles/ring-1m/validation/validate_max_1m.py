"""Validate max reasoning, real tool round trips, six concurrent streams, and long retrieval."""
from pathlib import Path
import json,time,urllib.request,uuid,threading,concurrent.futures,re,os,base64
ROOT=Path(__file__).resolve().parent;LOG=Path(os.environ.get('DSV41_RESULTS_DIR', str(ROOT/'local-results')));LOG.mkdir(parents=True,exist_ok=True);MODEL='deepseek-v4.1-flash'
BASE=os.environ.get('DSV41_BASE_URL','http://127.0.0.1:8041').rstrip('/');KW={'thinking':True,'reasoning_effort':'max','drop_thinking':False}
def save(name,data):
    p=LOG/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));os.replace(t,p)
def state(stage,**kw):
    d={'stage':stage,'unix':time.time(),**kw};save('max-validation-state.json',d);print(json.dumps(d),flush=True)
def req(path,payload=None,timeout=3600):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    q=urllib.request.Request(BASE+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    return opener.open(q,timeout=timeout)
def met():
    with req('/metrics',timeout=10) as f:t=f.read().decode()
    out={}
    for l in t.splitlines():
        if l.startswith('vllm:'):
            k=l.split()[0].split('{')[0]
            if k.endswith(('_sum','_count','_total')) or k in ['vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc']:
                out[k]=out.get(k,0)+float(l.split()[-1])
    return out
def stream(name,messages,output=262144,extra=None,explicit=True):
    payload={'model':MODEL,'messages':messages,'max_tokens':output,'temperature':1.0,'top_p':0.95,
             'stream':True,'stream_options':{'include_usage':True},'cache_salt':str(uuid.uuid4())}
    if explicit:payload['chat_template_kwargs']=KW.copy()
    if extra:payload.update(extra)
    t0=time.monotonic();first=None;first_answer=None;usage=None;finish=None;parts=[];thoughts=[];tool_calls={}
    with req('/v1/chat/completions',payload) as f:
        for line in f:
            if not line.startswith(b'data: '):continue
            b=line[6:].strip()
            if b==b'[DONE]':break
            item=json.loads(b)
            if item.get('error'):raise RuntimeError(item['error'])
            if item.get('usage'):usage=item['usage']
            for ch in item.get('choices',[]):
                d=ch.get('delta',{});content=d.get('content') or '';thought=d.get('reasoning') or d.get('reasoning_content') or ''
                if content or thought or d.get('tool_calls'):
                    now=time.monotonic()
                    if first is None:first=now
                    if content:
                        if first_answer is None:first_answer=now
                        parts.append(content)
                    if thought:thoughts.append(thought)
                    for tc in d.get('tool_calls',[]):
                        x=tool_calls.setdefault(tc['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                        if tc.get('id'):x['id']=tc['id']
                        for k in ['name','arguments']:
                            if tc.get('function',{}).get(k):x['function'][k]+=tc['function'][k]
                if ch.get('finish_reason'):finish=ch['finish_reason']
    elapsed=time.monotonic()-t0
    assert usage is not None
    return {'name':name,'usage':usage,'finish_reason':finish,'response':''.join(parts),'reasoning':''.join(thoughts),
            'tool_calls':list(tool_calls.values()),'elapsed_seconds':elapsed,'ttft_seconds':None if first is None else first-t0,
            'first_answer_seconds':None if first_answer is None else first_answer-t0,'max_output_tokens':output,
            'thinking':True,'reasoning_effort':'max','temperature':1.0,'top_p':0.95,
            'client_decode_tps':None if first is None else (usage['completion_tokens']-1)/(elapsed-(first-t0)),
            'explicit_thinking_flags':explicit}
def monitor(stop,records):
    while not stop.is_set():
        try:records.append({'unix':time.time(),**met()})
        except Exception as e:records.append({'error':str(e)})
        stop.wait(1)
def run():
    state('waiting_for_service')
    with req('/v1/models') as f:models=json.load(f)
    assert models['data'][0]['max_model_len']==1048576
    state('default_max_and_vision')
    image=base64.b64encode((ROOT/'vision-probe.png').read_bytes()).decode()
    c=stream('default_max_vision',[{'role':'user','content':[{'type':'text','text':'用中文简洁回答：左边是什么颜色的什么形状，右边是什么颜色的什么形状？'},
           {'type':'image_url','image_url':{'url':'data:image/png;base64,'+image}}]}],explicit=False)
    assert c['finish_reason']=='stop' and c['reasoning'] and all(s in c['response'] for s in ['红','圆','蓝','方']),c
    save('max-vision.json',{'passed':True,'case':c})
    state('tool_round_trips')
    tools=[{'type':'function','function':{'name':'lookup_test_record','description':'Read a synthetic inventory record by SKU.','parameters':{'type':'object','properties':{'sku':{'type':'string'}},'required':['sku'],'additionalProperties':False}}},
           {'type':'function','function':{'name':'multiply','description':'Multiply two integers exactly.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}]
    msgs=[{'role':'system','content':'You are validating read-only tools on synthetic inventory. Use actual tool responses, preserve exact integer arithmetic, and answer concisely.'},
          {'role':'user','content':'查询 SKU A17 的库存和单价，再调用 multiply 算全部库存的总价（单位分）。最后用一句中文告诉我库存数、单价、总价和记录版本。'}]
    history=[];seen=[]
    for turn in range(5):
        c=stream('tool_turn_'+str(turn),msgs,extra={'tools':tools,'tool_choice':'auto'})
        history.append(c);save('max-tools.json',{'passed':False,'turns':history,'seen':seen})
        assert c['finish_reason'] in ['stop','tool_calls'],c['finish_reason']
        msg={'role':'assistant','content':c['response'] or None,'reasoning':c['reasoning']}
        if c['tool_calls']:msg['tool_calls']=c['tool_calls']
        msgs.append(msg)
        if not c['tool_calls']:
            assert set(seen)=={'lookup_test_record','multiply'} and '6750' in c['response'].replace(',','') and 'VX-207' in c['response'],c['response']
            assert any(x['reasoning'] for x in history)
            save('max-tools.json',{'passed':True,'turns':history,'seen':seen,'messages':msgs});break
        for tc in c['tool_calls']:
            name=tc['function']['name'];args=json.loads(tc['function']['arguments']);seen.append(name)
            if name=='lookup_test_record':
                assert args=={'sku':'A17'},args
                val={'sku':'A17','stock':18,'unit_price_cents':375,'version':'VX-207'}
            elif name=='multiply':
                assert sorted(args.values())==[18,375],args
                val={'result':args['a']*args['b']}
            else:raise AssertionError(name)
            msgs.append({'role':'tool','tool_call_id':tc['id'],'content':json.dumps(val)})
    else:raise AssertionError('tool loop did not finish')
    state('six_concurrent_streams')
    prompts=[
      'Write a Python merge_intervals(intervals) function that merges overlapping intervals. Include a short docstring, three examples, and a brief explanation. Do not use tools.',
      'Write an iterative Python binary_search(numbers, target) function. Return the index or -1. Include three examples and explain empty-list handling. Do not use tools.',
      'Write a Python count_words(text) function that lowercases and counts alphabetic words. Use the standard library, include three examples and a brief explanation. Do not use tools.',
      'Write a Python running_totals(values) generator. Include three examples covering empty input and negative values. Briefly explain its memory use. Do not use tools.',
      'Write a Python flatten_once(rows) function that flattens one level of nested lists. Include three examples and explain the behavior on empty rows. Do not use tools.',
      'Write a Python is_palindrome(text) function ignoring case and nonalphanumeric characters. Include three examples and explain the two-pointer approach. Do not use tools.'
    ]
    records=[];stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,records),daemon=True);thread.start()
    before=met();started=time.monotonic();barrier=threading.Barrier(6)
    def one(i):
        barrier.wait();return stream('c6_code_'+str(i),[{'role':'user','content':prompts[i]}])
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:cases=list(ex.map(one,range(6)))
    finally:stop.set();thread.join(15)
    elapsed=time.monotonic()-started;after=met()
    passed=all(c['finish_reason']=='stop' and c['response'] and c['reasoning'] for c in cases)
    peak=max(x.get('vllm:num_requests_running',0) for x in records)
    result={'passed':passed and peak==6,'cases':cases,'elapsed_seconds':elapsed,
            'aggregate_output_tps':sum(c['usage']['completion_tokens'] for c in cases)/elapsed,
            'peak_running_requests':peak,'metrics_samples':records,'metrics_delta':{k:after.get(k,0)-v for k,v in before.items()},
            'scope':'Six independent short coding requests, max reasoning, temperature 1/top_p .95. Not six full 1M contexts.'}
    save('max-concurrency6.json',result);assert result['passed']
    state('short_validation_complete',aggregate_tps=result['aggregate_output_tps'],peak_running=peak)
if __name__ == "__main__":
    try:run()
    except Exception as e:state('failed',error=str(e));raise
