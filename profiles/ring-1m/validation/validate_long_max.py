from validate_max_1m import req,stream,met,save,LOG
import hashlib,json,time,threading,os,traceback
def state(stage,**kw):
    d={'stage':stage,'unix':time.time(),**kw};save('long-validation-state.json',d);print(json.dumps(d),flush=True)
def token_count(messages):
    with req('/tokenize',{'model':'deepseek-v4.1-flash','messages':messages,'add_generation_prompt':True,'chat_template_kwargs':{'thinking':True,'reasoning_effort':'max'}}) as f:
        d=json.load(f)
    return d['count']
def make_prompt(n,seed):
    needles={}
    rows=[]
    positions={max(1,int(n*f)):k for f,k in [(.12,'aurora'),(.51,'harbor'),(.89,'cedar')]}
    for i in range(n):
        key=hashlib.sha256((seed+':'+str(i)).encode()).hexdigest()[:20]
        rows.append('record {:06d} | code {} | units {} | zone {}\n'.format(i,key,17+i%983,i%37))
        if i in positions:
            name=positions[i];value='proof-'+hashlib.sha256((seed+name).encode()).hexdigest()[:16]
            rows.append('VERIFIED_LOOKUP '+name+' = '+value+'\n');needles[name]=value
    text='Read this synthetic log and locate the three VERIFIED_LOOKUP entries. Ordinary records are distractors.\n'+''.join(rows)+'\nReturn only one JSON object with the exact VERIFIED_LOOKUP values for aurora, harbor, and cedar. Do not infer values; copy the three entries from the log.'
    return [{'role':'user','content':text}],needles
def run():
    state('waiting_for_short_validation')
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        p=LOG/'max-validation-state.json'
        if p.exists():
            s=json.loads(p.read_text())['stage']
            if s=='failed':raise RuntimeError('Short validation failed; long requests were not sent')
            if s=='short_validation_complete':break
        time.sleep(5)
    else:raise TimeoutError('Short validation did not complete')
    results=[]
    for target,output in [(65536,262144),(262144,262144),(784384,262144),(982528,65536)]:
        state('preparing_prompt',target_prompt_tokens=target,max_output_tokens=output)
        seed='ring-max-'+str(target);n=max(100,target//30)
        for _ in range(6):
            messages,expected=make_prompt(n,seed);count=token_count(messages)
            if target-512<=count<=target:break
            n=max(100,int(n*(target-256)/count))
        assert target-2048<=count<=target,(target,count,n)
        assert count+output<=1048576
        state('prefilling',target_prompt_tokens=target,actual_prompt_tokens=count,max_output_tokens=output,rows=n)
        before=met();t0=time.monotonic()
        case=stream('long_retrieval_'+str(target),messages,output=output)
        after=met();delta={k:after.get(k,0)-v for k,v in before.items()}
        text=case['response'].strip()
        if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
        try:answer=json.loads(text)
        except Exception:answer=None
        case['expected']=expected;case['parsed_answer']=answer;case['metrics_delta']=delta
        case['prompt_tokens_verified_by_tokenizer']=count
        case['passed']=case['finish_reason']=='stop' and answer==expected and bool(case['reasoning'])
        pf=delta.get('vllm:request_prefill_time_seconds_sum',0)
        if pf>0:case['prefill_seconds']=pf;case['prefill_tokens_per_second']=case['usage']['prompt_tokens']/pf
        results.append(case);save('long-validation-results.json',{'passed':all(c['passed'] for c in results),'cases':results})
        assert case['passed'],{'target':target,'response':case['response'],'expected':expected,'finish':case['finish_reason']}
        state('case_passed',prompt_tokens=case['usage']['prompt_tokens'],elapsed_seconds=time.monotonic()-t0)
    state('long_validation_complete',passed=True)
if __name__=='__main__':
    try:run()
    except Exception as e:state('failed',error=repr(e),traceback=traceback.format_exc());raise
