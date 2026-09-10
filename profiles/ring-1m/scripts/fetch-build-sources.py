"""Fetch only the pinned FlashInfer build sources; verify every archive hash."""
from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request

root=Path(__file__).resolve().parents[1]/'build/flashinfer'
def fetch(row):
    path=root/(row['name']+'.tar.gz')
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']:
        return row['name']+' already verified'
    with urllib.request.urlopen(row['url'],timeout=180) as response:
        data=response.read()
    if hashlib.sha256(data).hexdigest()!=row['sha256']:
        raise ValueError('Archive SHA256 mismatch: '+row['name'])
    temp=path.with_suffix('.part');temp.write_bytes(data);temp.replace(path)
    return row['name']+' downloaded and verified'
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(fetch,json.loads((root/'sources.json').read_text())):
        print(result,flush=True)
