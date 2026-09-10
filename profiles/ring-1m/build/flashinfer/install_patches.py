from pathlib import Path
import json, hashlib
p = Path('/opt/dsv41-upgrade/patches')
v = Path('/usr/local/lib/python3.12/dist-packages/vllm')
for row in json.loads((p/'manifest.json').read_text()):
    dest = v / row['destination']
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == row['original_sha256'], dest
    data = (p / row['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['patched_sha256'], dest
    compile(data, str(dest), 'exec')
    dest.write_bytes(data)
    print('Patched', row['destination'])
