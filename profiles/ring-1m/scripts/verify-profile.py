"""Check the public profile's file hashes before mounting runtime source."""
import hashlib, json
from pathlib import Path
p=Path(__file__).resolve().parents[1]
manifest=json.loads((p/'config/public-file-hashes.json').read_text())
for name,expected in manifest.items():
    q=p/name
    if not q.is_file() or hashlib.sha256(q.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Profile file differs from manifest: '+name)
print('Public profile files verified:',len(manifest))
