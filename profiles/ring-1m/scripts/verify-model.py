"""Verify checkpoint shards once; check unchanged file fingerprints at launch."""
from pathlib import Path
import argparse, hashlib, json, os, time

p=argparse.ArgumentParser()
p.add_argument('model_dir',type=Path)
p.add_argument('--write-stamp',type=Path)
p.add_argument('--check-stamp',type=Path)
args=p.parse_args()
if bool(args.write_stamp)==bool(args.check_stamp):p.error('Select exactly one stamp mode')
manifest=json.loads((Path(__file__).resolve().parents[1]/'config/checkpoint-shards.json').read_text())
root=args.model_dir.resolve()
stamp=json.loads(args.check_stamp.read_text()) if args.check_stamp else None
if stamp and stamp.get('model_dir')!=str(root):raise SystemExit('Verification stamp belongs to another path')
rows={}
for row in manifest['files']:
    q=root/row['file']
    if q.is_symlink() or not q.is_file():raise SystemExit('Expected a regular shard: '+row['file'])
    s=q.stat();fingerprint=[s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino]
    if s.st_size!=row['size']:raise SystemExit('Wrong shard length: '+row['file'])
    if stamp:
        previous=stamp['files'][row['file']]
        if previous['fingerprint']!=fingerprint or previous['sha256']!=row['sha256']:
            raise SystemExit('Shard changed; repeat full SHA256 verification: '+row['file'])
    else:
        h=hashlib.sha256()
        with q.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=row['sha256']:raise SystemExit('SHA256 mismatch: '+row['file'])
        print('Verified',row['file'],flush=True)
    rows[row['file']]={'fingerprint':fingerprint,'sha256':row['sha256']}
for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
    if not (root/name).is_file():raise SystemExit('Missing checkpoint metadata: '+name)
if args.write_stamp:
    args.write_stamp.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.write_stamp.with_suffix('.tmp')
    tmp.write_text(json.dumps({'model_dir':str(root),'revision':manifest['revision'],'unix':time.time(),'files':rows},indent=2)+'\n')
    os.replace(tmp,args.write_stamp)
print('All 48 shard checks passed. Metadata presence checked; metadata is not covered by the shard hashes.')
