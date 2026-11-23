from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import urllib.request,json,time,os,hashlib
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data'/'raw';D.mkdir(parents=True,exist_ok=True)
URL='https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/resolve/2742263/XJTU-SY_Bearing_Datasets.zip?download=true'
SIZE=5447931446;N=8;CHUNK=(SIZE+N-1)//N
dest=D/'XJTU-SY_Bearing_Datasets.zip';old=dest.with_suffix('.zip.part')
EXPECTED_SHA256='3cc815649a315ac7da202980c489f33db44ca2db0317bbe3bcb9dcf415375e10'
if dest.exists() and dest.stat().st_size==SIZE:
 with dest.open('rb') as existing_file:
  if hashlib.file_digest(existing_file,'sha256').hexdigest()==EXPECTED_SHA256:
   print('Verified archive already present');raise SystemExit(0)
 raise RuntimeError('Existing raw archive checksum mismatch; move it aside before downloading')
if old.exists() and not (D/'segment_00.part').exists() and old.stat().st_size<CHUNK:old.rename(D/'segment_00.part')
def download(i):
 start=i*CHUNK;end=min(SIZE,(i+1)*CHUNK)-1;p=D/f'segment_{i:02d}.part'
 for attempt in range(10):
  have=p.stat().st_size if p.exists() else 0
  if have==end-start+1:return i
  try:
   req=urllib.request.Request(URL,headers={'Range':f'bytes={start+have}-{end}','User-Agent':'NEU-Bearing-Research/1.0'})
   with urllib.request.urlopen(req,timeout=120) as r:
    assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start+have}-'),r.headers
    with p.open('ab') as f:
     while b:=r.read(2*1024*1024):f.write(b)
   if p.stat().st_size!=end-start+1:raise IOError('Partial segment')
   return i
  except Exception as e:
   print('Retry segment',i,attempt+1,type(e).__name__,str(e)[:100],flush=True)
   if attempt==9:raise
   time.sleep(3)
with ThreadPoolExecutor(max_workers=N) as pool:
 pending={pool.submit(download,i) for i in range(N)}
 while pending:
  done,pending=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
  for f in done:f.result()
  total=sum(p.stat().st_size for p in D.glob('segment_*.part'))
  (D/'download_status.json').write_text(json.dumps({'downloaded_bytes':total,'total_bytes':SIZE,'pending_segments':len(pending),'url':URL,'time':time.time()},indent=2))
  print(f'{total/1e9:.3f}/{SIZE/1e9:.3f} GB; {len(pending)} segments pending',flush=True)
h=hashlib.sha256()
with dest.with_suffix('.assembling').open('wb') as w:
 for i in range(N):
  with (D/f'segment_{i:02d}.part').open('rb') as r:
   while b:=r.read(8*1024*1024):w.write(b);h.update(b)
assert h.hexdigest()==EXPECTED_SHA256,'Downloaded archive checksum mismatch'
os.replace(dest.with_suffix('.assembling'),dest)
(D/'download_manifest.json').write_text(json.dumps({'url':URL,'original_publisher':'https://biaowang.tech/xjtu-sy-bearing-datasets/','bytes':SIZE,'sha256':h.hexdigest(),'source_role':'Public mirror pinned to repository revision 2742263; original publisher cited','download_complete':True},indent=2))
print('DOWNLOAD COMPLETE',h.hexdigest(),flush=True)
