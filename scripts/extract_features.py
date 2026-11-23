from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import zipfile,re,json,hashlib,io,os,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'data'/'raw';DATA=ROOT/'data';DATA.mkdir(parents=True,exist_ok=True)
ARCHIVE=RAW/'XJTU-SY_Bearing_Datasets.zip';CACHE=ROOT/'data'/'feature_cache';CACHE.mkdir(exist_ok=True)
FS=25600.;N=32768
def descriptors(x,prefix,rpm):
 x=x.astype(np.float64);mean=x.mean();c=x-mean;std=c.std();rms=np.sqrt(np.mean(x*x));a=np.abs(x);peak=a.max();ma=a.mean();eps=1e-12
 c2=c*c
 f={prefix+'mean':mean,prefix+'std':std,prefix+'rms':rms,prefix+'absmean':ma,prefix+'peak':peak,prefix+'ptp':np.ptp(x),prefix+'kurtosis':np.mean(c2*c2)/(std**4+eps),prefix+'skewness':np.mean(c2*c)/(std**3+eps),prefix+'crest':peak/(rms+eps),prefix+'shape':rms/(ma+eps),prefix+'impulse':peak/(ma+eps)}
 window=np.hanning(len(x));spec=np.abs(np.fft.rfft(c*window))**2;freq=np.fft.rfftfreq(len(x),1/FS);total=spec.sum()+eps;prob=spec/total
 f[prefix+'spectral_entropy']=-np.sum(prob*np.log(prob+eps))/np.log(len(prob));f[prefix+'spectral_centroid']=np.sum(freq*spec)/total;f[prefix+'dominant_order']=freq[np.argmax(spec)]/(rpm/60)
 for low,high in [(0,1000),(1000,5000),(5000,12800)]:f[prefix+f'power_{low}_{high}']=spec[(freq>=low)&(freq<high)].sum()/total
 # Geometry-independent shaft-order bands avoid assuming unpublished dimensions.
 for order in [1,2,3,4]:
  fr=rpm/60;f[prefix+f'order{order}_power']=spec[(freq>=fr*(order-.2))&(freq<=fr*(order+.2))].sum()/total
 analytic=np.fft.fft(c);mask=np.zeros(len(c));mask[0]=1;mask[len(c)//2]=1;mask[1:len(c)//2]=2
 env=np.abs(np.fft.ifft(analytic*mask));ec=env-env.mean();es=ec.std()
 ec2=ec*ec
 f[prefix+'envelope_rms']=np.sqrt(np.mean(env*env));f[prefix+'envelope_kurtosis']=np.mean(ec2*ec2)/(es**4+eps)
 return f
def extract_bearing(bearing,names,last_total=None,shard=''):
 path=CACHE/(bearing+shard+'.jsonl');done={}
 if path.exists():
  for line in path.read_text().splitlines():
   try:r=json.loads(line);done[r['cycle']]=r
   except json.JSONDecodeError:pass
  # Rewrite valid records so a killed append never poisons the next append.
  path.write_text(''.join(json.dumps(done[k])+'\n' for k in sorted(done)))
 c=int(re.search(r'Bearing(\d)_',bearing).group(1));run=int(bearing.rsplit('_',1)[1]);rpm,load={1:(2100,12),2:(2250,11),3:(2400,10)}[c]
 last=last_total if last_total is not None else max(int(Path(n).stem) for n in names)
 with zipfile.ZipFile(ARCHIVE) as z,path.open('a',encoding='utf-8') as f:
  for name in sorted(names,key=lambda n:int(Path(n).stem)):
   cycle=int(Path(name).stem)
   if cycle in done:continue
   raw=z.read(name);x=pd.read_csv(io.BytesIO(raw),dtype=np.float64,float_precision='round_trip').to_numpy()
   if x.shape!=(N,2) or not np.isfinite(x).all():raise ValueError(f'Bad raw snapshot: {name}: {x.shape}')
   r={'bearing_id':bearing,'condition':c,'run':run,'cycle':cycle,'elapsed_minutes':cycle-1,'rpm':rpm,'load_kN':load,'rul_minutes':last-cycle,'observed_lifetime_minutes':last-1,'source_path':name,'raw_sha256':hashlib.sha256(raw).hexdigest()}
   r.update(descriptors(x[:,0],'h_',rpm));r.update(descriptors(x[:,1],'v_',rpm))
   r['channel_rms_ratio']=r['h_rms']/(r['v_rms']+1e-12)
   f.write(json.dumps(r)+'\n');f.flush()
   if cycle%100==0:print(bearing,cycle,'/',last,flush=True)
 print('EXTRACTED',bearing,last,flush=True)
 return path
def main():
 with zipfile.ZipFile(ARCHIVE) as z:
  groups={}
  for name in z.namelist():
   m=re.search(r'/(Bearing\d_\d)/(\d+)\.csv$',name)
   if m:groups.setdefault(m.group(1),[]).append(name)
 assert len(groups)==15
 known={}
 for cache in CACHE.glob('*.jsonl'):
  for line in cache.read_text().splitlines():
   try:
    row=json.loads(line);known.setdefault(row['bearing_id'],set()).add(row['cycle'])
   except json.JSONDecodeError:pass
 tasks=[]
 for b,names in sorted(groups.items()):
  last=max(int(Path(n).stem) for n in names)
  pending=sorted([n for n in names if int(Path(n).stem) not in known.get(b,set())],key=lambda n:int(Path(n).stem))
  for k in range(0,len(pending),250):tasks.append((b,pending[k:k+250],last,'_chunk'+str(k//250)))
 with ProcessPoolExecutor(max_workers=6) as pool:
  for f in as_completed([pool.submit(extract_bearing,*task) for task in tasks]):f.result()
 records=[]
 for p in sorted(CACHE.glob('*.jsonl')):records.extend(json.loads(l) for l in p.read_text().splitlines())
 import csv
 temp=DATA/'XJTU_SY_features.csv.tmp';target=DATA/'XJTU_SY_features.csv'
 with temp.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(sorted(records,key=lambda r:(r['condition'],r['run'],r['cycle'])))
 os.replace(temp,target)
 manifest={'benchmark':'XJTU-SY','source':'https://biaowang.tech/xjtu-sy-bearing-datasets/','citation':'Wang, B., Lei, Y., Li, N., & Li, N. (2020). A hybrid prognostics approach for estimating remaining useful life of rolling element bearings. IEEE Transactions on Reliability, 69(1), 401–412. https://doi.org/10.1109/TR.2018.2882682','feature_dataset_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'records':len(records),'bearings':len(groups),'sample_rate_Hz':FS,'samples_per_snapshot':N,'sampling_interval_minutes':1,'archive':json.loads((RAW/'download_manifest.json').read_text()),'extraction':'Whole snapshot per channel; no overlap or augmentation; FFT Hann window, geometry-independent shaft-order bands','target':'max recorded cycle - current cycle, minutes to recording endpoint','redistribution_note':'Derived features and hashes supplied for this research; raw archive preserved separately; cite original publisher; no commercial license assertion'}
 (DATA/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('FEATURE DATASET COMPLETE',len(records),len(records[0]),manifest['feature_dataset_sha256'],flush=True)

if __name__ == '__main__':
 main()
