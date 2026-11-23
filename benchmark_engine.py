"""Reproducible, checkpointed XJTU-SY experiments; no model runs at import."""
from pathlib import Path
import os,json,hashlib,time,threading,shutil,platform,sys,contextlib
import numpy as np
import pandas as pd
import joblib
import importlib.metadata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,f1_score
import xgboost as xgb

DEFAULT_CONFIG={'seed':42,'n_jobs':4,'rf_trees':200,'xgb_rounds':300,'baseline_snapshots':10,'checkpoint_every':25,'protocol_version':'1.1','variants':['basic','engineered','engineered_baseline_normalised']}
CONTEXT=['elapsed_minutes','rpm','load_kN']
def hash_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8');os.replace(tmp,path)
def atomic_joblib(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');joblib.dump(value,tmp,compress=1);os.replace(tmp,path)
def atomic_csv(path,df):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');df.to_csv(tmp,index=False);os.replace(tmp,path)
class DriveMirror:
 """Copy completed local files atomically to a user-mounted Drive path."""
 def __init__(self,local,drive=None,interval=30):
  self.local=Path(local);self.drive=Path(drive) if drive else None;self.interval=interval;self.stop_event=threading.Event();self.thread=None;self.last_error=None
 def sync(self):
  if self.drive is None:return
  self.drive.mkdir(parents=True,exist_ok=True)
  try:
   for p in self.local.rglob('*'):
    if not p.is_file() or '.tmp' in p.name or '.lock' in p.name:continue
    rel=p.relative_to(self.local);q=self.drive/rel
    if q.exists() and q.stat().st_size==p.stat().st_size and q.stat().st_mtime>=p.stat().st_mtime:continue
    q.parent.mkdir(parents=True,exist_ok=True);temp=q.with_name(q.name+'.tmp');shutil.copy2(p,temp);os.replace(temp,q)
   self.last_error=None
  except Exception as e:
   self.last_error=str(e);atomic_json(self.local/'drive_sync_error.json',{'error':str(e),'time':time.time()});print('Drive backup failed; local checkpoints remain:',e,flush=True)
 def start(self):
  self.sync()
  def loop():
   while not self.stop_event.wait(self.interval):self.sync()
  self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
 def close(self):
  self.stop_event.set()
  if self.thread:self.thread.join(timeout=5)
  self.sync()

def load_dataset(path):
 df=pd.read_csv(path).sort_values(['condition','run','cycle']).reset_index(drop=True)
 assert df.bearing_id.nunique()==15
 assert not df.duplicated(['bearing_id','cycle']).any()
 assert df.groupby('bearing_id').rul_minutes.min().eq(0).all()
 assert (df.groupby('bearing_id').apply(lambda g:np.diff(g.cycle).min(),include_groups=False)==1).all()
 assert (df.rul_minutes>=0).all()
 return df
def prepare_features(df,cfg):
 stats=[c for c in df if c.startswith(('h_','v_'))]+['channel_rms_ratio']
 basic=[c for c in stats if c in [p+n for p in ['h_','v_'] for n in ['mean','std','rms']]]
 X=df[CONTEXT+stats].astype(float).replace([np.inf,-np.inf],np.nan)
 assert not X.isna().any().any()
 normal=X.copy();n=cfg['baseline_snapshots']
 for bearing,g in df.groupby('bearing_id'):
  initial=g.sort_values('cycle').head(n).index;b=X.loc[initial,stats];med=b.median();mad=(b-med).abs().median();scale=np.maximum(np.maximum(mad,.05*med.abs()),1e-6)
  normal.loc[g.index,stats]=(X.loc[g.index,stats]-med)/scale
 variants={'basic':X[CONTEXT+basic],'engineered':X,'engineered_baseline_normalised':normal}
 # The first n snapshots are calibration history, not prediction/evaluation examples.
 eligible=df.cycle.gt(n)
 return variants,eligible
def equal_bearing_weights(df,idx):
 sizes=df.loc[idx,'bearing_id'].value_counts();w=df.loc[idx,'bearing_id'].map(lambda b:1/sizes[b]).to_numpy();return w/w.mean()
def per_bearing(df,idx,pred):
 d=df.loc[idx,['bearing_id','condition','run','cycle','elapsed_minutes','rul_minutes','observed_lifetime_minutes']].copy();d['prediction']=np.maximum(0,np.asarray(pred));records=[]
 for b,g in d.groupby('bearing_id'):
  y=g.rul_minutes.to_numpy();p=g.prediction.to_numpy();truth=y<=30;alarm=p<=30
  records.append({'bearing_id':b,'condition':int(g.condition.iloc[0]),'n_snapshots':len(g),'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p),'endpoint_precision_30':precision_score(truth,alarm,zero_division=0),'endpoint_recall_30':recall_score(truth,alarm,zero_division=0),'endpoint_F1_30':f1_score(truth,alarm,zero_division=0)})
 return pd.DataFrame(records),d
def macro_score(df,idx,p):return float(per_bearing(df,idx,p)[0].MAE.mean())
def candidates():
 return [('ridge_a1','ridge',{'alpha':1.}),('ridge_a10','ridge',{'alpha':10.}),('rf_d8_l3','rf',{'max_depth':8,'min_samples_leaf':3}),('rf_full_l5','rf',{'max_depth':None,'min_samples_leaf':5}),('xgb_d2','xgb',{'max_depth':2,'eta':.05}),('xgb_d4','xgb',{'max_depth':4,'eta':.03})]
class InterruptedTraining(Exception):pass
def train_model(kind,params,X,y,w,path,cfg,interrupt_after=None):
 """Resume forests in tree batches and XGBoost from an atomic JSON checkpoint."""
 path=Path(path);path.mkdir(parents=True,exist_ok=True);complete=path/'complete.json';final=path/'model.joblib'
 if complete.exists() and final.exists():
  meta=json.loads(complete.read_text());assert hash_file(final)==meta['model_sha256'],'Completed model checksum mismatch'
  return joblib.load(final)
 start=time.perf_counter();process_start=time.process_time();xv=np.asarray(X,dtype=float);yv=np.log1p(np.asarray(y));w=np.asarray(w)
 if kind=='ridge':
  scaler=StandardScaler().fit(xv,sample_weight=w);model=Ridge(**params).fit(scaler.transform(xv),yv,sample_weight=w);bundle={'kind':kind,'model':model,'scaler':scaler,'features':list(X.columns)}
 elif kind=='rf':
  checkpoint=path/'forest_checkpoint.joblib'
  model=joblib.load(checkpoint) if checkpoint.exists() else RandomForestRegressor(n_estimators=0,warm_start=True,random_state=cfg['seed'],n_jobs=cfg['n_jobs'],**params)
  existing=len(getattr(model,'estimators_',[]))
  for trees in range(existing+cfg['checkpoint_every'],cfg['rf_trees']+cfg['checkpoint_every'],cfg['checkpoint_every']):
   model.n_estimators=min(trees,cfg['rf_trees']);model.fit(xv,yv,sample_weight=w);atomic_joblib(checkpoint,model)
   atomic_json(path/'progress.json',{'completed_trees':model.n_estimators,'target_trees':cfg['rf_trees'],'time':time.time()})
   if interrupt_after and model.n_estimators>=interrupt_after:raise InterruptedTraining('Deliberate crash-resume verification')
  bundle={'kind':kind,'model':model,'features':list(X.columns)}
 else:
  dm=xgb.DMatrix(xv,label=yv,weight=w,feature_names=list(X.columns));checkpoint=path/'boost_checkpoint.json'
  booster=xgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None
  existing=booster.num_boosted_rounds() if booster else 0
  parameters={'objective':'reg:squarederror','eval_metric':'rmse','tree_method':'hist','seed':cfg['seed'],'nthread':cfg['n_jobs'],'subsample':1.,'colsample_bytree':1.,**params}
  for step in range(existing,cfg['xgb_rounds'],cfg['checkpoint_every']):
   booster=xgb.train(parameters,dm,num_boost_round=min(cfg['checkpoint_every'],cfg['xgb_rounds']-step),xgb_model=booster)
   temp=path/'boost_checkpoint.tmp.json';booster.save_model(temp);os.replace(temp,checkpoint)
   atomic_json(path/'progress.json',{'completed_rounds':booster.num_boosted_rounds(),'target_rounds':cfg['xgb_rounds'],'time':time.time()})
   if interrupt_after and booster.num_boosted_rounds()>=interrupt_after:raise InterruptedTraining('Deliberate crash-resume verification')
  bundle={'kind':kind,'model':booster,'features':list(X.columns)}
 atomic_joblib(final,bundle)
 atomic_json(complete,{'kind':kind,'params':params,'wall_seconds_this_process':time.perf_counter()-start,'cpu_seconds_this_process':time.process_time()-process_start,'model_sha256':hash_file(final),'resumed':bool(existing) if kind!='ridge' else False})
 return bundle
def predict_log(bundle,X):
 a=np.asarray(X[bundle['features']],dtype=float)
 if bundle['kind']=='xgb':return bundle['model'].predict(xgb.DMatrix(a,feature_names=bundle['features']))
 if bundle['kind']=='ridge':a=bundle['scaler'].transform(a)
 return bundle['model'].predict(a)
def predict(bundle,X):return np.maximum(0,np.expm1(np.minimum(predict_log(bundle,X),15)))
def fit_scope(df,variants,eligible,out,cfg,scope,trainmask,valmask,testmask,finalmask):
 out=Path(out);scope_dir=out/'models'/scope;selections=[];all_predictions=[];all_metrics=[];search=[]
 ti=df.index[eligible&trainmask];vi=df.index[eligible&valmask];ei=df.index[eligible&testmask];fi=df.index[eligible&finalmask]
 assert set(df.loc[ti].bearing_id).isdisjoint(set(df.loc[vi].bearing_id))
 assert set(df.loc[fi].bearing_id).isdisjoint(set(df.loc[ei].bearing_id))
 atomic_json(out/'results'/f'{scope}_split.json',{'train':sorted(df.loc[ti].bearing_id.unique()),'validation':sorted(df.loc[vi].bearing_id.unique()),'final_fit':sorted(df.loc[fi].bearing_id.unique()),'test':sorted(df.loc[ei].bearing_id.unique())})
 for variant in cfg['variants']:
  X=variants[variant];scores=[]
  for name,kind,params in candidates():
   path=scope_dir/variant/'selection'/name
   bundle=train_model(kind,params,X.loc[ti],df.loc[ti,'rul_minutes'],equal_bearing_weights(df,ti),path,cfg)
   score=macro_score(df,vi,predict(bundle,X.loc[vi]));entry={'scope':scope,'variant':variant,'candidate':name,'kind':kind,'params':params,'validation_macro_MAE':score};scores.append((score,name,kind,params,bundle));search.append(entry)
   atomic_json(out/'results'/f'{scope}_search.json',search)
   print(scope,variant,name,'validation macro MAE',round(score,3),flush=True)
  score,name,kind,params,valbundle=min(scores,key=lambda t:(t[0],t[1]))
  # Bands are diagnostic only; temporal dependence invalidates standard iid guarantees.
  residual=np.abs(df.loc[vi,'rul_minutes'].to_numpy()-predict(valbundle,X.loc[vi]));q=float(np.quantile(residual,.9,method='higher'))
  bundle=train_model(kind,params,X.loc[fi],df.loc[fi,'rul_minutes'],equal_bearing_weights(df,fi),scope_dir/variant/'final',cfg)
  pred=predict(bundle,X.loc[ei]);metric,predframe=per_bearing(df,ei,pred);predframe['variant']=variant;predframe['scope']=scope;predframe['lower']=np.maximum(0,pred-q);predframe['upper']=pred+q
  predframe['covered']=(predframe.rul_minutes>=predframe.lower)&(predframe.rul_minutes<=predframe.upper)
  metric['variant']=variant;metric['scope']=scope;metric['model']=kind
  for b in metric.bearing_id:metric.loc[metric.bearing_id.eq(b),'band_coverage']=predframe.loc[predframe.bearing_id.eq(b),'covered'].mean()
  selections.append({'variant':variant,'candidate':name,'kind':kind,'params':params,'validation_macro_MAE':score,'residual_band_halfwidth':q,'n_features':X.shape[1]})
  all_metrics.append(metric);all_predictions.append(predframe)
 # Training-only baselines, evaluated on the same eligible rows.
 train=df.loc[fi];constant=float(train.groupby('bearing_id').rul_minutes.mean().mean())
 medlife=train.groupby('condition').apply(lambda g:g.groupby('bearing_id').observed_lifetime_minutes.first().median(),include_groups=False)
 fallback=float(train.groupby('bearing_id').observed_lifetime_minutes.first().median())
 for name,pred in [('constant',np.full(len(ei),constant)),('age_prior',np.maximum(0,df.loc[ei,'condition'].map(medlife).fillna(fallback).to_numpy()-df.loc[ei,'elapsed_minutes'].to_numpy()))]:
  metric,predframe=per_bearing(df,ei,pred);metric['variant']=name;metric['scope']=scope;metric['model']='baseline';predframe['variant']=name;predframe['scope']=scope;all_metrics.append(metric);all_predictions.append(predframe)
 metrics=pd.concat(all_metrics,ignore_index=True);predictions=pd.concat(all_predictions,ignore_index=True)
 atomic_csv(out/'results'/f'{scope}_metrics.csv',metrics);atomic_csv(out/'results'/f'{scope}_predictions.csv',predictions);atomic_json(out/'results'/f'{scope}_selection.json',selections)
 return metrics,predictions,selections

def explain_xgb(df,variants,eligible,out,cfg):
 """Explain the main engineered XGBoost candidate, regardless of winner family."""
 out=Path(out);variant='engineered';X=variants[variant];ti=df.index[eligible&df.run.le(3)];vi=df.index[eligible&df.run.eq(4)];fi=df.index[eligible&df.run.le(4)];ei=df.index[eligible&df.run.eq(5)]
 scores=[]
 for name,kind,params in candidates():
  if kind!='xgb':continue
  b=train_model(kind,params,X.loc[ti],df.loc[ti,'rul_minutes'],equal_bearing_weights(df,ti),out/'models'/'main'/variant/'selection'/name,cfg)
  scores.append((macro_score(df,vi,predict(b,X.loc[vi])),name,params))
 _,name,params=min(scores)
 b=train_model('xgb',params,X.loc[fi],df.loc[fi,'rul_minutes'],equal_bearing_weights(df,fi),out/'models'/'main'/'explain_xgb',cfg)
 # Evenly spaced held-out examples per bearing avoid one long trajectory dominating.
 selected=[]
 for _,g in df.loc[ei].groupby('bearing_id'):selected.extend(g.index[np.linspace(0,len(g)-1,min(100,len(g))).astype(int)])
 selected=pd.Index(selected);dm=xgb.DMatrix(np.asarray(X.loc[selected]),feature_names=list(X.columns))
 values=b['model'].predict(dm,pred_contribs=True,approx_contribs=False);raw=b['model'].predict(dm,output_margin=True)
 error=float(np.max(np.abs(values.sum(axis=1)-raw)));assert error<1e-3,error
 sv=pd.DataFrame(values[:,:-1],columns=X.columns,index=selected)
 global_rank=sv.abs().mean().sort_values(ascending=False).rename('mean_abs_SHAP_log1p_minutes').rename_axis('feature').reset_index()
 atomic_csv(out/'results'/'shap_global.csv',global_rank)
 atomic_csv(out/'results'/'shap_values.csv',pd.concat([df.loc[selected,['bearing_id','condition','cycle','rul_minutes']],sv.add_prefix('SHAP_')],axis=1).reset_index(drop=True))
 ranks=[]
 for c in sorted(df.condition.unique()):
  rank=sv.loc[df.loc[selected,'condition'].eq(c)].abs().mean().rank(ascending=False);ranks.append(rank.rename(f'condition_{c}'))
 atomic_csv(out/'results'/'shap_condition_ranks.csv',pd.concat(ranks,axis=1).reset_index(names='feature'))
 atomic_json(out/'results'/'shap_audit.json',{'model':name,'scope':'Main held-out run-5 bearings, engineered XGBoost; not necessarily best overall family','max_additivity_error':error,'n_examples':len(selected),'explanation_output':'log1p remaining minutes; contributions do not add in minute units','background':'Tree-path-dependent distribution encoded by training leaf counts; no test-background fitting','causal':False})

def maintenance_scenarios(df,out):
 out=Path(out);pred=pd.read_csv(out/'results'/'main_predictions.csv');outcomes=[]
 for variant,frame in pred.groupby('variant'):
  for b,g in frame.groupby('bearing_id'):
   g=g.sort_values('cycle');alarms=g[g.prediction<=30];first=alarms.iloc[0] if len(alarms) else None
   remaining=float(first.rul_minutes) if first is not None else 0.;late=first is None or remaining<5
   fraction=remaining/float(g.observed_lifetime_minutes.iloc[0]) if first is not None else 0.
   outcomes.append({'variant':variant,'bearing_id':b,'alarm_cycle':int(first.cycle) if first is not None else None,'remaining_at_alarm_minutes':remaining,'late_or_no_alarm':int(late),'discarded_life_fraction':fraction})
 outcomes=pd.DataFrame(outcomes);atomic_csv(out/'results'/'maintenance_outcomes.csv',outcomes);scenarios=[]
 for c in [2,5,10]:
  for w in [0,.5,1]:
   for variant,g in outcomes.groupby('variant'):
    costs=np.where(g.late_or_no_alarm,c,1)+w*g.discarded_life_fraction
    scenarios.append({'variant':variant,'assumed_unplanned_cost_ratio':c,'assumed_discarded_life_weight':w,'mean_assumed_cost':float(costs.mean()),'late_or_no_alarm_count':int(g.late_or_no_alarm.sum()),'mean_discarded_life_fraction':float(g.discarded_life_fraction.mean()),'n_bearings':len(g)})
 atomic_csv(out/'results'/'maintenance_scenarios.csv',pd.DataFrame(scenarios))

def plots(out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'figure.dpi':160,'savefig.dpi':240})
 out=Path(out);figdir=out/'figures';figdir.mkdir(exist_ok=True)
 metrics=pd.concat([pd.read_csv(p) for p in (out/'results').glob('*_metrics.csv')],ignore_index=True)
 summary=metrics.groupby(['scope','variant'])[['MAE','RMSE']].mean().reset_index();atomic_csv(out/'results'/'macro_summary.csv',summary)
 main=summary[summary.scope.eq('main')].sort_values('MAE');fig,ax=plt.subplots(figsize=(7,3.6));ax.barh(main.variant,main.MAE,color='#4a6572');ax.set_xlabel('Macro MAE (minutes, lower is better)');ax.set_title('Unseen-bearing test performance');ax.invert_yaxis();fig.tight_layout();fig.savefig(figdir/'main_mae.png');plt.close(fig)
 pred=pd.read_csv(out/'results'/'main_predictions.csv');variants=['basic','engineered','engineered_baseline_normalised'];bearings=sorted(pred.bearing_id.unique());fig,axes=plt.subplots(len(bearings),1,figsize=(7,8),squeeze=False)
 for ax,b in zip(axes[:,0],bearings):
  g=pred[pred.bearing_id.eq(b)];truth=g[g.variant.eq(variants[0])];ax.plot(truth.elapsed_minutes,truth.rul_minutes,'k--',label='Recorded-endpoint RUL')
  for v in variants:
   q=g[g.variant.eq(v)];ax.plot(q.elapsed_minutes,q.prediction,label=v,linewidth=1)
  ax.set_title(b);ax.set_ylabel('RUL (minutes)')
 axes[-1,0].set_xlabel('Observed elapsed minutes');axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/'main_rul_trajectories.png');plt.close(fig)
 s=pd.read_csv(out/'results'/'shap_global.csv').head(12).iloc[::-1];fig,ax=plt.subplots(figsize=(7,4));ax.barh(s.feature,s.iloc[:,1],color='#586f7c');ax.set_xlabel('Mean |Tree SHAP| in log(1 + remaining minutes)');ax.set_title('Held-out explanations of the engineered XGBoost model');fig.tight_layout();fig.savefig(figdir/'shap_importance.png');plt.close(fig)
 loco=summary[~summary.scope.eq('main')];fig,ax=plt.subplots(figsize=(7,4))
 for v in variants:
  g=loco[loco.variant.eq(v)];ax.plot(g.scope,g.MAE,'o-',label=v)
 ax.set_ylabel('Macro MAE (minutes)');ax.set_title('Unseen operating condition');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/'condition_transfer.png');plt.close(fig)
 policy=pd.read_csv(out/'results'/'maintenance_outcomes.csv').groupby('variant')[['late_or_no_alarm','discarded_life_fraction']].mean();fig,ax=plt.subplots(figsize=(7,4));ax.scatter(policy.discarded_life_fraction,policy.late_or_no_alarm)
 for v,r in policy.iterrows():ax.annotate(v,(r.discarded_life_fraction,r.late_or_no_alarm),fontsize=7,xytext=(4,4),textcoords='offset points')
 ax.set_xlabel('Discarded-life fraction (scenario proxy)');ax.set_ylabel('Fraction late or without alarm');ax.set_title('Maintenance trade-off on three test bearings');fig.tight_layout();fig.savefig(figdir/'maintenance_tradeoff.png');plt.close(fig)

def verify_resume(df,variants,eligible,out,cfg):
 idx=df.index[eligible&df.run.eq(1)][:120];X=variants['basic'].loc[idx];y=df.loc[idx,'rul_minutes'];w=equal_bearing_weights(df,idx);results=[]
 for kind,params in [('rf',{'max_depth':4,'min_samples_leaf':3}),('xgb',{'max_depth':2,'eta':.05})]:
  c={**cfg,'rf_trees':50,'xgb_rounds':50,'checkpoint_every':25};p=Path(out)/'resume_verification'/kind
  try:train_model(kind,params,X,y,w,p/'interrupted',c,interrupt_after=25)
  except InterruptedTraining:pass
  resumed=train_model(kind,params,X,y,w,p/'interrupted',c);fresh=train_model(kind,params,X,y,w,p/'fresh',c)
  error=float(np.max(np.abs(predict_log(resumed,X)-predict_log(fresh,X))));assert error<1e-6,(kind,error);results.append({'kind':kind,'max_prediction_difference':error,'passed':True})
 atomic_json(Path(out)/'results'/'resume_verification.json',results)

def run_all(dataset_path,output_dir,config=None,drive_path=None):
 cfg={**DEFAULT_CONFIG,**(config or {})};out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);(out/'results').mkdir(exist_ok=True)
 config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib']};fingerprint={'config':cfg,'config_sha256':config_hash,'dataset_sha256':hash_file(dataset_path),'python':sys.version,'platform':platform.platform(),'package_versions':versions}
 manifest=out/'run_manifest.json'
 if manifest.exists():
  old=json.loads(manifest.read_text());assert old['config_sha256']==config_hash and old['dataset_sha256']==fingerprint['dataset_sha256'],'Different data/config: choose a new output directory'
  assert old.get('package_versions')==versions,'Dependency versions changed; restore the recorded environment before loading checkpoints'
 else:atomic_json(manifest,fingerprint)
 mirror=DriveMirror(out,drive_path).start()
 try:
  df=load_dataset(dataset_path);variants,eligible=prepare_features(df,cfg)
  inventory=df.groupby(['bearing_id','condition','run']).agg(snapshots=('cycle','size'),endpoint_minutes=('elapsed_minutes','max')).reset_index();atomic_csv(out/'results'/'bearing_inventory.csv',inventory)
  atomic_json(out/'results'/'feature_inventory.json',{k:list(v.columns) for k,v in variants.items()})
  verify_resume(df,variants,eligible,out,cfg)
  fit_scope(df,variants,eligible,out,cfg,'main',df.run.le(3),df.run.eq(4),df.run.eq(5),df.run.le(4));mirror.sync()
  for c in [1,2,3]:
   source=df.condition.ne(c);fit_scope(df,variants,eligible,out,cfg,f'loco_{c}',source&df.run.le(3),source&df.run.eq(4),~source,source);mirror.sync()
  explain_xgb(df,variants,eligible,out,cfg);maintenance_scenarios(df,out);plots(out)
  atomic_json(out/'completed.json',{'success':True,'completed_time':time.time(),'config_sha256':config_hash,'dataset_sha256':fingerprint['dataset_sha256']})
 finally:mirror.close()
 return pd.read_csv(out/'results'/'macro_summary.csv')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);args=p.parse_args();print(run_all(args.data,args.output).to_string(index=False))
