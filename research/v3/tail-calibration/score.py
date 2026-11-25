"""Score declared candidates C0, C1, C2 on dev walk-forward OOS; then 2026 post-hoc."""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/tmp/claude-0/v3/tail-calibration')
import json, pickle
import numpy as np, pandas as pd
from scipy.optimize import nnls
from common import build, q_and_pred
from fearsignal import data, features, stats, model as mdl
c=build()
F=pickle.load(open('/tmp/claude-0/v3/shared/fold_models_S1.pkl','rb'))['folds']
OUT={}

def logit(p): p=np.clip(p,1e-9,1-1e-9); return np.log(p/(1-p))
def sig(z): return 1/(1+np.exp(-z))
def fit_logistic(q,y):
    r=stats.logistic_calibration(q,y); return r['intercept'],r['slope']
def ll(q,y): q=np.clip(q,1e-9,1-1e-9); return -(y*np.log(q)+(1-y)*np.log(1-q))

# ---------------- dev
q0,y,b=c['dev_q'],c['dev_y'],c['dev_block']; T,N=q0.shape
dates=pd.DatetimeIndex(c['dev_dates'])
ok=np.isfinite(q0)&np.isfinite(y)&np.isfinite(c['dev_qewma'])&(b[:,None]>=0)
day=np.broadcast_to(np.arange(T)[:,None],(T,N))
panel=data.load_dev_panel(); obs=features.compute(panel).obs

# C1: walk-forward stacking
q1=q0.copy(); c1_params=[]
for k in range(3):
    te=ok&(b[:,None]==k)
    prior=ok&(b[:,None]<k)&(b[:,None]>=0)
    if prior.sum()==0:
        c1_params.append((0.0,1.0)); continue
    a,s=fit_logistic(q0[prior],y[prior]); c1_params.append((a,s))
    q1[te]=sig(a+s*logit(q0[te]))
# C2: training-window NNLS state tail probabilities
q2=q0.copy(); c2_theta=[]
for k,f in enumerate(F):
    _,pred,tk=q_and_pred(f['model'],obs,f['zero_rate'])
    tr=(dates<=pd.Timestamp(f['train_end']))[:,None]&np.isfinite(pred[...,0])&np.isfinite(y)
    th,_=nnls(pred[tr],y[tr]); c2_theta.append(th)
    te=ok&(b[:,None]==k)
    q2[te]=(pred[te]@th)
OUT['C1_params_by_fold']=c1_params; OUT['C2_theta_by_fold']=[t.tolist() for t in c2_theta]
OUT['C0_theta_by_fold']=c['dev_tailk'].tolist()
print('C1 (a,b) by fold:',np.round(c1_params,3).tolist())
print('C2 theta by fold:',np.round(c2_theta,4).tolist(),' vs parametric',np.round(c['dev_tailk'],4).tolist())

def evaluate(q,ref,mask):
    m=mask
    out={'n':int(m.sum()),'brier':float(np.mean((q[m]-y[m])**2)),'logloss':float(np.mean(ll(q[m],y[m]))),
         'mean_q':float(q[m].mean()),'event':float(y[m].mean())}
    cal=stats.logistic_calibration(q[m],y[m],groups=day[m]); out['cal']=cal
    # paired daily gains vs reference (positive = candidate better)
    for nm,L in (('brier',lambda p:(p-y)**2),('logloss',lambda p:ll(p,y))):
        d=np.where(m,L(ref)-L(q),np.nan); has=m.any(1)
        daily=np.nanmean(d[has],axis=1)
        mu=float(daily.mean()); se=stats.newey_west_se_mean(daily,5)
        out[f'gain_{nm}_daily']=mu; out[f'gain_{nm}_se']=se; out[f'gain_{nm}_t']=mu/se if se>0 else 0.0
    return out

res={}
for nm,q in (('C0',q0),('C1',q1),('C2',q2)):
    res[nm]={'ABC':evaluate(q,q0,ok),'BC':evaluate(q,q0,ok&(b[:,None]>=1))}
    for k in range(3): res[nm][f'fold{k}']=evaluate(q,q0,ok&(b[:,None]==k))
OUT['dev']=res
def show(tag):
    rows=[]
    for nm in res:
        r=res[nm][tag]
        rows.append([nm,r['n'],r['brier'],r['logloss'],r['mean_q'],r['event'],r['cal']['slope'],r['cal']['se_slope'],r['cal']['intercept'],
                     r['gain_brier_daily']*1e5,r['gain_brier_se']*1e5,r['gain_brier_t'],r['gain_logloss_daily']*1e4,r['gain_logloss_se']*1e4,r['gain_logloss_t']])
    print(f'--- dev OOS {tag}')
    print(pd.DataFrame(rows,columns=['cand','n','brier','logloss','meanq','event','slope','se','int','dBrier_e5','se_e5','t','dLL_e4','se_e4','t']).to_string(index=False,float_format='%.4f'))
for tag in ('ABC','BC','fold0','fold1','fold2'): show(tag)

# ---------------- 2026 POST-HOC (diagnostic only; never used for selection)
qh,yh=c['h_q'],c['h_y']; Th,Nh=qh.shape
hd=pd.DatetimeIndex(c['h_dates'])
okh=np.isfinite(qh)&np.isfinite(yh)&c['h_mask'][:,None]&np.isfinite(c['h_qewma'])&np.isfinite(c['h_qewma2'])
dayh=np.broadcast_to(np.arange(Th)[:,None],(Th,Nh))
a,s=fit_logistic(q0[ok],y[ok])                         # frozen C1: all dev OOS folds
qh1=sig(a+s*logit(qh))
devd=(hd<=pd.Timestamp('2025-12-31'))[:,None]&np.isfinite(c['h_pred'][...,0])&np.isfinite(yh)
thf,_=nnls(c['h_pred'][devd],yh[devd])                 # frozen C2: frozen model, all dev stock-days
qh2=c['h_pred']@thf
print('\nfrozen C1 (a,b)=',round(a,3),round(s,3),' frozen C2 theta=',np.round(thf,4),' parametric',np.round(c['h_tailk'],4))
OUT['frozen_C1']=(a,s); OUT['frozen_C2_theta']=thf.tolist()
post={}
for nm,q in (('C0',qh),('C1',qh1),('C2',qh2)):
    m=okh
    r={'brier':float(np.mean((q[m]-yh[m])**2)),'logloss':float(np.mean(ll(q[m],yh[m]))),'mean_q':float(q[m].mean()),
       'bss_clim':1-np.mean((q[m]-yh[m])**2)/np.mean((float(c['h_clim'])-yh[m])**2),
       'cal':stats.logistic_calibration(q[m],yh[m],groups=dayh[m]),
       'reliability':stats.reliability_table(q[m],yh[m])}
    for lnm,L in (('brier',lambda p:(p-yh)**2),('logloss',lambda p:ll(p,yh))):
        d=np.where(m,L(qh)-L(q),np.nan); has=m.any(1); daily=np.nanmean(d[has],axis=1)
        se=stats.newey_west_se_mean(daily,5); r[f'gain_{lnm}_vs_C0']=float(daily.mean()); r[f'gain_{lnm}_t']=float(daily.mean()/se) if se>0 else 0.0
    post[nm]=r
    print(f'2026 POST-HOC {nm}: brier={r["brier"]:.6f} BSS_clim={r["bss_clim"]:+.4f} logloss={r["logloss"]:.5f} meanq={r["mean_q"]:.4f} slope={r["cal"]["slope"]:.3f}({r["cal"]["se_slope"]:.3f}) int={r["cal"]["intercept"]:.3f} dBrier_t={r["gain_brier_t"]:.2f} dLL_t={r["gain_logloss_t"]:.2f}')
    print('   ',[(x['bin'],x['n'],round(x['mean_forecast'],4),round(x['observed'],4)) for x in r['reliability']])
OUT['post_hoc_2026']=post
json.dump(OUT,open('/tmp/claude-0/v3/tail-calibration/score_results.json','w'),indent=1,default=float)
