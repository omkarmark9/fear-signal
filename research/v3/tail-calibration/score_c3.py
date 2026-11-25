"""Score addendum candidate C3 (mean-preserving within-day shrinkage) on dev WF; 2026 post-hoc."""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/tmp/claude-0/v3/tail-calibration')
import json, pickle
import numpy as np, pandas as pd
from common import build, q_and_pred
from fearsignal import data, features, stats
c=build()
F=pickle.load(open('/tmp/claude-0/v3/shared/fold_models_S1.pkl','rb'))['folds']
def logit(p): p=np.clip(p,1e-9,1-1e-9); return np.log(p/(1-p))
def sig(z): return 1/(1+np.exp(-z))
def ll(q,y): q=np.clip(q,1e-9,1-1e-9); return -(y*np.log(q)+(1-y)*np.log(1-q))

def shrink(q,b):
    """q (T,N) with NaN where no forecast; mean-preserving per day in logit space."""
    fin=np.isfinite(q); L=np.where(fin,logit(np.where(fin,q,0.5)),np.nan)
    Lbar=np.nanmean(L,1,keepdims=True); m=np.nanmean(q,1,keepdims=True)
    base=Lbar+b*(L-Lbar); cc=np.zeros_like(Lbar)
    for _ in range(50):
        p=sig(cc+base); f=np.nanmean(p,1,keepdims=True)-m; g=np.nanmean(p*(1-p),1,keepdims=True)
        st=np.where(g>0,f/np.maximum(g,1e-12),0); st=np.nan_to_num(st); cc-=st
        if np.nanmax(np.abs(st))<1e-12: break
    return np.where(fin,sig(cc+base),np.nan)

GRID=np.round(np.arange(0.05,1.5001,0.01),2)
def fit_b(q,y,mask):
    best=None
    for b in GRID:
        qs=shrink(q,b); v=ll(qs[mask],y[mask]).sum()
        if best is None or v<best[1]: best=(b,v)
    return best[0]

y=c['dev_y']; b_=c['dev_block']; q0=c['dev_q']; T,N=q0.shape
dates=pd.DatetimeIndex(c['dev_dates'])
ok=np.isfinite(q0)&np.isfinite(y)&np.isfinite(c['dev_qewma'])&(b_[:,None]>=0)
day=np.broadcast_to(np.arange(T)[:,None],(T,N))
obs=features.compute(data.load_dev_panel()).obs
q3=q0.copy(); bs=[]
for k,f in enumerate(F):
    qf,_,_=q_and_pred(f['model'],obs,f['zero_rate'])
    tr=(dates<=pd.Timestamp(f['train_end']))[:,None]&np.isfinite(qf)&np.isfinite(y)
    bk=fit_b(np.where((dates<=pd.Timestamp(f['train_end']))[:,None],qf,np.nan),y,tr); bs.append(float(bk))
    # transform over the fold's full-panel forecasts, then take its test days (day means use all stocks with a forecast)
    qs=shrink(qf,bk); te=ok&(b_[:,None]==k); q3[te]=qs[te]
    print(f'fold {k}: b={bk:.2f}  (train stock-days {tr.sum()})')

def evaluate(q,ref,m):
    out={'n':int(m.sum()),'brier':float(np.mean((q[m]-y[m])**2)),'logloss':float(np.mean(ll(q[m],y[m]))),'mean_q':float(q[m].mean())}
    out['cal']=stats.logistic_calibration(q[m],y[m],groups=day[m])
    for nm,Lf in (('brier',lambda p:(p-y)**2),('logloss',lambda p:ll(p,y))):
        d=np.where(m,Lf(ref)-Lf(q),np.nan); has=m.any(1); daily=np.nanmean(d[has],axis=1)
        mu=float(daily.mean()); se=stats.newey_west_se_mean(daily,5)
        out[f'gain_{nm}']=mu; out[f'gain_{nm}_se']=se; out[f'gain_{nm}_t']=mu/se if se>0 else 0.0
    return out
res={}
for tag,m in (('ABC',ok),('BC',ok&(b_[:,None]>=1)),('fold0',ok&(b_[:,None]==0)),('fold1',ok&(b_[:,None]==1)),('fold2',ok&(b_[:,None]==2))):
    r=evaluate(q3,q0,m); r0=evaluate(q0,q0,m); res[tag]=r
    print(f'{tag:5s} C3 brier={r["brier"]:.6f} (C0 {r0["brier"]:.6f}) dBrier={r["gain_brier"]*1e5:+.3f}e-5 (se {r["gain_brier_se"]*1e5:.3f}, t {r["gain_brier_t"]:+.2f}) '
          f'dLL={r["gain_logloss"]*1e4:+.3f}e-4 (se {r["gain_logloss_se"]*1e4:.3f}, t {r["gain_logloss_t"]:+.2f}) slope={r["cal"]["slope"]:.3f}({r["cal"]["se_slope"]:.3f}) [C0 {r0["cal"]["slope"]:.3f}] int={r["cal"]["intercept"]:.3f}')
# stationary bootstrap (days) of pooled daily gains as a robustness check
rng=np.random.default_rng(0)
for nm,Lf in (('brier',lambda p:(p-y)**2),('logloss',lambda p:ll(p,y))):
    d=np.where(ok,Lf(q0)-Lf(q3),np.nan); has=ok.any(1); daily=np.nanmean(d[has],axis=1)
    bo=[daily[stats.stationary_bootstrap_indices(len(daily),10,rng)].mean() for _ in range(2000)]
    print(f'  stationary-bootstrap (block 10) {nm} gain: {daily.mean():.3e}  95% CI [{np.quantile(bo,.025):.3e}, {np.quantile(bo,.975):.3e}]  boot se {np.std(bo):.3e}')

# ---- 2026 POST-HOC (diagnostic only)
qh,yh=c['h_q'],c['h_y']; Th,Nh=qh.shape; hd=pd.DatetimeIndex(c['h_dates'])
devm=(hd<=pd.Timestamp('2025-12-31'))
trm=devm[:,None]&np.isfinite(qh)&np.isfinite(yh)
bf=fit_b(np.where(devm[:,None],qh,np.nan),yh,trm)
qh3=shrink(qh,bf)
okh=np.isfinite(qh)&np.isfinite(yh)&c['h_mask'][:,None]&np.isfinite(c['h_qewma'])&np.isfinite(c['h_qewma2'])
dayh=np.broadcast_to(np.arange(Th)[:,None],(Th,Nh))
clim=float(c['h_clim'])
for nm,q in (('C0',qh),('C3',qh3)):
    cal=stats.logistic_calibration(q[okh],yh[okh],groups=dayh[okh])
    bri=np.mean((q[okh]-yh[okh])**2)
    d=np.where(okh,(qh-yh)**2-(q-yh)**2,np.nan); has=okh.any(1); daily=np.nanmean(d[has],1); t=daily.mean()/stats.newey_west_se_mean(daily,5) if nm!='C0' else 0
    d2=np.where(okh,ll(qh,yh)-ll(q,yh),np.nan); daily2=np.nanmean(d2[has],1); t2=daily2.mean()/stats.newey_west_se_mean(daily2,5) if nm!='C0' else 0
    print(f'2026 POST-HOC {nm} (frozen b={bf:.2f}): brier={bri:.6f} BSS_clim={1-bri/np.mean((clim-yh[okh])**2):+.4f} logloss={np.mean(ll(q[okh],yh[okh])):.5f} slope={cal["slope"]:.3f}({cal["se_slope"]:.3f}) int={cal["intercept"]:.3f} dBrier_t={t:+.2f} dLL_t={t2:+.2f}')
    print('   ',[(x['bin'],x['n'],round(x['mean_forecast'],4),round(x['observed'],4)) for x in stats.reliability_table(q[okh],yh[okh])])
json.dump({'b_by_fold':bs,'frozen_b':float(bf),'dev':res},open('/tmp/claude-0/v3/tail-calibration/score_c3_results.json','w'),indent=1,default=float)
