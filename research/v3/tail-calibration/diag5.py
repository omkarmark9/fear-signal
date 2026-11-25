"""Idiosyncratic vs market-wide stress: is the within-day over-dispersion from transient stock-specific shocks?"""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/tmp/claude-0/v3/tail-calibration')
import numpy as np, pandas as pd
from common import build
c=build()
def logit(p): p=np.clip(p,1e-9,1-1e-9); return np.log(p/(1-p))
def grp(nm,q,y,ok,day):
    d=pd.DataFrame({'day':day[ok],'r':y[ok]-q[ok]}).groupby('day')['r'].sum(); se=np.sqrt((d**2).sum())/ok.sum()
    return f'{nm:44s} n={ok.sum():6d} days={len(d):4d} meanq={q[ok].mean():.4f} obs={y[ok].mean():.4f} ratio={y[ok].mean()/q[ok].mean():.2f} t={(y[ok].mean()-q[ok].mean())/se:+.2f}'
for which in ('dev','2026'):
    if which=='dev':
        q,y,x,kind=c['dev_q'],c['dev_y'],c['dev_x'],c['dev_kind']; b=c['dev_block']
        ok=np.isfinite(q)&np.isfinite(y)&np.isfinite(c['dev_qewma'])&(b[:,None]>=0)
    else:
        q,y,x,kind=c['h_q'],c['h_y'],c['h_x'],c['h_kind']
        ok=np.isfinite(q)&np.isfinite(y)&c['h_mask'][:,None]&np.isfinite(c['h_qewma'])&np.isfinite(c['h_qewma2'])
    T,N=q.shape; day=np.broadcast_to(np.arange(T)[:,None],(T,N))
    # yesterday's event (stock was hit at t-1) and yesterday's market event share
    yprev=np.vstack([np.full((1,N),np.nan),y[:-1]])
    share_prev=np.nanmean(yprev,axis=1)[:,None]*np.ones((1,N))
    L=np.where(ok,logit(q),np.nan); dev_=L-np.nanmean(L,1,keepdims=True)
    print(f'\n##### {which}')
    print(grp('hit yesterday, market share yesterday <10%',q,y,ok&(yprev==1)&(share_prev<0.10),day))
    print(grp('hit yesterday, market share yesterday >=25%',q,y,ok&(yprev==1)&(share_prev>=0.25),day))
    print(grp('not hit yesterday, share yesterday <10%',q,y,ok&(yprev==0)&(share_prev<0.10),day))
    print(grp('not hit yesterday, share yesterday >=25%',q,y,ok&(yprev==0)&(share_prev>=0.25),day))
    print(grp('q>=0.05 & logit-dev from day mean >=1',q,y,ok&(q>=0.05)&(dev_>=1),day))
    print(grp('q>=0.05 & logit-dev from day mean <0.5',q,y,ok&(q>=0.05)&(dev_<0.5),day))
