"""Mechanism checks (2026 diagnostics; dev OOS for contrast)."""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/tmp/claude-0/v3/tail-calibration')
import numpy as np, pandas as pd
from scipy.special import ndtr
from common import build
from fearsignal.features import EXACT, CENSORED, ZERO
c=build()
pd.set_option('display.width',220)

def S(which):
    if which=='dev':
        b=c['dev_block']; ok=np.isfinite(c['dev_q'])&np.isfinite(c['dev_y'])&np.isfinite(c['dev_qewma'])&(b[:,None]>=0)
        import pickle; F=pickle.load(open('/tmp/claude-0/v3/shared/fold_models_S1.pkl','rb'))['folds']
        s=np.array([f['model'].emission.s for f in F])[b.clip(0)]; z0=np.array([f['zero_rate'] for f in F])[b.clip(0)]
        return dict(q=c['dev_q'],y=c['dev_y'],pred=c['dev_pred'],x=c['dev_x'],kind=c['dev_kind'],ok=ok,s=s,z0=z0,
                    sig=c['dev_sigma'],r=c['dev_r'],dates=pd.DatetimeIndex(c['dev_dates']),block=b)
    T=len(c['h_dates'])
    ok=np.isfinite(c['h_q'])&np.isfinite(c['h_y'])&c['h_mask'][:,None]&np.isfinite(c['h_qewma'])&np.isfinite(c['h_qewma2'])
    return dict(q=c['h_q'],y=c['h_y'],pred=c['h_pred'],x=c['h_x'],kind=c['h_kind'],ok=ok,
                s=np.broadcast_to(c['frozen_s'],(T,3)),z0=np.full(T,float(c['frozen_z0'])),
                sig=c['h_sigma'],r=c['h_r'],dates=pd.DatetimeIndex(c['h_dates']),block=np.where(c['h_mask'],9,-1))

for which in ('dev','2026'):
    D=S(which); q,y,pred,x,kind,ok=D['q'],D['y'],D['pred'],D['x'],D['kind'],D['ok']
    T,N=q.shape
    print(f'\n######## {which}')
    # (a) threshold profile by predictive panic mass group
    xx=np.where(kind==CENSORED,10.0,np.where(kind==EXACT,x,0.0))
    pp=np.nan_to_num(pred[...,2])
    groups=[('pred_panic<0.03',ok&(pp<0.03)),('0.03-0.10',ok&(pp>=0.03)&(pp<0.10)),('0.10-0.25',ok&(pp>=0.10)&(pp<0.25)),('>=0.25',ok&(pp>=0.25))]
    rows=[]
    for nm,m in groups:
        r=[nm,m.sum(),len(np.unique(np.nonzero(m)[0]))]
        for th in (1,2,3,4,6):
            tp=(1-D['z0'])[:,None,None]*2*ndtr(-th/D['s'][:,None,:])     # (T,1,K)
            f=(np.nan_to_num(pred)*tp).sum(2)
            o=(xx>=th)
            r+= [f[m].mean(), o[m].mean()]
        rows.append(r)
    cols=['group','n','ndays']+[f'{a}{th}' for th in (1,2,3,4,6) for a in ('f','o')]
    print(pd.DataFrame(rows,columns=cols).to_string(index=False,float_format='%.4f'))
    # (b) days after a market shock day (share of eligible stocks with event >= 20%)
    el=np.isfinite(y)
    share=np.nansum(y,1)/np.maximum(el.sum(1),1)
    shock=share>=0.20
    for h in (1,2,3,5):
        aft=np.zeros(T,bool); aft[h:]=shock[:-h]
        m=ok&aft[:,None]
        dd=np.unique(np.nonzero(m)[0])
        if m.sum(): print(f'  h={h} days after shock: ndays={len(dd)} meanq={q[m].mean():.4f} obs={y[m].mean():.4f}  per-day obs shares={np.round(np.nanmean(np.where(ok,y,np.nan)[dd],1),3)}')
    near=np.zeros(T,bool)
    for t0 in np.flatnonzero(shock): near[t0+1:t0+6]=True
    m=ok&~near[:,None]
    print(f'  not within 5 days after a shock: meanq={q[m].mean():.4f} obs={y[m].mean():.4f}')
    print('  shock days:',[str(d.date()) for d in D['dates'][shock & (D['block']>=0)]])
    # (c) volatility level vs sigma: 21-day robust scale of r over sigma
    ar=pd.DataFrame(np.abs(D['r']))
    s21=1.4826*ar.rolling(21,min_periods=15).median().to_numpy()
    ratio=s21/D['sig']
    rm=pd.Series(np.nanmedian(np.where(ok,ratio,np.nan),1),index=D['dates'])[D['block']>=0]
    print('  monthly median (21d robust scale / 126d sigma):', rm.resample('QE').mean().round(3).to_dict())
    print('  mean x on scored eligible days: %.3f   share ZERO %.3f'%(np.nanmean(np.where(ok,xx,np.nan)),(kind[ok]==ZERO).mean()))
