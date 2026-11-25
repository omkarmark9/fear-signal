"""2026 DIAGNOSTICS (post-hoc, not selection evidence) + same on dev OOS for contrast."""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/tmp/claude-0/v3/tail-calibration')
import numpy as np, pandas as pd
from common import build
from fearsignal import stats
c=build()

def setup(which):
    if which=='dev':
        q,y,pred,x,kind=c['dev_q'],c['dev_y'],c['dev_pred'],c['dev_x'],c['dev_kind']
        ok=np.isfinite(q)&np.isfinite(y)&np.isfinite(c['dev_qewma'])&(c['dev_block'][:,None]>=0)
        dates=pd.DatetimeIndex(c['dev_dates']); qe=c['dev_qewma']
        tk=c['dev_tailk'][c['dev_block'].clip(0)]   # (T,K) per-day fold tail probs
    else:
        q,y,pred,x,kind=c['h_q'],c['h_y'],c['h_pred'],c['h_x'],c['h_kind']
        ok=np.isfinite(q)&np.isfinite(y)&c['h_mask'][:,None]&np.isfinite(c['h_qewma'])&np.isfinite(c['h_qewma2'])
        dates=pd.DatetimeIndex(c['h_dates']); qe=c['h_qewma']
        tk=np.broadcast_to(c['h_tailk'],(len(dates),3))
    T,N=q.shape
    day=np.broadcast_to(np.arange(T)[:,None],(T,N))
    return dict(q=q,y=y,pred=pred,x=x,kind=kind,ok=ok,dates=dates,day=day,qe=qe,tk=tk)

for which in ('dev','2026'):
    S=setup(which); q,y,ok,day,pred=S['q'],S['y'],S['ok'],S['day'],S['pred']
    print(f'\n================ {which}  n={ok.sum()} days={np.unique(day[ok]).size} event={y[ok].mean():.4f} meanq={q[ok].mean():.4f}')
    cal=stats.logistic_calibration(q[ok],y[ok],groups=day[ok]); print('HMM  cal slope %.3f (%.3f) int %.3f'%(cal['slope'],cal['se_slope'],cal['intercept']))
    cal=stats.logistic_calibration(S['qe'][ok],y[ok],groups=day[ok]); print('EWMA cal slope %.3f (%.3f) int %.3f'%(cal['slope'],cal['se_slope'],cal['intercept']))
    # reliability with per-bin day-clustered s.e. of (obs - forecast)
    bins=[0,.01,.02,.035,.05,.075,.1,.2,1]
    rows=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=ok&(q>=lo)&(q<hi)
        if m.sum()==0: continue
        d=pd.DataFrame({'day':day[m],'r':y[m]-q[m]}).groupby('day')['r'].sum()
        # cluster se of mean residual
        se=np.sqrt((d**2).sum())/m.sum()
        rows.append((f'[{lo},{hi})',m.sum(),len(d),q[m].mean(),y[m].mean(),(y[m].mean()-q[m].mean())/se if se>0 else np.nan))
    print(pd.DataFrame(rows,columns=['bin','n','ndays','meanq','obs','t_resid_clust']).to_string(index=False,float_format='%.4f'))
    # by dominant predictive state
    am=np.argmax(np.nan_to_num(pred),axis=2)
    for k,nm in enumerate(['calm','stressed','panic']):
        m=ok&(am==k)
        d=pd.DataFrame({'day':day[m],'r':y[m]-q[m]}).groupby('day')['r'].sum(); se=np.sqrt((d**2).sum())/m.sum()
        print(f'argmax pred={nm:8s} n={m.sum():6d} days={len(d):4d} meanq={q[m].mean():.4f} obs={y[m].mean():.4f} t={(y[m].mean()-q[m].mean())/se:.2f}')
    # effective state tail probabilities: y ~ sum_k pred_k * theta_k (no intercept), OLS with day clusters
    P=pred[ok]; yy=y[ok]; g=day[ok]
    XtX=P.T@P; th=np.linalg.solve(XtX,P.T@yy); e=yy-P@th
    sc=P*e[:,None]; _,gi=np.unique(g,return_inverse=True); G=np.zeros((gi.max()+1,3)); np.add.at(G,gi,sc)
    V=np.linalg.inv(XtX)@(G.T@G)@np.linalg.inv(XtX)
    print('implied state tail probs theta (clustered se):',np.round(th,4),np.round(np.sqrt(np.diag(V)),4))
    print('model state tail probs (mean over scored days):',np.round(S['tk'][np.unique(g)].mean(0),4))
    # predictive state mass
    print('mean pred mass calm/stressed/panic:',np.round(P.mean(0),4))
