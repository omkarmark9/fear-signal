"""Between-day vs within-day calibration; fold-FE slope; design effect; top-bin deficit concentration.
2026 parts are post-hoc diagnostics only."""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/tmp/claude-0/v3/tail-calibration')
import numpy as np, pandas as pd
from common import build
c=build()

def logit(p): p=np.clip(p,1e-9,1-1e-9); return np.log(p/(1-p))
def glm(X,y,g,iters=60):
    beta=np.zeros(X.shape[1]); beta[0]=logit(y.mean())
    for _ in range(iters):
        mu=1/(1+np.exp(-(X@beta))); W=mu*(1-mu); H=X.T@(X*W[:,None]); st=np.linalg.solve(H,X.T@(y-mu)); beta+=st
        if np.abs(st).max()<1e-10: break
    mu=1/(1+np.exp(-(X@beta))); sc=X*(y-mu)[:,None]; _,gi=np.unique(g,return_inverse=True)
    G=np.zeros((gi.max()+1,X.shape[1])); np.add.at(G,gi,sc); Hi=np.linalg.inv(H); V=Hi@(G.T@G)@Hi
    return beta,np.sqrt(np.diag(V))

def sets():
    b=c['dev_block']; q,y=c['dev_q'],c['dev_y']; T,N=q.shape
    ok=np.isfinite(q)&np.isfinite(y)&np.isfinite(c['dev_qewma'])&(b[:,None]>=0)
    yield 'dev ABC',q,y,ok,np.broadcast_to(b[:,None],(T,N))
    for k in range(3): yield f'dev fold{k}',q,y,ok&(b[:,None]==k),np.broadcast_to(b[:,None],(T,N))
    q,y=c['h_q'],c['h_y']; T,N=q.shape
    ok=np.isfinite(q)&np.isfinite(y)&c['h_mask'][:,None]&np.isfinite(c['h_qewma'])&np.isfinite(c['h_qewma2'])
    yield '2026 (post-hoc)',q,y,ok,np.zeros((T,N),int)

for nm,q,y,ok,fold in sets():
    T,N=q.shape
    day=np.broadcast_to(np.arange(T)[:,None],(T,N))
    L=np.where(ok,logit(q),np.nan)
    Lbar=np.nanmean(L,axis=1,keepdims=True)*np.ones((1,N))
    m=ok; g=day[m]
    X=np.column_stack([np.ones(m.sum()),Lbar[m],(L-Lbar)[m]])
    be,se=glm(X,y[m],g)
    line=f'{nm:16s} between-day slope {be[1]:.3f} ({se[1]:.3f})  within-day slope {be[2]:.3f} ({se[2]:.3f})'
    # common slope with fold fixed intercepts
    fl=fold[m]; uf=np.unique(fl)
    if len(uf)>1:
        Xf=np.column_stack([(fl==u).astype(float) for u in uf]+[L[m]])
        bf,sf=glm(Xf,y[m],g); line+=f'  | fold-FE common slope {bf[-1]:.3f} ({sf[-1]:.3f})'
    # design effect of the daily event share
    has=m.any(1); n_d=m[has].sum(1); s_d=np.nansum(np.where(m,y,0),1)[has]/n_d
    pbar=y[m].mean(); deff=np.var(s_d)/np.mean(pbar*(1-pbar)/n_d)
    # residual design effect after conditioning on q: var of daily mean residual vs binomial var
    rd=np.nansum(np.where(m,y-q,0),1)[has]/n_d; vb=np.nansum(np.where(m,q*(1-q),0),1)[has]/n_d**2
    line+=f'  | design effect raw {deff:.1f}, residual {np.mean(rd**2)/np.mean(vb):.1f}'
    print(line)

# top-bin deficit concentration in 2026 (post-hoc) and fold C (dev)
D=pd.read_csv('/tmp/claude-0/v3/tail-calibration/h2026_daily.csv')
D['deficit']=D.exp_hi-D.ev_hi
Ds=D.sort_values('deficit',ascending=False)
print('\n2026 q>=0.075: total expected-observed = %.1f; top-6 days contribute %.1f:'%(D.deficit.sum(),Ds.deficit.head(6).sum()))
print(Ds.head(6)[['date','n_hi','exp_hi','ev_hi','deficit']].to_string(index=False,float_format='%.2f'))
# fold C
q,y,b=c['dev_q'],c['dev_y'],c['dev_block']; dates=pd.DatetimeIndex(c['dev_dates'])
ok=np.isfinite(q)&np.isfinite(y)&np.isfinite(c['dev_qewma'])&(b[:,None]==2)
hi=ok&(q>=0.075)
rows=[(dates[t].date(),int(hi[t].sum()),q[t,hi[t]].sum(),int(y[t,hi[t]].sum())) for t in np.flatnonzero(hi.any(1))]
F=pd.DataFrame(rows,columns=['date','n_hi','exp_hi','ev_hi']); F['deficit']=F.exp_hi-F.ev_hi
print('\ndev fold C (2025H2) q>=0.075: days %d, stock-days %d, expected %.1f, observed %d'%(len(F),F.n_hi.sum(),F.exp_hi.sum(),F.ev_hi.sum()))
print(F.sort_values('deficit',ascending=False).head(6).to_string(index=False,float_format='%.2f'))
