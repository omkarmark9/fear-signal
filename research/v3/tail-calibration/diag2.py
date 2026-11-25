"""Per-fold dev OOS calibration; 2026 per-day influence (post-hoc diagnostics)."""
import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/tmp/claude-0/v3/tail-calibration')
import numpy as np, pandas as pd
from common import build
from fearsignal import stats
c=build()
# ---- dev per fold
q,y,b=c['dev_q'],c['dev_y'],c['dev_block']; T,N=q.shape
day=np.broadcast_to(np.arange(T)[:,None],(T,N))
ok=np.isfinite(q)&np.isfinite(y)&np.isfinite(c['dev_qewma'])&(b[:,None]>=0)
for k in range(3):
    m=ok&(b[:,None]==k)
    cal=stats.logistic_calibration(q[m],y[m],groups=day[m])
    bs=np.mean((q[m]-y[m])**2); bc=np.mean((c['dev_clim'][m]-y[m])**2)
    print(f'fold {k}: n={m.sum()} event={y[m].mean():.4f} meanq={q[m].mean():.4f} slope={cal["slope"]:.3f}({cal["se_slope"]:.3f}) int={cal["intercept"]:.3f} BSS={1-bs/bc:+.4f} clim={c["dev_clim"][m][0]:.4f}')
# ---- 2026 per-day
q,y=c['h_q'],c['h_y']; T,N=q.shape
dates=pd.DatetimeIndex(c['h_dates'])
ok=np.isfinite(q)&np.isfinite(y)&c['h_mask'][:,None]&np.isfinite(c['h_qewma'])&np.isfinite(c['h_qewma2'])
day=np.broadcast_to(np.arange(T)[:,None],(T,N))
rows=[]
for t in np.flatnonzero(ok.any(1)):
    m=ok[t]
    rows.append(dict(date=dates[t].date(),n=m.sum(),meanq=q[t,m].mean(),events=int(y[t,m].sum()),exp=q[t,m].sum(),
                     n_hi=int((q[t,m]>=0.075).sum()),ev_hi=int(y[t,m][q[t,m]>=0.075].sum()),exp_hi=q[t,m][q[t,m]>=0.075].sum()))
D=pd.DataFrame(rows)
# leave-one-day-out slope
qv,yv,gv=q[ok],y[ok],day[ok]
full=stats.logistic_calibration(qv,yv)['slope']
lodo=[]
for t in np.unique(gv):
    m=gv!=t
    lodo.append(stats.logistic_calibration(qv[m],yv[m],iters=30)['slope'])
D['slope_drop']=np.array(lodo)
D['infl']=D['slope_drop']-full
print('full slope',full,'LODO range',D.slope_drop.min(),D.slope_drop.max())
pd.set_option('display.width',200)
print('most influential days (dropping lowers slope most -> day supports higher slope; dropping raises slope most -> day drags slope down):')
print(D.sort_values('infl').head(8).to_string(index=False,float_format='%.4f'))
print(D.sort_values('infl').tail(8).to_string(index=False,float_format='%.4f'))
# top-bin day concentration
hi=D[D.n_hi>0]
print('days with any q>=0.075:',len(hi),' stock-days',hi.n_hi.sum(),' events',hi.ev_hi.sum(),' expected',round(hi.exp_hi.sum(),1))
print(hi.sort_values('n_hi',ascending=False).head(12).to_string(index=False,float_format='%.3f'))
D.to_csv('/tmp/claude-0/v3/tail-calibration/h2026_daily.csv',index=False)
# remove top-k days by top-bin stock-days and recompute slope
for drop in ([],):
    pass
