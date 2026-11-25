import sys; sys.dont_write_bytecode=True; sys.path.insert(0,'/tmp/claude-0/v3/tail-calibration')
import numpy as np
from common import build
from fearsignal import stats
c=build()
# dev OOS
q,y,qe,qe_cl=c['dev_q'],c['dev_y'],c['dev_qewma'],c['dev_clim']
T,N=q.shape
ok=np.isfinite(q)&np.isfinite(y)&np.isfinite(qe)&np.isfinite(qe_cl)&(c['dev_block'][:,None]>=0)
day=np.broadcast_to(np.arange(T)[:,None],(T,N))
print('dev n',ok.sum(),'brier',np.mean((q[ok]-y[ok])**2),'clim',np.mean((qe_cl[ok]-y[ok])**2))
print('dev cal',stats.logistic_calibration(q[ok],y[ok],groups=day[ok]))
# 2026
q,y=c['h_q'],c['h_y']; T,N=q.shape
ok=np.isfinite(q)&np.isfinite(y)&c['h_mask'][:,None]&np.isfinite(c['h_qewma'])&np.isfinite(c['h_qewma2'])
day=np.broadcast_to(np.arange(T)[:,None],(T,N))
print('2026 n',ok.sum(),'brier',np.mean((q[ok]-y[ok])**2), 'evt',y[ok].mean(),'meanq',q[ok].mean())
print('2026 cal',stats.logistic_calibration(q[ok],y[ok],groups=day[ok]))
print('tailk dev folds',c['dev_tailk'],'frozen',c['h_tailk'])
