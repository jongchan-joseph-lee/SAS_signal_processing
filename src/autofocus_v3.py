"""Original CPU GPGA-style and per-ping minimum-entropy implementations.

References: Wahl et al. (1994), doi:10.1109/7.303752; generalized BP
interpretation, doi:10.1109/TCI.2019.2899453 and Henrik Forsten's author
implementation notes. This is a local adaptation, not their source code.
No correct navigation, clean image, or target coordinates enter estimators.
"""
import numpy as np
from numba import njit,prange
from scipy.optimize import minimize


@njit(parallel=True,cache=True)
def ping_stack(q,tx,rx,c,ping,x,y,ratio=8):
 """Complex ping contributions; sum must match frozen v2 BP exactly."""
 out=np.zeros((ping.max()+1,len(y),len(x)),np.complex128)
 ta=np.arcsin(.4429464707*.005/.05);ra=np.arcsin(.4429464707*.005/.0333)
 for iy in prange(len(y)):
  for ix in range(len(x)):
   for m in range(len(tx)):
    dx=x[ix]-tx[m,0];dy=y[iy]-tx[m,1];dz=-tx[m,2]
    if dy<=0 or abs(dx)>dy*np.tan(.05):continue
    rt=np.sqrt(dx*dx+dy*dy+dz*dz)
    ax=x[ix]-rx[m,0];ay=y[iy]-rx[m,1];az=-rx[m,2]
    rr=np.sqrt(ax*ax+ay*ay+az*az);tau=(rt+rr)/c[m]
    for _ in range(3):
     rr=np.sqrt((ax-.6*tau)**2+ay*ay+az*az);tau=(rt+rr)/c[m]
    u=tau*75000.*ratio+299*ratio;j=int(np.floor(u))
    if j<0 or j+1>=q.shape[1]:continue
    a=u-j;v=q[m,j]*(1-a)+q[m,j+1]*a
    at=np.arctan2(dx,dy);ar=np.arctan2(ax-.6*tau,ay)
    w=np.exp(-.5*np.log(2.)*((at/ta)**2+(ar/ra)**2))
    ph=2*np.pi*300000.*tau
    out[ping[m],iy,ix]+=w*v*(np.cos(ph)+1j*np.sin(ph))
 return out


def active_stack(stack):
 e=np.sum(np.abs(stack)**2,axis=(1,2));ids=np.flatnonzero(e>max(e.max()*1e-9,1e-30))
 return np.ascontiguousarray(stack[ids]),ids


def gauge_matrix(n):
 a=np.column_stack([np.ones(n),np.linspace(-1,1,n)])
 return np.eye(n)-a@np.linalg.pinv(a)


def image(stack,phase):return np.einsum('p,pyx->yx',np.exp(-1j*phase),stack,optimize=False)


def entropy(z):
 p=np.abs(z)**2;p/=max(p.sum(),1e-30)
 return float(-np.sum(p*np.log(np.maximum(p,1e-30))))


def gpga(stack,iterations=3):
 """Local generalized PGA: strongest-pixel selection, adjacent products.

 Narrow PoSSM local apertures have only a few pings. Use no temporal
 low-pass filtering so the injected white phase component is not removed.
 Constant/linear ambiguity is explicitly projected out at every update.
 """
 n=len(stack);phase=np.zeros(n);project=gauge_matrix(n)
 for _ in range(iterations):
  z=image(stack,phase);iy,ix=np.unravel_index(np.abs(z).argmax(),z.shape)
  u=stack[:,iy,ix]*np.exp(-1j*phase)
  g=np.angle(np.conj(u[:-1])*u[1:]);delta=np.r_[0.,np.cumsum(g)]
  phase+=project@delta
 return image(stack,phase),phase


def objective(v,flat,project,regularization):
 ph=project@v;rot=np.exp(-1j*ph);z=rot@flat
 power=abs(z)**2;total=max(power.sum(),1e-30);p=power/total
 logp=np.log(np.maximum(p,1e-30));h=-np.sum(p*logp)
 dp=-(logp+h)/total
 # d |sum C exp(-i phi)|^2 / d phi = 2 Im(conj(I) C exp(-i phi)).
 grad=2*np.imag(rot*(flat@(dp*np.conj(z))))
 penalty=regularization*np.mean(ph*ph)
 grad+=2*regularization*ph/max(len(ph),1)
 return float(h+penalty),project@grad


def minimum_entropy(stack,regularization=.001,maxiter=100):
 n=len(stack);project=gauge_matrix(n);flat=np.ascontiguousarray(stack.reshape(n,-1))
 flat=flat/max(np.max(np.abs(flat)),1e-30)
 _,p0=gpga(stack,3);solutions=[]
 for start in [np.zeros(n),p0]:
  fit=minimize(objective,start,args=(flat,project,regularization),jac=True,method='L-BFGS-B',
   options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-7,'maxls':25})
  ph=project@fit.x;solutions.append((float(fit.fun),ph,fit))
 _,ph,fit=min(solutions,key=lambda r:r[0])
 return image(stack,ph),ph,{'iterations':int(fit.nit),'success':bool(fit.success),'message':str(fit.message)}
