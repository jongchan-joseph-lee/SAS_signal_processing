"""Explicit Gaussian beam and native-grid diagnostics; original v1 retained."""
import numpy as np
from numba import njit,prange


@njit(parallel=True,cache=True)
def bp(q,tx,rx,c,x,y,fs,ratio,beam=0,cap=.05,normalize=False,
       continuous=True,spreading=False,phase_reference=0.,sample_shift=0.):
    """beam: 0 uniform; 1 horizontal Gaussian; 2 horizontal+vertical.

    The recorded source element dimensions set Gaussian half-power widths.
    Vertical boresight = horizontal starboard is an explicit hypothesis,
    not an undocumented assertion about ASASIN. cap truncates only numerical
    support. normalize divides by summed squared beam/path response.
    """
    out=np.zeros((len(y),len(x)),np.complex128)
    ln2=np.log(2.)
    wavelength=1500./300000.
    ta=np.arcsin(.4429464707*wavelength/.05)
    ra=np.arcsin(.4429464707*wavelength/.0333)
    te=np.arcsin(.4429464707*wavelength/.0192)
    re=np.arcsin(.4429464707*wavelength/.0139)
    # This PoSSM-specific implementation requires monotonically sorted Tx X.
    # Conservative interval pruning preserves observation order and exact gate.
    tx_x=tx[:,0];tx_ymin=np.min(tx[:,1]);tx_ymax=np.max(tx[:,1])
    for iy in prange(len(y)):
        for ix in range(len(x)):
            val=0j;norm=0.
            margin=max(abs(y[iy]-tx_ymin),abs(y[iy]-tx_ymax))*np.tan(cap)
            first=np.searchsorted(tx_x,x[ix]-margin)
            last=np.searchsorted(tx_x,x[ix]+margin,side='right')
            for m in range(first,last):
                dx=x[ix]-tx[m,0];dy=y[iy]-tx[m,1];dz=-tx[m,2]
                if dy<=0 or abs(dx)>dy*np.tan(cap):continue
                rt=np.sqrt(dx*dx+dy*dy+dz*dz)
                ax=x[ix]-rx[m,0];ay=y[iy]-rx[m,1];az=-rx[m,2]
                rr=np.sqrt(ax*ax+ay*ay+az*az);tau=(rt+rr)/c[m]
                if continuous:
                    for _ in range(3):
                        rr=np.sqrt((ax-.6*tau)**2+ay*ay+az*az);tau=(rt+rr)/c[m]
                u=tau*fs*ratio+299*ratio+sample_shift*ratio
                j=int(np.floor(u))
                if j<0 or j+1>=q.shape[1]:continue
                alpha=u-j;sample=q[m,j]*(1-alpha)+q[m,j+1]*alpha
                weight=1.
                if beam:
                    a_tx=np.arctan2(dx,dy);a_rx=np.arctan2(ax-(.6*tau if continuous else 0.),ay)
                    exponent=(a_tx/ta)**2+(a_rx/ra)**2
                    if beam==2:
                        e_tx=np.arctan2(dz,np.sqrt(dx*dx+dy*dy))
                        e_rx=np.arctan2(az,np.sqrt((ax-(.6*tau if continuous else 0.))**2+ay*ay))
                        exponent+=(e_tx/te)**2+(e_rx/re)**2
                    weight=np.exp(-.5*ln2*exponent)
                if spreading:weight/=rt*rr
                phase=2*np.pi*300000.*(tau-phase_reference)
                val+=weight*sample*(np.cos(phase)+1j*np.sin(phase));norm+=weight*weight
            if normalize and norm>1e-25:val/=norm
            out[iy,ix]=val
    return out


def quadratic_peak(z,x,y):
    """Local quadratic fit of log power in a 3x3 patch; no GT in estimator."""
    p=np.abs(z)**2;iy,ix=np.unravel_index(p.argmax(),p.shape)
    dx=float(np.median(np.diff(x)));dy=float(np.median(np.diff(y)))
    result={'grid_x':float(x[ix]),'grid_y':float(y[iy]),'valid_subpixel':False}
    if 0<iy<len(y)-1 and 0<ix<len(x)-1:
        gy,gx=np.mgrid[-1:2,-1:2]
        design=np.column_stack([np.ones(9),gx.ravel(),gy.ravel(),gx.ravel()**2,gx.ravel()*gy.ravel(),gy.ravel()**2])
        fit=np.linalg.lstsq(design,np.log(np.maximum(p[iy-1:iy+2,ix-1:ix+2],p.max()*1e-15)).ravel(),rcond=None)[0]
        hessian=np.array([[2*fit[3],fit[4]],[fit[4],2*fit[5]]])
        if np.linalg.eigvalsh(hessian).max()<0:
            off=-np.linalg.solve(hessian,fit[1:3])
            if np.all(np.abs(off)<=1):
                result.update({'subpixel_x':float(x[ix]+off[0]*dx),'subpixel_y':float(y[iy]+off[1]*dy),'valid_subpixel':True})
    if not result['valid_subpixel']:result.update({'subpixel_x':float(x[ix]),'subpixel_y':float(y[iy])})
    return result


def psf_metrics(z,x,y):
    a=np.abs(z);iy,ix=np.unravel_index(a.argmax(),a.shape)
    def cut(line,grid,k):
        power=(line/line[k])**2
        lo=hi=k
        while lo>0 and power[lo]>.5:lo-=1
        while hi<len(line)-1 and power[hi]>.5:hi+=1
        left=np.interp(.5,[power[lo],power[min(lo+1,k)]],[grid[lo],grid[min(lo+1,k)]])
        right=np.interp(.5,[power[hi],power[max(hi-1,k)]],[grid[hi],grid[max(hi-1,k)]])
        # First local minima delimit the main lobe; if absent, censored.
        ml=k;mr=k
        while ml>1 and power[ml-1]<=power[ml]:ml-=1
        while mr<len(power)-2 and power[mr+1]<=power[mr]:mr+=1
        outside=np.r_[power[:ml],power[mr+1:]]
        pslr=10*np.log10(max(float(outside.max()) if len(outside) else 0.,1e-15))
        return {'width_3db_interpolated_m':float(right-left),'pslr_db':float(pslr),
                'mainlobe_samples':int(mr-ml+1),'coarse_sampling':bool(right-left<2*np.median(np.diff(grid)))}
    return {'x':cut(a[iy],x,ix),'y':cut(a[:,ix],y,iy),'peak':quadratic_peak(z,x,y)}
