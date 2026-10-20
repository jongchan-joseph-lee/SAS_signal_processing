"""CPU coherent backprojection for complex baseband measurements."""
import numpy as np
from numba import njit, prange
from scipy import signal, fft


def pulse_compress(iq,kernel,mode="convolution"):
    """Apply the supplied GDF KBID=1 time-domain matched-filter kernel.

    GDF names KBID=1 a matched-filter kernel. For this release its samples
    require conjugate reversal (mode='correlation'). The convention and
    timing are checked on fixed scene00 development scatterers and recorded.
    """
    if mode=="correlation":
        kernel=np.conj(kernel[::-1])
    length=iq.shape[-1]+len(kernel)-1
    nfft=fft.next_fast_len(length)
    kernel_f=fft.fft(kernel,nfft)
    out=fft.ifft(fft.fft(iq,nfft,axis=-1)*kernel_f,axis=-1)[...,:length]
    return np.ascontiguousarray(out,dtype=np.complex64)


@njit(parallel=True,cache=True)
def backproject(q,tx,rx,speeds,x,y,z,fs,fc,sample_offset,phase_sign,
                fov_deg=16.,phase_errors=None,receiver_velocity=None):
    image=np.zeros((len(y),len(x)),np.complex128)
    tan_half=np.tan(fov_deg*np.pi/360)
    m_count=q.shape[0]
    for iy in prange(len(y)):
        py=y[iy]
        for ix in range(len(x)):
            px=x[ix];value=0j
            for m in range(m_count):
                dx=px-tx[m,0];dy=py-tx[m,1]
                if dy<=0 or abs(dx)>abs(dy)*tan_half:
                    continue
                dz=z-tx[m,2]
                rt=np.sqrt(dx*dx+dy*dy+dz*dz)
                drx=px-rx[m,0];dry=py-rx[m,1];drz=z-rx[m,2]
                rr=np.sqrt(drx*drx+dry*dry+drz*drz)
                tau=(rt+rr)/speeds[m]
                if receiver_velocity is not None:
                    # Solve receive time for a receiver moving during the
                    # acoustic round trip, keeping the transmit phase centre
                    # at the nominal emission pose. Fixed-point contraction
                    # is |v|/c; three iterations are ample for this dataset.
                    for iteration in range(3):
                        vx=drx-receiver_velocity[m,0]*tau
                        vy=dry-receiver_velocity[m,1]*tau
                        vz=drz-receiver_velocity[m,2]*tau
                        rr=np.sqrt(vx*vx+vy*vy+vz*vz)
                        tau=(rt+rr)/speeds[m]
                u=tau*fs+sample_offset
                j=int(np.floor(u))
                if j<0 or j+1>=q.shape[1]:
                    continue
                # Linear complex interpolation at Fs is used after explicit
                # resampling; resampling ratio is recorded by the runner.
                alpha=u-j
                sample=q[m,j]*(1-alpha)+q[m,j+1]*alpha
                phase=phase_sign*2*np.pi*fc*tau
                if phase_errors is not None:
                    phase+=phase_errors[m]
                value+=sample*(np.cos(phase)+1j*np.sin(phase))
            image[iy,ix]=value
    return image


def resample_complex(q,ratio):
    return np.ascontiguousarray(signal.resample_poly(q,ratio,1,axis=-1),dtype=np.complex64)


def complex_metrics(reference,image):
    reference=np.asarray(reference).ravel()
    image=np.asarray(image).ravel()
    scale=np.vdot(image,reference)/(np.vdot(image,image)+1e-30)
    corr=abs(np.vdot(reference,image))/np.sqrt(np.vdot(reference,reference).real*np.vdot(image,image).real+1e-30)
    ref_mag=np.abs(reference);img_mag=np.abs(image)
    return {"complex_correlation":float(corr),
            "amplitude_correlation":float(np.corrcoef(ref_mag,img_mag)[0,1]),
            "aligned_nrmse":float(np.linalg.norm(scale*image-reference)/(np.linalg.norm(reference)+1e-30)),
            "scale_real":float(scale.real),"scale_imag":float(scale.imag)}
