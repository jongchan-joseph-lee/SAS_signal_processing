"""Point-ROI cases for the deep autofocus benchmark (v4).

Same geometry, taper, BP and sway injection as evaluate_autofocus_v3.stacks.
Additions:
  * additive complex white noise on the range-compressed channels before the
    Hann taper (band-limited after taper); level set as raw SNR in dB against
    the RMS of the whole scene00 range-compressed record
  * signal and unit-noise ping stacks are returned separately; because BP is
    linear, stack(signal + a*noise) = stack_signal + a*stack_noise
  * linear-phase registered NCC (best along-track shift), since constant and
    linear ping phase are not observable by a blind estimator
"""
import numpy as np
from evaluate_possm import load_scene
from imaging import resample_complex, complex_metrics
from imaging_v2 import bp
from diagnose_taper import taper
from stress_points_v2 import navigation_path
from autofocus_v3 import ping_stack, image
from protocol_v3 import ROOT

POINTS_CSV = ROOT / 'references/Scatterer_Locations_Scene00.csv'
SLOPES = np.linspace(-np.pi, np.pi, 721)


def scene(name='scene00'):
    q, tx, rx, c = load_scene(ROOT, name)
    return q, tx, rx, c, np.loadtxt(POINTS_CSV, delimiter=',', skiprows=1)


def scene_rms(q):
    s = 0.0
    for i in range(0, len(q), 200):
        s += float(np.sum(np.abs(np.asarray(q[i:i + 200], np.complex128)) ** 2))
    return np.sqrt(s / (q.shape[0] * q.shape[1]))


def pools(pts):
    """Test = v3 held-out rows (<60). Training pool = rows >= 60 with the same isolation rule."""
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)); np.fill_diagonal(d, 1e9)
    ok = (d.min(1) > 1) & (pts[:, 0] > 10) & (pts[:, 0] < 50) & (pts[:, 1] > 15) & (pts[:, 1] < 80)
    return np.flatnonzero(ok & (np.arange(len(pts)) >= 60))


def case(q, tx, rx, c, point, seed, amount, noise_seed=None, rms=1.0, clean=True):
    """Return dict with clean image, signal stack, unit-noise stack (std rms per raw sample), ping ids, grid."""
    x = (np.floor(point[0] / .005) + np.arange(-40, 41)) * .005
    y = (np.floor(point[1] / .005) + np.arange(-40, 41)) * .005
    sel = (tx[:, 0] >= x.min() - y.max() * np.tan(.05) - .11) & (tx[:, 0] <= x.max() + y.max() * np.tan(.05) + .11)
    obs = np.flatnonzero(sel); p = obs // 36; ts = tx[sel].copy(); rs = rx[sel].copy(); cs = c[sel]
    qs = np.asarray(q[sel])
    qr = resample_complex(taper(qs, 'hann60'), 8)
    out = {'x': x, 'y': y}
    if clean:
        out['clean'] = bp(qr, ts, rs, cs, x, y, 75000., 8, 1)
    if amount:
        dd = amount * navigation_path(seed)[p, 1]; ts[:, 1] += dd; rs[:, 1] += dd
    st = ping_stack(qr, ts, rs, cs, p - p.min(), x, y)
    e = np.sum(np.abs(st) ** 2, axis=(1, 2)); ids = np.flatnonzero(e > max(e.max() * 1e-9, 1e-30))
    out['signal'] = np.ascontiguousarray(st[ids]); out['ids'] = ids + p.min()
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        n = (rng.standard_normal(qs.shape) + 1j * rng.standard_normal(qs.shape)) * (rms / np.sqrt(2))
        nr = resample_complex(taper(n.astype(np.complex64), 'hann60'), 8)
        out['noise'] = np.ascontiguousarray(ping_stack(nr, ts, rs, cs, p - p.min(), x, y)[ids])
    return out


def noise_scale(snr_db):
    return 0.0 if snr_db is None else 10 ** (-snr_db / 20)


def ncc(clean, z):
    return complex_metrics(clean, z)['complex_correlation']


def registered_ncc(clean, stack, ids, phase):
    """Max complex NCC over an added linear ping phase (= along-track shift); FFT-free grid search."""
    k = (ids - ids.mean()).astype(float)
    a = clean.ravel(); na = np.vdot(a, a).real
    flat = stack.reshape(len(stack), -1)
    rot = np.exp(-1j * (phase[None, :] + SLOPES[:, None] * k[None, :]))    # slopes x pings
    z = rot @ flat                                                       # slopes x pixels
    num = np.abs(z @ np.conj(a)); den = np.sqrt(na * np.sum(np.abs(z) ** 2, 1) + 1e-30)
    i = int(np.argmax(num / den))
    return float(num[i] / den[i]), float(SLOPES[i])


def oracle_phase(signal_corrupt, signal_clean):
    return np.angle(np.einsum('pyx,pyx->p', signal_corrupt, np.conj(signal_clean)))
