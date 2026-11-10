"""Unrolled learned autofocus for per-ping phase error (v4).

Each iteration looks at every ping against the leave-one-out composite image:
  m_k(x) = c_k(x) * conj(z(x) - c_k(x)),   c_k = s_k exp(-i phi_k)
A shared CNN encodes [Re m, Im m, |c_k|, |z - c_k|] per ping; a small
transformer across pings mixes the evidence (smooth sway prior) and returns a
phase update per ping. Gauge (constant + linear phase) is projected out.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gauge_project(phi, mask):
    """Remove the masked least-squares constant + linear trend over valid pings."""
    k = torch.arange(phi.shape[1], dtype=phi.dtype, device=phi.device)[None].expand_as(phi)
    w = mask.to(phi.dtype); n = w.sum(1, keepdim=True).clamp(min=1)
    km = (w * k).sum(1, keepdim=True) / n; kc = (k - km) * w
    pm = (w * phi).sum(1, keepdim=True) / n
    b = (kc * phi).sum(1, keepdim=True) / (kc * kc).sum(1, keepdim=True).clamp(min=1e-9)
    return (phi - pm - b * (k - km)) * w


class Encoder(nn.Module):
    def __init__(self, ch=(24, 48, 64, 96)):
        super().__init__(); layers = []; c0 = 4
        for c in ch:
            layers += [nn.Conv2d(c0, c, 3, stride=2, padding=1), nn.GroupNorm(4, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1), nn.GELU()]; c0 = c
        self.net = nn.Sequential(*layers); self.out = 2 * c0

    def forward(self, x):
        h = self.net(x)
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)


class DeepAutofocus(nn.Module):
    def __init__(self, iterations=4, d=96, heads=4, layers=2, max_pings=24):
        super().__init__()
        self.iterations = iterations; self.enc = Encoder()
        self.inp = nn.Linear(self.enc.out + 3, d)
        self.pos = nn.Parameter(torch.zeros(max_pings, d)); nn.init.normal_(self.pos, std=0.02)
        self.step = nn.Embedding(8, d)
        self.mix = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 2)

    @staticmethod
    def features(s, phi, mask):
        """s: (B,P,H,W) complex; phi: (B,P); returns per-ping maps (B,P,4,H,W) and global stats (B,P,3)."""
        c = s * torch.exp(-1j * phi)[..., None, None]
        z = c.sum(1, keepdim=True); loo = z - c
        m = c * loo.conj()
        a = m.abs().mean((2, 3), keepdim=True).clamp(min=1e-12)
        r = (z.abs() ** 2).mean((1, 2, 3), keepdim=True).sqrt().clamp(min=1e-12)
        maps = torch.stack([(m / a).real, (m / a).imag, c.abs() / r, loo.abs() / r], 2)
        tot = m.sum((2, 3)); g = tot / tot.abs().clamp(min=1e-12)
        e = (s.abs() ** 2).sum((2, 3)); e = e / e.sum(1, keepdim=True).clamp(min=1e-12)
        stats = torch.stack([g.real, g.imag, e.sqrt()], -1) * mask[..., None]
        return maps * mask[..., None, None, None], stats

    def forward(self, s, mask, iterations=None):
        B, P = mask.shape; phi = torch.zeros(B, P, device=mask.device); out = []
        for t in range(iterations or self.iterations):
            maps, stats = self.features(s, phi, mask)
            v = self.enc(maps.reshape(B * P, *maps.shape[2:])).reshape(B, P, -1)
            h = self.inp(torch.cat([v, stats], -1)) + self.pos[:P] + self.step.weight[min(t, 7)]
            h = self.mix(h, src_key_padding_mask=~mask)
            u = self.head(h); delta = torch.atan2(u[..., 1], u[..., 0])
            phi = gauge_project(phi + delta * mask, mask); out.append(phi)
        return out


def d2_loss(phi, theta, mask, energy):
    """Gauge-invariant loss on the second difference of the phase error, weighted by ping energy."""
    e = phi - theta
    d2 = e[:, 2:] - 2 * e[:, 1:-1] + e[:, :-2]
    m = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    w = (energy[:, 2:] * energy[:, 1:-1] * energy[:, :-2]).clamp(min=0) ** (1 / 3) * m
    return ((1 - torch.cos(d2)) * w).sum() / w.sum().clamp(min=1e-12)


def n_params(m): return sum(p.numel() for p in m.parameters())


if __name__ == '__main__':
    m = DeepAutofocus(); print('params', n_params(m))
    s = torch.randn(2, 10, 81, 81, dtype=torch.complex64); mask = torch.ones(2, 10, dtype=torch.bool); mask[1, 7:] = False
    out = m(s, mask); print([o.shape for o in out])
