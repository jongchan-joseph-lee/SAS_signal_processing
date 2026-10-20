"""Small learned phase-correction baseline and blind entropy optimization.

This is an original compact baseline inspired by phase-estimation autofocus.
It is NOT a reproduction of Gerg & Monga's Deep Autofocus architecture.
Training uses procedural sparse complex scenes with known injected phase
errors. Public PoSSM images are held out from training and model selection.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

LIMITS=torch.tensor([6.,3.])


def basis(n,device=None):
    k=torch.fft.fftshift(torch.fft.fftfreq(n,device=device))*2
    b2=k*k-(k*k).mean()
    b3=k**3-(k**4).sum()/(k*k).sum()*k
    b3-=b3.mean()
    return torch.stack([b2/b2.abs().max(),b3/b3.abs().max()])


def phase_apply(z,coefficients,sign=1.):
    b=basis(z.shape[-1],z.device)
    phi=torch.einsum("bi,ik->bk",coefficients,b)
    spectrum=torch.fft.fftshift(torch.fft.fft(z,dim=-1),dim=-1)
    return torch.fft.ifft(torch.fft.ifftshift(spectrum*torch.exp(sign*1j*phi[:,None,:]),dim=-1),dim=-1)


def features(z):
    # Peak-centering and a global phase normalization reduce nuisance
    # position/global-phase variation without access to a clean image.
    n=z.shape[-1]
    indices=z.abs().square().flatten(1).argmax(1)
    centered=torch.stack([torch.roll(z[i],(n//2-int(indices[i]//n),n//2-int(indices[i]%n)),dims=(-2,-1)) for i in range(len(z))])
    anchor=centered[:,n//2,n//2]
    centered=centered*torch.exp(-1j*torch.angle(anchor))[:,None,None]
    scale=centered.abs().amax((-2,-1),keepdim=True).clamp_min(1e-8)
    centered=centered/scale
    return torch.stack([centered.real,centered.imag,torch.log1p(10*centered.abs())/np.log(11)],dim=1)


class PhaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,16,5,stride=2,padding=2),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv2d(32,48,3,stride=2,padding=1),nn.GroupNorm(8,48),nn.SiLU(),
            nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),
            nn.Linear(48*4*4,64),nn.SiLU(),nn.Linear(64,2),nn.Tanh())
        self.register_buffer("limits",LIMITS.clone())

    def forward(self,x):
        return self.encoder(x)*self.limits


def procedural(count,seed,n=64):
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:n,:n]
    clean=np.zeros((count,n,n),np.complex64)
    for i in range(count):
        for _ in range(int(rng.integers(1,9))):
            px,py=rng.uniform(6,n-6,2)
            sx,sy=rng.uniform(.6,1.4,2)
            amplitude=rng.uniform(.3,1.5)*np.exp(1j*rng.uniform(-np.pi,np.pi))
            clean[i]+=amplitude*np.exp(-.5*((x-px)**2/sx**2+(y-py)**2/sy**2))
        noise=(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))).astype(np.complex64)
        clean[i]+=noise*rng.uniform(.001,.012)
    coefficients=rng.uniform(-1,1,(count,2)).astype(np.float32)*LIMITS.numpy()
    clean=torch.from_numpy(clean)
    coefficients=torch.from_numpy(coefficients)
    return clean,phase_apply(clean,coefficients),coefficients


def entropy(z):
    power=z.abs().square()
    p=power/power.sum((-2,-1),keepdim=True).clamp_min(1e-20)
    return -(p*torch.log(p.clamp_min(1e-20))).sum((-2,-1))


def blind_entropy(z,steps=70,starts=(-3.,0.,3.)):
    """No reference/ground-truth image used in fitting or start selection."""
    best=None;best_coeff=None;best_h=None
    for start in starts:
        coeff=torch.zeros((len(z),2),dtype=torch.float32,requires_grad=True)
        with torch.no_grad():coeff[:,0]=start
        opt=torch.optim.Adam([coeff],lr=.15)
        for _ in range(steps):
            opt.zero_grad()
            h=entropy(phase_apply(z,coeff,-1))
            h.mean().backward()
            opt.step()
            with torch.no_grad():coeff.clamp_(-10,10)
        with torch.no_grad():
            restored=phase_apply(z,coeff,-1);h=entropy(restored)
            if best is None:
                best=restored.clone();best_coeff=coeff.detach().clone();best_h=h.clone()
            else:
                improve=h<best_h
                best[improve]=restored[improve]
                best_coeff[improve]=coeff.detach()[improve]
                best_h[improve]=h[improve]
    return best,best_coeff


def metrics(clean,z):
    a=clean.flatten(1);b=z.flatten(1)
    inner=(a.conj()*b).sum(1)
    corr=inner.abs()/(a.abs().square().sum(1)*b.abs().square().sum(1)).sqrt().clamp_min(1e-20)
    # Global complex scalar alignment only: no per-case translation fitting.
    scale=(b.conj()*a).sum(1)/b.abs().square().sum(1).clamp_min(1e-20)
    nrmse=(scale[:,None]*b-a).abs().square().sum(1).sqrt()/a.abs().square().sum(1).sqrt().clamp_min(1e-20)
    return {"complex_correlation":corr.cpu().numpy(),
            "aligned_nrmse":nrmse.cpu().numpy(),
            "entropy":entropy(z).cpu().numpy()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=60)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42)
    train_clean,train_bad,train_coeff=procedural(1024,1101)
    val_clean,val_bad,val_coeff=procedural(128,2202)
    train_x=features(train_bad);val_x=features(val_bad)
    model=PhaseNet()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float("inf");best_epoch=0;history=[];start=time.perf_counter()
    checkpoint=args.output/"phase_net.pt"
    for epoch in range(args.epochs):
        model.train();order=torch.randperm(len(train_x));losses=[]
        for batch in order.split(64):
            opt.zero_grad()
            pred=model(train_x[batch])
            loss=((pred-train_coeff[batch])/LIMITS).square().mean()
            loss.backward();opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():val_loss=float(((model(val_x)-val_coeff)/LIMITS).square().mean())
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":val_loss})
        if val_loss<best:
            best=val_loss;best_epoch=epoch+1
            torch.save({"state_dict":model.state_dict(),"epoch":best_epoch,"validation_loss":best,
                        "model":"PhaseNet","phase_limits":[6.,3.],"seed":42},checkpoint)
        if (epoch+1)%5==0:print(history[-1],flush=True)
        if epoch+1-best_epoch>=12:break
    training_seconds=time.perf_counter()-start
    model.load_state_dict(torch.load(checkpoint,weights_only=True)["state_dict"]);model.eval()
    # Test scenes and errors are generated only after validation selection.
    test_clean,test_bad,test_coeff=procedural(256,3303)
    with torch.no_grad():
        test_x=features(test_bad)
        start=time.perf_counter()
        pred=model(test_x);learned=phase_apply(test_bad,pred,-1)
        learned_seconds=time.perf_counter()-start
    start=time.perf_counter()
    classical,classical_coeff=blind_entropy(test_bad)
    classical_seconds=time.perf_counter()-start
    result={}
    arrays={}
    for label,z in [("corrupted",test_bad),("entropy",classical),("learned",learned)]:
        stats=metrics(test_clean,z.detach())
        arrays.update({label+"_"+k:v for k,v in stats.items()})
        result[label]={k:{"mean":float(v.mean()),"median":float(np.median(v))} for k,v in stats.items()}
    result.update({"training_scenes":1024,"validation_scenes":128,"test_scenes":256,
                   "split_seeds":{"train":1101,"validation":2202,"test":3303},
                   "best_epoch":best_epoch,"best_validation_loss":best,
                   "model_parameters":sum(p.numel() for p in model.parameters()),
                   "training_seconds":training_seconds,
                   "learned_forward_and_correction_seconds_256":learned_seconds,
                   "classical_optimization_seconds_256":classical_seconds,
                   "classical_steps":70,"classical_starts":[-3,0,3],
                   "scope":"Procedural sparse complex images and injected 2-coefficient phase errors; no measured-data generalization claim"})
    (args.output/"learning_results.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.output/"training_history.json").write_text(json.dumps(history,indent=2)+"\n")
    np.savez_compressed(args.output/"procedural_test_metrics.npz",**arrays,
                        true_coeff=test_coeff.numpy(),learned_coeff=pred.detach().numpy(),
                        classical_coeff=classical_coeff.detach().numpy())
    np.savez_compressed(args.output/"procedural_examples.npz",clean=test_clean[:8].numpy(),
                        corrupted=test_bad[:8].numpy(),entropy=classical[:8].detach().numpy(),
                        learned=learned[:8].numpy())
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
