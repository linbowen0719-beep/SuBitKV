import sys, math
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np
dev=torch.device("cuda"); dt=torch.bfloat16; D=128; L,C=4,256
CB=Path("/data/bowen/rvq_diag/codebooks")
def rope1d(x,pos0,base=10000.0):
    S,Dd=x.shape; half=Dd//2
    fr=1.0/(base**(torch.arange(half,device=x.device).float()/half))
    ang=(torch.arange(S,device=x.device).float()+pos0)[:,None]*fr[None,:]
    cos,sin=torch.cos(ang),torch.sin(ang); x1,x2=x[:,:half],x[:,half:]
    return torch.cat([x1*cos-x2*sin,x2*cos+x1*sin],-1)
def enc_l0(K,cb0):
    sc=torch.sqrt((K.float()**2).mean(-1,keepdim=True)).clamp_min(1e-6); r=K.float()/sc
    d2=(r**2).sum(-1,keepdim=True)+(cb0**2).sum(-1)[None,:]-2*(r@cb0.T); return d2.argmin(-1)
def relerr(o,ref): return ((o-ref).norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-6)).mean().item()
def run(LAYER,NQ=64,P=32,fracs=(0.05,0.10,0.20)):
    d=torch.load(f"/data/bowen/rvq_diag/kv_dump/test/0/layer_{LAYER}.pt",map_location="cpu",weights_only=False)
    Qd,Kd,Vd=d["Q"],d["K"],d["V"]; T,H,N,_=Qd.shape; tq=T-1; t0=max(0,tq-20)
    cb=torch.zeros(H,L,C,D)
    for h in range(H): cb[h]=torch.load(CB/f"layer_{LAYER}_head_{h}.pt",map_location="cpu",weights_only=False)["codebooks"]
    cb=cb.to(dev); sm=1/math.sqrt(D); qpos0=(tq-t0)*N; ar=torch.arange(P,device=dev)
    eff=[]; rs={f:[] for f in fracs}; rh={f:[] for f in fracs}; rg=[]; mm={f:[] for f in fracs}
    for h in range(H):
        Kpre=Kd[t0:tq+1,h].reshape(-1,D).to(dev).float(); Kr=rope1d(Kpre,0); S=Kr.shape[0]; nP=S//P; S=nP*P
        Kr=Kr[:S]; Kpre=Kpre[:S]; Vp=Vd[t0:tq+1,h].reshape(-1,D).to(dev).float()[:S]
        Qr=rope1d(Qd[tq,h,:NQ].to(dev).float(),qpos0)
        p=torch.softmax((Qr@Kr.T)*sm,-1); eff.append((-(p*p.clamp_min(1e-20).log()).sum(-1)).exp().mean().item())
        ref=F.scaled_dot_product_attention(Qr[None,None].to(dt),Kr[None,None].to(dt),Vp[None,None].to(dt)).float()[0,0]
        pagemean=Kr.view(nP,P,D).mean(1); kid0=enc_l0(Kpre,cb[h,0])
        oh=F.one_hot(kid0,C).float(); gKs=torch.einsum("sc,sd->cd",oh,Kr); gVs=torch.einsum("sc,sd->cd",oh,Vp); gc=oh.sum(0)
        # pure grouped (all tokens by group mean) baseline:
        gKm0=(gKs/gc.clamp_min(1)[:,None]).to(dt); gVm0=(gVs/gc.clamp_min(1)[:,None]).to(dt)
        bias0=torch.where(gc>0.5,gc.log(),torch.full_like(gc,-1e9))[None,None,None,:].to(dt)
        og=F.scaled_dot_product_attention(Qr[None,None].to(dt),gKm0[None,None],gVm0[None,None],attn_mask=bias0).float()[0,0]
        rg.append(relerr(og,ref))
        q=Qr.to(dt)
        for frac in fracs:
            G=max(1,int(frac*nP)); sel=((q@pagemean.to(dt).T).float()).amax(0).topk(G).indices
            tok=(sel[:,None]*P+ar[None,:]).reshape(-1); mm[frac].append(p[:,tok].sum(-1).mean().item())
            Ks=Kr[tok].to(dt); Vs=Vp[tok].to(dt)
            osp=F.scaled_dot_product_attention(q[None,None],Ks[None,None],Vs[None,None]).float()[0,0]
            rs[frac].append(relerr(osp,ref))
            selc=kid0[tok]; cnt=gc.clone().scatter_add_(0,selc,-torch.ones_like(selc,dtype=torch.float))
            gK=gKs.clone().scatter_add_(0,selc[:,None].expand(-1,D),-Ks.float()); gV=gVs.clone().scatter_add_(0,selc[:,None].expand(-1,D),-Vs.float())
            nz=cnt.clamp_min(1.0); gKm=(gK/nz[:,None]).to(dt); gVm=(gV/nz[:,None]).to(dt)
            lc=torch.where(cnt>0.5,cnt.log(),torch.full_like(cnt,-1e9))
            Ka=torch.cat([Ks,gKm],0)[None,None]; Va=torch.cat([Vs,gVm],0)[None,None]
            bias=torch.cat([torch.zeros(G*P,device=dev),lc],0)[None,None,None,:].to(dt)
            ohy=F.scaled_dot_product_attention(q[None,None],Ka,Va,attn_mask=bias).float()[0,0]
            rh[frac].append(relerr(ohy,ref))
    print(f"L{LAYER} S={S} effN={np.mean(eff):.0f}({100*np.mean(eff)/S:.0f}%) grouped_rel={np.mean(rg):.3f}",flush=True)
    for f in fracs:
        print(f"   frac={f} mass={np.mean(mm[f]):.3f}  sparse_rel={np.mean(rs[f]):.3f}  hybrid_rel={np.mean(rh[f]):.3f}",flush=True)
for LAYER in [10,15,20,25,29]: run(LAYER)
