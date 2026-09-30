import sys, math, torch, torch.nn.functional as F
sys.path.insert(0,"/home/bowen/Quant-VideoGen"); sys.path.insert(0,".")
from subitkv.kernels.triton.rvq import rvq_decode_bhsd_triton
from rvq_sparse import build_cache, sparse_attn
dev=torch.device("cuda"); dt=torch.bfloat16
H,NQ,D,L,C,P=24,64,128,4,256,32
def tf(fn,w=20,it=100):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record();e.synchronize(); return s.elapsed_time(e)/it
for S in [16384,32768,65536,131072]:
    nP=S//P; G=int(0.10*nP); sm=1/math.sqrt(D)
    g=torch.Generator(device=dev).manual_seed(0)
    kidx=torch.randint(0,C,(L,1,H,S),device=dev,dtype=torch.uint8,generator=g).contiguous()
    vidx=torch.randint(0,C,(L,1,H,S),device=dev,dtype=torch.uint8,generator=g).contiguous()
    kcb=torch.randn(L,H,C,D,device=dev,dtype=dt,generator=g).contiguous(); vcb=torch.randn(L,H,C,D,device=dev,dtype=dt,generator=g).contiguous()
    kscale=(0.25+torch.rand(1,H,S,1,device=dev,generator=g)).to(dt).contiguous(); vscale=(0.25+torch.rand(1,H,S,1,device=dev,generator=g)).to(dt).contiguous()
    query=torch.randn(1,H,NQ,D,device=dev,dtype=dt,generator=g)
    cache=build_cache(kidx,vidx,kcb,kscale,vcb,vscale,P,dt)
    Kf,Vf=cache["Kf"],cache["Vf"]
    td=tf(lambda:F.scaled_dot_product_attention(query,Kf,Vf))
    ts=tf(lambda:sparse_attn(query,kidx,vidx,kcb,kscale,vcb,vscale,cache,P=P,G=G,use_tail=False))
    th=tf(lambda:sparse_attn(query,kidx,vidx,kcb,kscale,vcb,vscale,cache,P=P,G=G,use_tail=True))
    print(f"S={S:7d} G={G} frac={G/nP:.3f}: dense={td:.4f} sparse={ts:.4f}({td/ts:.2f}x) hybrid={th:.4f}({td/th:.2f}x)",flush=True)
