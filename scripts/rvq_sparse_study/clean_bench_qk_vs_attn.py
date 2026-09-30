import sys, math, statistics, torch, torch.nn.functional as F
sys.path.insert(0,"."); sys.path.insert(0,"/tmp/claude-1007/-home-bowen-Quant-VideoGen/d9dfbba0-8d43-4ce5-8eb0-f2e7cdbbbd93/scratchpad")
from subitkv.kernels.triton.rvq import rvq_decode_bhsd_triton
from subitkv.kernels.triton.rvq_qk import rvq_qk_decode_matmul_b1_triton
from splitk_attn import splitk_attn
dev=torch.device("cuda"); dt=torch.bfloat16; H,Q,D=24,64,128; L,C=4,256; sm=1/math.sqrt(D)
def med(fn,reps=7,it=40):
    for _ in range(20): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(it): fn()
        e.record();e.synchronize(); ts.append(s.elapsed_time(e)/it)
    return statistics.median(ts)
print(f"{'K':>7} {'dense_sdpa':>11} {'splitk_attn':>12} {'spd':>5} {'err':>7} | {'dense_qk':>9} {'fused_qk':>9} {'spd':>5}")
for K in [8192,16384,24576,32768,49152]:
    g=torch.Generator(device=dev).manual_seed(0)
    query=torch.randn(1,H,Q,D,device=dev,dtype=dt,generator=g)
    kcb=torch.randn(L,H,C,D,device=dev,dtype=dt,generator=g).contiguous(); vcb=torch.randn(L,H,C,D,device=dev,dtype=dt,generator=g).contiguous()
    kidx=torch.randint(0,C,(L,1,H,K),device=dev,dtype=torch.uint8,generator=g).contiguous(); vidx=torch.randint(0,C,(L,1,H,K),device=dev,dtype=torch.uint8,generator=g).contiguous()
    kscale=(0.25+torch.rand(1,H,K,1,device=dev,generator=g)).to(dt).contiguous(); vscale=(0.25+torch.rand(1,H,K,1,device=dev,generator=g)).to(dt).contiguous()
    Kf=rvq_decode_bhsd_triton(kidx,kcb,kscale,dt); Vf=rvq_decode_bhsd_triton(vidx,vcb,vscale,dt)
    ref=F.scaled_dot_product_attention(query,Kf,Vf)
    spl=4096 if K>=16384 else 2048
    out=splitk_attn(query,kidx,kcb,kscale,vidx,vcb,vscale,softmax_scale=sm,split=spl,block_k=64,num_warps=8)
    err=(out.float()-ref.float()).abs().max().item()
    d_sdpa=med(lambda:F.scaled_dot_product_attention(query,Kf,Vf))
    sk=med(lambda:splitk_attn(query,kidx,kcb,kscale,vidx,vcb,vscale,softmax_scale=sm,split=spl,block_k=64,num_warps=8))
    d_qk=med(lambda:torch.matmul(query,Kf.transpose(-1,-2)))
    f_qk=med(lambda:rvq_qk_decode_matmul_b1_triton(query,kidx,kcb,kscale,block_q=64,block_k=128,num_warps=8,num_stages=4))
    print(f"{K:7d} {d_sdpa:11.4f} {sk:12.4f} {d_sdpa/sk:5.2f} {err:7.4f} | {d_qk:9.4f} {f_qk:9.4f} {d_qk/f_qk:5.2f}")
