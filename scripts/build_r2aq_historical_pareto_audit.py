#!/usr/bin/env python3
"""Create a protocol-aware historical R²AQ comparison table (create-only)."""
from __future__ import annotations

import argparse, hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--history', type=Path, required=True)
    ap.add_argument('--exact', type=Path, required=True)
    ap.add_argument('--integrated', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    a=ap.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    h=pd.read_csv(a.history)
    keep=['method','label_zh','method_family','prefix_bits','K_MSE','V_MSE','exact_KV_output_MSE_2half_mean','pixel_MSE','PSNR','SSIM','LPIPS_normalized_v2','vbench_mean_10d','joint_payload_MiB_primary','joint_ratio_primary','quant_time_s_per_video','video_status','video_prompts']
    out=h[[c for c in keep if c in h]].copy()
    out['comparison_scope']='historical_10prompt_video; historical_strict_v4_exact'
    out['r2aq_comparability']='reference_only: capture/history accounting differs from R2AQ exact replay'
    ex=pd.read_csv(a.exact/'tables/prompt_macro_metrics.csv')
    r=[]
    for bits in (2,4):
      for method in ('rvq_plus_quantvideo','r2aq_coherent_tail_g128'):
        q=ex[(ex.method==method)&(ex.bits==bits)&(ex.intervention=='K_and_V')&(ex.attention_half=='camera_prope')]
        r.append({'method':method+f'_B{bits}_revalidation','label_zh':('RVQ+QuantVideo' if method=='rvq_plus_quantvideo' else 'R²AQ coherent-tail G128')+f' B{bits}','method_family':'r2aq_strict_revalidation','prefix_bits':bits,'K_MSE':q.historical_K_mse.mean(),'V_MSE':q.historical_V_mse.mean(),'exact_KV_output_MSE_2half_mean':q.attention_output_squared_error_mean.mean(),'joint_payload_MiB_primary':(q.total_payload_K_MiB+q.total_payload_V_MiB).mean(),'comparison_scope':'fresh R2AQ exact replay: test prompts 0,4,9; layers 0,10,20,29; camera PRoPE; payload exact','r2aq_comparability':'direct only R2AQ vs QV rows in this same fresh replay'})
    rr=pd.DataFrame(r)
    all=pd.concat([out,rr],ignore_index=True,sort=False)
    all.to_csv(a.output_dir/'all_methods_large_table.csv',index=False)
    sim=pd.read_csv(a.integrated/'three_prompt_similarity.csv')
    gate=pd.read_csv(a.integrated/'gate_status.csv')
    video=sim.assign(method='R²AQ G128 B4 vs RVQ+QV4', scope='3-prompt development only; not formal 10 prompt')
    video.to_csv(a.output_dir/'r2aq_closed_loop_development.csv',index=False)
    decision={'global_quality_memory_pareto':'NOT_ESTABLISHED','reason':'R²AQ B4 3-prompt closed-loop gate failed (macro composite 0.345%, CI crosses zero, 2/3 wins); no formal 10-prompt similarity/VBench is authorized. Historical methods use a different exact replay/capture accounting, so cross-protocol exact Pareto is reference-only.','speed_optimization_authorized':False,'exact_revalidation':'PASS: strict equal payload and all 12 units complete; R²AQ improves QV in same replay, but proxy does not override closed-loop failure.'}
    (a.output_dir/'pareto_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    md=['# R²AQ 历史方法对比与 Pareto 重验证','','## 结论','','R²AQ **尚不能被认定为全历史方法中质量—内存 Pareto 最优**，因此不进入速度优化阶段。新 exact replay 证明它在同 payload 下显著降低 attention/KV proxy 误差；但 B4 的独立三 prompt 闭环视频开发 gate 为失败，不能用 proxy 覆盖该反证，也未授权 VBench。','','## 新鲜严格重验（同一 replay 内可直接比较）','',rr.to_markdown(index=False),'','所有 B2/B4 R²AQ 与 QV 行的 K、V payload 均逐字节相同；prompt=0/4/9、layer=0/10/20/29、24 heads、standard/camera 机制均已运行。上表展示 camera PRoPE K+V 聚合，`exact_KV_output_MSE_2half_mean` 此处是 attention-output mean。','','## 闭环反证','','| 方法 | macro composite 改善 | bootstrap CI | 胜率 | 判断 |','|---|---:|---|---:|---|','| R²AQ G128 B4 vs RVQ+QV4 | 0.345% | [-18.253%, 3.060%] | 2/3 | FAIL |','','这不是正式 10-prompt benchmark，且其失败正是停止扩大视频/VBench评估的预注册门槛。','','## 大表与可比性','','完整历史大表写入 `all_methods_large_table.csv`。其中历史 10-prompt video 数值彼此可比较；R²AQ 新 exact 行仅能与其同一 fresh replay 的 QV 行直接比较。不同 capture history、exact 聚合或 video protocol 的行被明确标为 reference-only，不能据此宣称全局 Pareto。','','## 速度决策','','既有 production full-restore median decode 相对 QV：B2 +48.53%，B4 +54.02%，均未满足 ≤10% 门槛。由于质量 gate 未通过，本轮不对 R²AQ 进行性能优化，避免优化一个尚未验证闭环优势的方向。']
    (a.output_dir/'README_zh.md').write_text('\n'.join(md)+'\n')
    info={'timestamp':datetime.now(timezone.utc).isoformat(),'git_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'seed':20260724,'prompt_list':['0','4','9'],'sources':{str(x):sha(x) for x in (a.history,a.exact/'tables/prompt_macro_metrics.csv',a.integrated/'three_prompt_similarity.csv',a.integrated/'gate_status.csv')}}
    (a.output_dir/'run_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
