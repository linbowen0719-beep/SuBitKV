#!/usr/bin/env python3
"""Queue HY all-GPU KV latency measurement after the registered Wan experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.main.hyworldplay_i2v import SUBBIT_SCHEDULES
from subitkv.hyworldplay8b_cache_codec import load_avg2_profile
DATA=Path('/data/bowen/hyworldplay_moviegen_rvq_avg2_20260824T193800Z')
PYTHON='/home/bowen/anaconda3/envs/longcat-official/bin/python'
METHODS=('bf16','ours_rvq_avg0','ours_rvq_avg0p2_alt','ours_rvq_avg0p5','ours_rvq_avg1','ours_rvq_avg2','ours_rvq_avg4')

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def replace(cmd,flag,value):cmd[cmd.index(flag)+1]=str(value)

def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    template=json.loads(Path('/data/bowen/hy_fixed_two_capacity_20260911_attempt01/command_h11_a0.json').read_text())
    config=dict(prerequisite=str(a.after_wan),methods=METHODS,gpus=list(range(7)),frames=173,chunks=11,seed=123,
        fps=24,width=848,height=464,steps=4,component_offload=True,kv_cpu_offload=False,
        append_only=True,full_history=True,untimed_full_warmups=1,repeats=3,
        e2e='native independent generation process start to finalized MP4 mtime; warm filesystem/JIT caches',
        fast_encoder='monolithic near-tie approximate',decode='preallocated segmented restore',
        calibration='existing train4 artifacts; no retraining or test-prompt parameter selection',
        input='original MovieGen test prompt 0 / dataset row 223 and frozen conditioning image',
        failures='retain failed/OOM runs; never enable KV offload as fallback; no mean without three successes',
        prior_bf16_oom=str(a.prior_bf16_oom) if a.prior_bf16_oom else None)
    write(root/'config.json',config)
    source_files=[Path(__file__).resolve(),ROOT/'scripts/HY-WorldPlay/run_moviegen_i2v_job.py',ROOT/'subitkv/hyworldplay8b_cache_codec.py',ROOT/'experiments/HY-WorldPlay/hyvideo/generate.py',ROOT/'experiments/HY-WorldPlay/hyvideo/models/transformers/worldplay_1_5_transformer.py']
    inherited=json.loads(Path('/data/bowen/hy_fixed_two_capacity_20260911_attempt01/run_info.json').read_text())
    info={**inherited,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'hostname':socket.gethostname(),'timestamp':datetime.now(timezone.utc).isoformat(),'seed':123,
        'config_path':str(root/'config.json'),'config_hash':sha(root/'config.json'),
        'source_hashes':{str(p):sha(p) for p in source_files},
        'dirty_state':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)}
    write(root/'run_info.json',info)
    base_path=DATA/'advanced_avg2_profile_b16_fix_20260825/temporal_avg2_profile.json'
    base=json.loads(base_path.read_text());profiles=root/'profiles';profiles.mkdir()
    for method in METHODS[1:-1]:
        if method=='ours_rvq_avg1':
            raw=json.loads(Path('/data/bowen/hyworldplay_avg1_perfdiag_20260826T134643Z/profile_avg1/temporal_avg2_profile.json').read_text())
        else:raw=deepcopy(base)
        if method in SUBBIT_SCHEDULES:
            bits,capacity,terminal,schedule=SUBBIT_SCHEDULES[method]
            raw['allowed_bits']=[16,8,4,3,2,1,0]
            for t in raw['tensor_profiles'].values():t.update(capacity_bit_chunks=capacity,terminal_bit_chunks=terminal,joint_schedule=schedule)
            raw['selection_provenance']={'policy':'published frozen subbit schedule','test_prompts_read':False}
        load_avg2_profile(raw);write(profiles/f'{method}.json',raw)
    calibration=Path(json.loads((base_path.parent/'config.json').read_text())['calibration'])
    params=DATA/'advanced_avg2_params'
    build=[PYTHON,str(ROOT/'scripts/HY-WorldPlay/build_moviegen_avg2_profile.py'),'--calibration',str(calibration),
        '--output-dir',str(profiles/'avg4'),'--codebook-root',str(DATA),'--routing',str(params/'routing.json'),
        '--tree',str(params/'tree_params.pt'),'--v-params',str(params/'v_progressive_params.pt'),'--train-prompts','4','--capacity-bit-chunks','40']
    with (root/'profile_build.log').open('x') as log:subprocess.run(build,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(profiles/'ours_rvq_avg4.json',json.loads((profiles/'avg4/temporal_avg4_profile.json').read_text()))
    write(root/'profile_hashes.json',{p.name:sha(p) for p in profiles.glob('*.json')})
    write(root/'queued.json',{'status':'waiting_for_wan','after':str(a.after_wan),'timestamp':datetime.now(timezone.utc).isoformat()})
    print('QUEUED after '+str(a.after_wan),flush=True)
    while not (a.after_wan/'summary.json').exists():
        if any((a.after_wan/p).exists() for p in ('blocked.json','driver_failure.json')):
            write(root/'blocked.json',{'reason':'Wan prerequisite failed; HY not launched'});return
        time.sleep(15)
    locks=[]
    for gpu in range(7):
        f=open(f'/tmp/qvg_wan_e2e_gpu_{gpu}.lock','a')
        while True:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(5)
        locks.append(f)
    import pynvml as nv
    nv.nvmlInit()
    def run(round_index,index,frames=173):
        method=METHODS[index];gpu=(index+2*round_index)%7;handle=nv.nvmlDeviceGetHandleByIndex(gpu)
        for retry in range(3):
            while nv.nvmlDeviceGetComputeRunningProcesses(handle) or nv.nvmlDeviceGetMemoryInfo(handle,version=nv.nvmlMemory_v2).used>512*1024**2 or nv.nvmlDeviceGetUtilizationRates(handle).gpu>5:time.sleep(5)
            task=root/f'round_{round_index}'/method/f'attempt_{retry:02d}';task.mkdir(parents=True,exist_ok=False)
            out=task/'native';cmd=list(template)
            for flag,value in [('--method',method),('--gpu',gpu),('--frames',frames),('--output-dir',out),('--profile-path',profiles/f'{method}.json')]:replace(cmd,flag,value)
            cmd.append('--no-kv-offload')
            if method=='bf16':cmd.remove('--gpu-resident-compressed-kv')
            write(task/'command.json',cmd)
            cache=root/'jit_cache'/method;cache.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                HYWORLDPLAY_RVQ_RELATIVE_STOP_FAST_APPROXIMATE='1',HYWORLDPLAY_RVQ_RELATIVE_STOP_MONOLITHIC_EXACT='0',
                HYWORLDPLAY_RVQ_MONOLITHIC_EXACT_AUDIT='0',SUBITKV_HY_CACHE_DECODE_PREFETCH='0',
                TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
            print(f'START round={round_index} method={method} GPU={gpu}',flush=True)
            start=time.perf_counter()
            with (task/'launcher.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if (out/'result.json').exists():row=json.loads((out/'result.json').read_text())
            else:row={'status':'failed','error':'native result missing','returncode':r.returncode}
            row.update(method=method,round=round_index,gpu=gpu,path=str(out),launcher_wall_seconds=time.perf_counter()-start)
            if row['status']!='success':
                logs=(task/'launcher.log').read_text(errors='replace')
                if (out/'generation.log').exists():logs+=(out/'generation.log').read_text(errors='replace')
                if 'out of memory' in logs.lower():row['status']='cuda_oom'
            write(task/'result.json',row)
            print(f'DONE {method} round={round_index}: {row["status"]}',flush=True)
            if row['status']!='contaminated':return row
        return row
    # Real text-cache initialization, two history admissions and VAE before full runs.
    smoke=run(-1,1,45)
    write(root/'smoke_result.json',smoke)
    if smoke['status']!='success':
        write(root/'blocked.json',{'reason':'GPU-resident 3-chunk smoke failed','path':smoke['path']});return
    rows=[];active=list(range(7))
    if a.prior_bf16_oom:
        prior=json.loads(a.prior_bf16_oom.read_text())
        assert prior['status']=='cuda_oom' and prior['method']=='bf16'
        prior['reused_failure_source']=str(a.prior_bf16_oom)
        prior['reused_failure_hash']=sha(a.prior_bf16_oom)
        rows.append(prior);active=list(range(1,7))
    for r in range(4):
        with ThreadPoolExecutor(max_workers=7) as pool:batch=list(pool.map(lambda i:run(r,i),active))
        rows.extend(batch);write(root/f'round_{r}_results.json',batch)
        active=[i for i in active if any(v['method']==METHODS[i] and v['status']=='success' for v in batch)]
    summary=[]
    for method in METHODS:
        selected=[v for v in rows if v['method']==method and v['round']>0 and v['status']=='success']
        row={'method':method,'success_samples':len(selected)}
        if len(selected)==3:
            values=[v['end_to_end_seconds'] for v in selected]
            row.update(status='complete',e2e_mean_seconds=statistics.mean(values),e2e_stdev_seconds=statistics.stdev(values),e2e_samples_seconds=values,nvml_peak_mib=max(v['nvml_peak_used_mib'] for v in selected))
        else:row.update(status='incomplete',failures=[{'round':v['round'],'status':v['status'],'path':v['path']} for v in rows if v['method']==method and v['status']!='success'])
        summary.append(row)
    write(root/'summary.json',{'config':config,'results':summary,'observations':rows})
    print('COMPLETE '+str(root),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--after-wan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prior-bf16-oom',type=Path);a=p.parse_args()
    try:main(a)
    except Exception as exc:
        if a.output.is_dir():write(a.output/'driver_failure.json',{'error':repr(exc),'timestamp':datetime.now(timezone.utc).isoformat()})
        raise
