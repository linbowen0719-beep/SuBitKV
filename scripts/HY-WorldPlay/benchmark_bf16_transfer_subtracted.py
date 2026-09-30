#!/usr/bin/env python3
"""Measured BF16 offload with explicitly labeled transfer-subtracted simulation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from quant_videogen.hy_kv_transfer_timer import adjusted_e2e

def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main(a):
    import pynvml as nv
    nv.nvmlInit()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    old=Path('/data/bowen/hy_q_e2e_no_kv_offload_20260924_run02/round_0/bf16/attempt_00')
    template=json.loads((old/'command.json').read_text());template.remove('--no-kv-offload')
    def replace(cmd,k,v):cmd[cmd.index(k)+1]=str(v)
    config=dict(frames=173,chunks=11,steps=4,seed=123,width=848,height=464,fps=24,method='BF16',
        component_offload=True,kv_offload=True,prefetch=False,untimed_smoke_frames=45,untimed_full_warmups=1,
        measured_repeats=3,gpus=[2,4,6],reference_command=str(old/'command.json'),
        runtime='Native cold-process E2E minus isolated, serialized KV H2D+D2H wall intervals. Model weight offload and preceding compute waits retained.',
        caveat='Simulation only, not measured resident BF16 or proof of resident speedup. Three fresh-process repeats run concurrently after full warmup; shared-host and synchronization effects remain.')
    write(root/'config.json',config)
    info=json.loads((old/'native/run_info.json').read_text())
    sources=[Path(__file__),ROOT/'quant_videogen/hy_kv_transfer_timer.py',ROOT/'experiments/HY-WorldPlay/hyvideo/generate.py',ROOT/'experiments/HY-WorldPlay/hyvideo/models/transformers/worldplay_1_5_transformer.py']
    info.update(config_path=str(root/'config.json'),config_hash=sha(root/'config.json'),timestamp=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={str(p):sha(p) for p in sources})
    write(root/'run_info.json',info)
    locks=[]
    for gpu in (2,4,6):
        f=open(f'/tmp/qvg_wan_e2e_gpu_{gpu}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    cache=root/'jit_cache';cache.mkdir()
    def run(label,gpu,frames):
        handle=nv.nvmlDeviceGetHandleByIndex(gpu)
        for retry in range(3):
            while nv.nvmlDeviceGetComputeRunningProcesses(handle) or nv.nvmlDeviceGetMemoryInfo(handle,version=nv.nvmlMemory_v2).used>512*1024**2 or nv.nvmlDeviceGetUtilizationRates(handle).gpu>5:time.sleep(5)
            task=root/label/f'attempt_{retry:02d}';task.mkdir(parents=True,exist_ok=False);out=task/'native'
            cmd=list(template)
            for k,v in [('--gpu',gpu),('--frames',frames),('--output-dir',out)]:replace(cmd,k,v)
            env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                QVG_HY_MEASURE_KV_TRANSFER='1',QVG_HY_CACHE_DECODE_PREFETCH='0',
                TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
            env.pop('QVG_WARM_BENCH_PATH',None)
            write(task/'command.json',cmd)
            write(task/'run_info.json',{**info,'gpu_index':gpu,'label':label,'frames':frames,'QVG_HY_MEASURE_KV_TRANSFER':1})
            print(f'START {label} GPU {gpu} frames {frames}',flush=True)
            with (task/'launcher.log').open('x') as log:completed=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {'status':'failed','returncode':completed.returncode}
            row={'label':label,'gpu':gpu,'path':str(out),'status':result['status']}
            if result['status']=='success':
                m=json.loads((out/'generation_metrics.json').read_text());events=m['kv_transfer_subtraction_events']
                assert m['kv_cache_offload'] and m['kv_cache_cpu_peak_mib']>0
                assert {e['direction'] for e in events}=={'h2d','d2h'}
                row.update(adjusted_e2e(result['end_to_end_seconds'],events))
                row.update(transfer_calls=len(events),nvml_peak_mib=result['nvml_peak_used_mib'],video=result['video'],
                    bytes_by_direction={d:sum(e['bytes'] for e in events if e['direction']==d) for d in ('h2d','d2h')})
            write(task/'result.json',row);print(json.dumps(row),flush=True)
            if row['status']!='contaminated':return row
        return row
    for label,frames in [('smoke',45),('warmup',173)]:
        row=run(label,2,frames)
        if row['status']!='success':write(root/'blocked.json',row);return
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(lambda pair:run(f'measured_{pair[0]}',pair[1],173),enumerate((2,4,6),1)))
    summary={'config':config,'samples':rows,'status':'complete' if all(r['status']=='success' for r in rows) else 'incomplete'}
    if summary['status']=='complete':
        for k in ('actual_e2e_seconds','kv_h2d_isolated_seconds','kv_d2h_isolated_seconds','simulated_resident_e2e_seconds'):
            v=[r[k] for r in rows];summary[k]={'mean':statistics.mean(v),'stdev':statistics.stdev(v),'min':min(v),'max':max(v)}
    write(root/'summary.json',summary);print('FINISHED '+str(root),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception as e:
        if a.output.is_dir():write(a.output/'driver_failure.json',{'error':repr(e),'timestamp':datetime.now(timezone.utc).isoformat()})
        raise
