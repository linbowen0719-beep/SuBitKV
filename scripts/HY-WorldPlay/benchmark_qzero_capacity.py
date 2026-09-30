#!/usr/bin/env python3
"""Measure the published Q=0 schedule against a device-memory threshold.

Replay the preceding native HY workload, preserving its component offload and
compressed-vision GPU residency. Stop at the first above-budget coarse probe;
optionally refine to adjacent history lengths. No memory-slope extrapolation.
"""
from copy import deepcopy
import argparse
import csv
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from subitkv.hyworldplay8b_cache_codec import QZERO_LONG_PROFILE_SCHEMA, load_avg2_profile
from scripts.main.hyworldplay_i2v import SUBBIT_SCHEDULES


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value,f,indent=2)


def make_profile(source, horizon):
    if horizon < 10:
        raise ValueError('requires at least 10 history spans')
    result=deepcopy(source)
    prefix=[list(r) for r in SUBBIT_SCHEDULES['ours_rvq_avg0'][3]]
    for tensor in result['tensor_profiles'].values():
        tensor.update(capacity_bit_chunks=4,terminal_bit_chunks=0,
            joint_schedule=prefix+[[0]*n for n in range(11,horizon+1)])
    result.update(schema_version=QZERO_LONG_PROFILE_SCHEMA,horizon=horizon,
        extension=dict(policy='published_Q0_prefix_then_B0',calibrated_age_weights_extended=False))
    load_avg2_profile(result)
    return result


def replace(command, flag, value):
    command[command.index(flag)+1]=str(value)


def wait_idle():
    while True:
        usage=subprocess.check_output(['nvidia-smi','--id=7',
            '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        memory,util=map(int,usage.strip().split(','))
        pids=subprocess.check_output(['nvidia-smi','--id=7',
            '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if memory<=512 and util<=5 and not pids:
            return
        print('Waiting for exclusive GPU 7:',usage.strip(),pids,flush=True)
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--budget-mib',type=int,default=24576)
    parser.add_argument('--refine',action='store_true')
    args=parser.parse_args()
    lock=open('/tmp/hy_qzero_legacy_gpu_7.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=args.output_root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    ref=args.reference_root.resolve()
    source=json.loads((ref/'profile.json').read_text())
    template=json.loads((ref/'capacity_173_command.json').read_text())
    info=json.loads((ref/'run_info.json').read_text())
    config=dict(reference_root=str(ref),budget_mib=args.budget_mib,refine=args.refine,
        gpu=7,policy='published_Q0_prefix_then_B0',coarse_step_history_spans=5,
        model_component_offload=True,compressed_vision_kv_gpu_resident=True,
        text_kv_offload=True,full_history=True,terminal_cache_audit=False,
        nominal_residual_bits='terminal Q=0; base/index/metadata retained',
        resolution=[848,464],fps=24,seed=123,prompts=1)
    write(root/'config.json',config)
    diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=REPO)
    info.update(config_path=str(root/'config.json'),config_hash=sha(root/'config.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        git_diff_hash=hashlib.sha256(diff).hexdigest(),hostname=socket.gethostname(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_hashes={str(p.relative_to(REPO)):sha(p) for p in (
            Path(__file__).resolve(),REPO/'subitkv/hyworldplay8b_cache_codec.py',
            REPO/'tests/test_hyworldplay_qzero_capacity.py')})
    for stale in ('scope','video_frames','launcher_hash'):
        info.pop(stale,None)
    write(root/'run_info.json',info)
    (root/'source_diff.patch').write_bytes(diff)
    observations=[]

    def probe(horizon):
        for attempt in range(3):
            wait_idle()
            profile_path=root/f'profile_h{horizon}_a{attempt}.json'
            write(profile_path,make_profile(source,horizon))
            out=root/f'h{horizon}_a{attempt}'
            command=list(template)
            replace(command,'--method','ours_rvq_avg0')
            replace(command,'--output-dir',out)
            replace(command,'--profile-path',profile_path)
            replace(command,'--frames',16*(horizon+1)-3)
            write(root/f'command_h{horizon}_a{attempt}.json',command)
            print(f'Starting {horizon} history spans / {16*(horizon+1)-3} frames',flush=True)
            env=dict(os.environ,PYTHONPATH=str(REPO),
                HYWORLDPLAY_RVQ_RELATIVE_STOP_FAST_APPROXIMATE='0',
                HYWORLDPLAY_RVQ_STAGED_STOP_CHECK_STRIDE='1')
            with (root/f'launcher_h{horizon}_a{attempt}.log').open('x') as log:
                subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            result=json.loads((out/'result.json').read_text())
            if result['status']=='contaminated':
                print('Contaminated: retaining and retrying',flush=True)
                continue
            if result['status']!='success':
                raise RuntimeError(f'Probe failed; inspect {out}. Not reported as a measured length.')
            accounting=result['rvq_temporal_physical_accounting']
            snapshot=accounting['terminal_snapshot']
            assert snapshot['history_spans']==horizon and snapshot['layers']==54
            metrics=json.loads((out/'generation_metrics.json').read_text())
            events=metrics['kv_codec_encode_events']
            terminal=[e for e in events if e.get('history_spans')==horizon]
            assert len({e['layer'] for e in terminal})==54
            for event in terminal:
                audit=event['post_truncation_accounting']
                assert audit['profile_horizon']==horizon
                assert audit['old_spans_reencoded'] is False
                for name in ('selected_k_bits','selected_v_bits'):
                    assert audit[name]==[0]*horizon
            assert all(e.get('d2h_seconds',0)==0 for e in events)
            assert abs(snapshot['nominal_residual_avg_bits'])<1e-12
            row=dict(history_spans=horizon,frames=result['video']['frames'],
                video_seconds=result['video']['frames']/24,
                nominal_residual_bits=0,peak_gpu_mib=result['nvml_peak_used_mib'],
                compressed_vision_kv_peak_mib=result['kv_cache_gpu_compressed_peak_mib'],
                cpu_kv_peak_mib=metrics['kv_cache_cpu_peak_mib'],result_path=str(out/'result.json'))
            row['above_budget']=row['peak_gpu_mib']>args.budget_mib
            observations.append(row)
            write(root/f'observation_h{horizon}.json',row)
            print(json.dumps(row),flush=True)
            return row
        raise RuntimeError('Three contaminated attempts; no valid capacity observation')

    first=probe(10)
    if first['above_budget']:
        raise RuntimeError('11-chunk Q0 exceeds threshold; smaller-length search required, no maximum established')
    low=10
    high=15
    while not probe(high)['above_budget']:
        low=high
        high+=5
    if args.refine:
        while high-low>1:
            middle=(low+high)//2
            if probe(middle)['above_budget']:
                high=middle
            else:
                low=middle
        if low==10:
            if probe(10)['above_budget']:
                raise RuntimeError('Reference 10-span case exceeded budget on retest; bracket invalid')
    summary=dict(budget_mib=args.budget_mib,below_or_equal_history_spans=low,
        above_history_spans=high,refined=args.refine,observations=observations,
        reference_10_span_result=str(ref/'capacity_173/result.json'),
        caveat='single-prompt peak-memory observations; not an OOM/device limit or quality guarantee')
    write(root/'summary.json',summary)
    with (root/'observations.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(observations[0]))
        writer.writeheader();writer.writerows(observations)
    print('Completed threshold search:',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()

