"""Opt-in serialized BF16 KV transfer timing for an explicitly simulated baseline."""
import time
import torch


def move_and_measure(layer, device, state):
    if state.kv_cache_codec != 'none' or not state.kv_cache_offload:
        raise ValueError('transfer subtraction requires BF16 CPU-offloaded KV')
    moving=[v for v in layer.values() if isinstance(v,torch.Tensor)
            and v.device.type != device.type and {v.device.type,device.type}=={'cpu','cuda'}]
    if not moving:
        return
    # Finish preceding attention/MLP work BEFORE starting the deductible interval.
    ready_start=time.perf_counter()
    torch.cuda.synchronize()
    started=time.perf_counter()
    size=sum(v.numel()*v.element_size() for v in moving)
    for name,value in layer.items():
        if isinstance(value,torch.Tensor) and value.device != device:
            layer[name]=value.to(device)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    direction='d2h' if device.type=='cpu' else 'h2d'
    event={'direction':direction,'bytes':size,'seconds':elapsed,
           'preceding_compute_wait_seconds_not_subtracted':started-ready_start}
    if not hasattr(state,'kv_transfer_subtraction_events'):
        state.kv_transfer_subtraction_events=[]
    state.kv_transfer_subtraction_events.append(event)
    state.add_part_seconds('kv_transfer_isolated_'+direction,elapsed)


def adjusted_e2e(raw,events):
    totals={d:sum(e['seconds'] for e in events if e['direction']==d) for d in ('h2d','d2h')}
    if not events or any(e['seconds']<0 for e in events):
        raise ValueError('missing or invalid measured transfers')
    adjusted=raw-sum(totals.values())
    if adjusted<=0:raise ValueError('transfer subtraction exceeds E2E')
    return {'actual_e2e_seconds':raw,'kv_h2d_isolated_seconds':totals['h2d'],
            'kv_d2h_isolated_seconds':totals['d2h'],'simulated_resident_e2e_seconds':adjusted,
            'preceding_compute_wait_not_subtracted_seconds':sum(e['preceding_compute_wait_seconds_not_subtracted'] for e in events),
            'caveat':'Measured offload E2E minus serialized KV move intervals; not measured GPU-resident execution. Allocation, residency, synchronization and host effects can remain.'}
