from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch

from subitkv import hyworldplay8b_cache_codec as codec
from scripts.main.hyworldplay_i2v import SUBBIT_SCHEDULES


def profile_payload(horizon):
    prefix = SUBBIT_SCHEDULES['ours_rvq_avg0p2_alt'][3]
    rows = [list(row) for row in prefix]
    rows += [[0] * (n - 2) + [1, 1] for n in range(11, horizon + 1)]
    tensor = dict(capacity_bit_chunks=6, terminal_bit_chunks=2,
                  distortions={'8': .1, '4': .2, '3': .3, '2': .4, '1': .5},
                  age_weights=[1.] * 10, joint_schedule=rows)
    return dict(schema_version=codec.FIXED_TWO_PROFILE_SCHEMA, horizon=horizon,
                allowed_bits=list(codec.AVG2_SCHEDULE_BITS),
                tensor_profiles={'K': deepcopy(tensor), 'V': deepcopy(tensor)})


@pytest.mark.parametrize('horizon', [10, 11, 15, 31, 100])
def test_fixed_two_budget_and_frozen_prefix(horizon):
    p = codec.load_avg2_profile(profile_payload(horizon))
    for name in ('K', 'V'):
        for n in range(1, 11):
            assert p.schedule(name, n) == SUBBIT_SCHEDULES['ours_rvq_avg0p2_alt'][3][n-1]
        for n in range(10, horizon + 1):
            assert p.schedule(name, n) == (0,) * (n-2) + (1, 1)
        assert sum(p.schedule(name, horizon)) / horizon == 2 / horizon


def test_fixed_two_rejects_extra_bits_and_nonlatest_bits():
    for row in ([0]*8+[1,1,1], [0]*8+[1,1,0]):
        raw=profile_payload(11)
        raw['tensor_profiles']['K']['joint_schedule'][-1]=row
        with pytest.raises(ValueError, match='newest two'):
            codec.load_avg2_profile(raw)


def test_legacy_horizon_is_not_silently_extended():
    raw=profile_payload(11)
    raw['schema_version']=codec.TEMPORAL_PROFILE_SCHEMA
    with pytest.raises(ValueError, match='horizon'):
        codec.load_avg2_profile(raw)


def test_append_across_tenth_span_preserves_base_and_accounting(tmp_path, monkeypatch):
    path=tmp_path/'profile.json'
    path.write_text(json.dumps(profile_payload(15)))
    config=SimpleNamespace()
    settings=codec.CodecSettings(name='rvq_ours_avg0p2_alt', quant_type='unused',
        quant_config=config,frame_spatial_tokens=1,latent_frames_per_chunk=4,
        temporal_profile_path=str(path))
    shape=(2,16,4,128)
    bases=[object() for _ in range(10)]
    initial=[dict(bits=b,shape=shape,base=base) for b,base in zip([0]*8+[1,1],bases)]
    k=codec._segmented(initial,torch.bfloat16,{})
    v=codec._segmented([dict(s) for s in initial],torch.bfloat16,{})
    admissions=[]
    def transition(state,bits,**kwargs):
        assert bits <= state['bits']
        return dict(state,bits=bits)
    def compress(k,v,*args,span_idx,**kwargs):
        admissions.append(span_idx)
        return (dict(bits=config.temporal_adaptive_k_admission_target_bits,shape=shape,base=object()),
                dict(bits=config.temporal_adaptive_v_admission_target_bits,shape=shape,base=object()))
    monkeypatch.setattr(codec,'transition_state',transition)
    monkeypatch.setattr(codec,'compress_kv_cache',compress)
    monkeypatch.setattr(codec,'get_quantize_fn',lambda *args: None)
    monkeypatch.setattr(codec,'_advanced_state_bytes',lambda state: 10)
    monkeypatch.setattr(codec,'_advanced_shared_bytes',lambda state: 0)
    new=torch.zeros(shape,dtype=torch.bfloat16)
    for n in range(11,16):
        k,v=codec.append_encoded_vision_kv(k,v,new,new,layer_idx=0,settings=settings)
        for packed in (k,v):
            assert [s['cache']['bits'] for s in packed['segments']] == [0]*(n-2)+[1,1]
            assert all(packed['segments'][i]['cache']['base'] is bases[i] for i in range(10))
            assert packed['qvg_accounting']['terminal_k_bits'] == [0]*13+[1,1]
    assert admissions == list(range(10,15))
    audit=codec.validate_avg2_terminal_budget([k['qvg_accounting']],expected_layers=1)
    assert audit['nominal_residual_avg_bits'] == 2/15
