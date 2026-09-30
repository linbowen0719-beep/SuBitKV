import importlib.util
from pathlib import Path
import pytest
from subitkv.hyworldplay8b_cache_codec import load_avg2_profile
from test_hyworldplay_fixed_two_capacity import profile_payload

def driver():
    p=Path(__file__).resolve().parents[1]/'scripts/HY-WorldPlay/benchmark_qzero_capacity.py'
    s=importlib.util.spec_from_file_location('qzero_driver',p)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

@pytest.mark.parametrize('n',[10,11,15,64])
def test_qzero_frozen_prefix_and_long_extension(n):
    raw=driver().make_profile(profile_payload(10),n)
    p=load_avg2_profile(raw)
    for stream in ('K','V'):
        assert p.schedule(stream,1)==(4,)
        assert p.schedule(stream,n)==(0,)*n
        for t in range(2,n+1):
            assert all(a>=b for a,b in zip(p.schedule(stream,t-1),p.schedule(stream,t)))
    raw['tensor_profiles']['K']['joint_schedule'][-1][-1]=1
    with pytest.raises(ValueError):
        load_avg2_profile(raw)
