from types import SimpleNamespace
import pytest
from quant_videogen import hy_kv_transfer_timer as timer

@pytest.mark.parametrize('source,target,direction',[('cuda','cpu','d2h'),('cpu','cuda','h2d')])
def test_compute_wait_is_not_subtracted(monkeypatch,source,target,direction):
    clock=[0.]; waits=iter((11.,3.))
    class Tensor:
        def __init__(self,kind):self.device=SimpleNamespace(type=kind)
        def numel(self):return 10
        def element_size(self):return 2
        def to(self,device):clock[0]+=2;return Tensor(device.type)
    monkeypatch.setattr(timer,'torch',SimpleNamespace(Tensor=Tensor,cuda=SimpleNamespace(synchronize=lambda:clock.__setitem__(0,clock[0]+next(waits)))))
    monkeypatch.setattr(timer.time,'perf_counter',lambda:clock[0])
    state=SimpleNamespace(kv_cache_codec='none',kv_cache_offload=True,add_part_seconds=lambda *args:None)
    layer={'k':Tensor(source)}
    timer.move_and_measure(layer,SimpleNamespace(type=target),state)
    event=state.kv_transfer_subtraction_events[0]
    assert event['direction']==direction and event['bytes']==20
    assert event['seconds']==5 and event['preceding_compute_wait_seconds_not_subtracted']==11
    assert timer.adjusted_e2e(100,[event])['simulated_resident_e2e_seconds']==95

def test_adjustment_rejects_invalid_accounting():
    with pytest.raises(ValueError):timer.adjusted_e2e(10,[])
    with pytest.raises(ValueError):timer.adjusted_e2e(1,[dict(direction='h2d',seconds=2,preceding_compute_wait_seconds_not_subtracted=0)])
