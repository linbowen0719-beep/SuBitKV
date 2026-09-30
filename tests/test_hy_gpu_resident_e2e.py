import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

def test_actual_entrypoint_accepts_compression_without_cpu_offload():
    path=Path(__file__).resolve().parents[1]/'experiments/HY-WorldPlay/hyvideo/generate.py'
    tree=ast.parse(path.read_text())
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_validate_gpu_resident_codec'],type_ignores=[])
    ns={};exec(compile(module,str(path),'exec'),ns)
    validate=ns['_validate_gpu_resident_codec']
    for codec in ('rvq_ours_avg0','rvq_ours_avg0p2_alt','rvq_ours_avg0p5','rvq_ours_avg1','rvq_ours_avg2','rvq_ours_avg4'):
        validate(SimpleNamespace(kv_cache_gpu_resident_compressed=True,kv_cache_offload=False,kv_cache_codec=codec))
    with pytest.raises(ValueError,match='non-none'):
        validate(SimpleNamespace(kv_cache_gpu_resident_compressed=True,kv_cache_offload=False,kv_cache_codec='none'))
    validate(SimpleNamespace(kv_cache_gpu_resident_compressed=False,kv_cache_offload=False,kv_cache_codec='none'))
    generate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_video')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_validate_gpu_resident_codec' for n in ast.walk(generate))

def test_avg4_optimizer_uses_only_calibrated_bits():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/HY-WorldPlay/build_moviegen_avg2_profile.py'
    spec=importlib.util.spec_from_file_location('profile_builder',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.SCHEDULE_BITS==(16,8,4,3,2,1)
    rows,_=m.solve_schedule({'distortions':{'8':.1,'4':.2,'3':.3,'2':.4,'1':.5},'age_weights':[1.]*10},capacity_bit_chunks=40)
    assert sum(rows[-1])==40
    assert all(sum(row)<=40 for row in rows)

def test_release_gpu_packed_span_preserves_text_without_cpu_move():
    path=Path(__file__).resolve().parents[1]/'experiments/HY-WorldPlay/hyvideo/models/transformers/worldplay_1_5_transformer.py'
    tree=ast.parse(path.read_text())
    names={'_retain_compressed_vision_on_gpu','_managed_vision_cache','_offload_cache_layer'}
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    class Tensor:
        device=SimpleNamespace(type='cuda')
        def to(self,device):pytest.fail(f'unexpected transfer to {device}')
    state=SimpleNamespace(kv_cache_offload=False,kv_cache_gpu_resident_compressed=True,kv_cache_codec='rvq_ours_avg0',kv_cache_offload_count=0)
    packed=({'indices':'K'},{'indices':'V'});text=Tensor()
    layer={'_qvg_packed_vision':packed,'_qvg_packed_vision_gpu':packed,'k_vision':Tensor(),'v_vision':Tensor(),'k_txt':text,'v_txt':text}
    ns={'get_infer_state':lambda:state,'torch':SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor))}
    exec(compile(module,str(path),'exec'),ns)
    ns['_offload_cache_layer'](layer,0)
    assert layer['k_vision'] is packed[0] and layer['k_txt'] is text
    assert layer['v_vision'] is packed[1] and layer['v_txt'] is text
    # The first text-cache pass has no compressed/video KV: test that branch too.
    state.kv_residual_stats_dir=''
    ns['_codec_settings']=lambda:object()
    ns['_move_cache_layer']=lambda *args:pytest.fail('unexpected CPU move')
    text_only={'k_vision':None,'v_vision':None,'k_txt':text,'v_txt':text}
    ns['_offload_cache_layer'](text_only,0)
    assert text_only['k_txt'] is text and text_only['v_txt'] is text
    state.kv_cache_gpu_resident_compressed=False
    assert not ns['_managed_vision_cache'](state)
