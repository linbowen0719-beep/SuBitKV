"""CUDA codec workspace measurements, separate from persistent KV storage.

The reported footprint sums persistent payload, shared parameters and the largest
codec workspace observed in the run. It is a conservative component sum, not a
simultaneous whole-process peak. Model weights/attention activations are outside
the cache metric; the whole-process allocated peak is retained separately.
"""

from functools import wraps

_depth = 0
_workspace = 0
_process_peak = 0
_calls = 0
_shared = 0
_shared_current = 0


def reset_workspace_stats():
    global _workspace, _process_peak, _calls, _shared, _shared_current
    _workspace = _process_peak = _calls = _shared = _shared_current = 0


def _storages(value):
    import torch

    result = {}
    def visit(item):
        if torch.is_tensor(item) and item.is_cuda:
            storage = item.untyped_storage()
            result[(str(item.device), storage.data_ptr())] = (storage.nbytes(), item.device)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
    visit(value)
    return result


def track_workspace(kind):
    def decorate(function):
        @wraps(function)
        def tracked(*args, **kwargs):
            global _depth, _workspace, _process_peak, _calls, _shared, _shared_current
            import torch

            if _depth or not torch.cuda.is_available():
                return function(*args, **kwargs)
            inputs = _storages((args, kwargs))
            if not inputs:
                return function(*args, **kwargs)
            device = next(iter(inputs.values()))[1]
            if any(value[1] != device for value in inputs.values()):
                raise ValueError("Codec workspace measurement requires one CUDA device")
            torch.cuda.synchronize(device)
            _process_peak = max(_process_peak, torch.cuda.max_memory_allocated(device))
            start = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            _depth += 1
            result = None
            try:
                result = function(*args, **kwargs)
                return result
            finally:
                _depth -= 1
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                end = torch.cuda.memory_allocated(device)
                _process_peak = max(_process_peak, peak)
                outputs = _storages(result)
                # CUDA's small allocation granularity is 512 bytes. Persistent
                # allocations beyond returned payload are codec parameter caches.
                new_payload = sum((size + 511) // 512 * 512 for key, (size, _) in outputs.items() if key not in inputs)
                _shared_current = max(0, _shared_current + end - start - new_payload)
                _shared = max(_shared, _shared_current)
                if kind == "encode":
                    # Charge unretained source K/V buffers plus temporary encode
                    # allocations; retained BF16 bypass tensors are payload.
                    source = sum(size for key, (size, _) in inputs.items() if key not in outputs)
                    workspace = max(0, peak - max(start, end)) + source
                else:
                    # Decoded full-precision output is cache workspace too.
                    workspace = max(0, peak - start)
                _workspace = max(_workspace, workspace)
                _calls += 1
        return tracked
    return decorate


def workspace_stats():
    return dict(
        codec_workspace_mib=_workspace / 1024**2,
        codec_resident_shared_mib=_shared / 1024**2,
        codec_workspace_calls=_calls,
        codec_workspace_scope="max encode source/scratch or decode output/scratch; CUDA allocated bytes",
        codec_process_peak_mib=_process_peak / 1024**2,
        memory_accounting_scope="persistent KV including metadata + shared parameters + maximum codec workspace (component sum)",
    )


def cache_footprint(payload_mib, shared_mib, workspace_mib):
    if any(v is None for v in (payload_mib, shared_mib, workspace_mib)):
        return None
    if any(v < 0 for v in (payload_mib, shared_mib, workspace_mib)):
        raise ValueError("Negative cache memory component")
    return payload_mib + shared_mib + workspace_mib
