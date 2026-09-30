"""Public package exports for SuBitKV.

The package root stays lightweight so standalone submodules such as
``subitkv.cache_budget`` can be imported without initializing every quantizer.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ChunkedKVCache",
    "QuantizeConfig",
    "TimeLoggingContext",
    "compress_kv_cache",
    "get_quantize_fn",
    "get_quantize_type",
    "logger",
    "offload_kv_cache_layer",
    "onload_kv_cache_layer",
    "time_logging_decorator",
    "uncompress_kv_cache",
]

_EXPORTS = {
    "compress_kv_cache": (".compress", "compress_kv_cache"),
    "get_quantize_fn": (".compress", "get_quantize_fn"),
    "get_quantize_type": (".compress", "get_quantize_type"),
    "uncompress_kv_cache": (".uncompress", "uncompress_kv_cache"),
    "ChunkedKVCache": (".kv_cache", "ChunkedKVCache"),
    "offload_kv_cache_layer": (".kv_cache", "offload_kv_cache_layer"),
    "onload_kv_cache_layer": (".kv_cache", "onload_kv_cache_layer"),
    "QuantizeConfig": (".sim.quant.quantize_config", "QuantizeConfig"),
    "TimeLoggingContext": (".timer", "TimeLoggingContext"),
    "time_logging_decorator": (".timer", "time_logging_decorator"),
    "logger": (".logger", "logger"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
