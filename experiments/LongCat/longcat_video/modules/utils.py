import torch


def _move_item(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return {name: _move_item(value, device) for name, value in item.items()}
    if isinstance(item, list):
        return [_move_item(value, device) for value in item]
    if isinstance(item, tuple):
        return tuple(_move_item(value, device) for value in item)
    assert not hasattr(item, "device")
    return item


def _offload_kv_cache(kv_cache: tuple) -> tuple:
    """
    Offload KV cache to CPU.
    
    Args:
        kv_cache: Tuple of (k_quant, v_quant), where each can be a Tensor or dict
        
    Returns:
        Tuple of offloaded (k_quant, v_quant)
    """
    k_quant, v_quant = kv_cache
    
    return _move_item(k_quant, "cpu"), _move_item(v_quant, "cpu")


def _onload_kv_cache(kv_cache: tuple, device: torch.device) -> tuple:
    """
    Onload KV cache to the specified device.
    
    Args:
        kv_cache: Tuple of (k_quant, v_quant), where each can be a Tensor or dict
        device: Target device
        
    Returns:
        Tuple of onloaded (k_quant, v_quant)
    """
    k_quant, v_quant = kv_cache
    
    return _move_item(k_quant, device), _move_item(v_quant, device)
