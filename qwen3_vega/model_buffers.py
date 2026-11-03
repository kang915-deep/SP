"""ZeRO Init partitions parameters, but leaves non-parameter RoPE buffers on CPU."""
import torch
def snapshot_buffers(model):
    return {name:buffer.detach().clone() for name,buffer in model.named_buffers()}

def restore_buffers(model,buffers,device):
    for name,buffer in buffers.items():
        prefix,leaf=name.rsplit('.',1) if '.' in name else ('',name)
        model.get_submodule(prefix)._buffers[leaf]=buffer.to(device)
    for module in model.modules():
        if hasattr(module,'original_inv_freq'):
            module.original_inv_freq=module.inv_freq
