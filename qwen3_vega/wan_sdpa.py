"""Use PyTorch SDPA for the full-window, single-example Wan attention used by VEGA."""
import torch
from torch.nn.functional import scaled_dot_product_attention

def configure_attention():
    from qwen_vl.model.generative_encoders.wan.modules import model
    from qwen_vl.model.generative_encoders.wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        model.flash_attention=wan_sdpa

def wan_sdpa(q,k,v,k_lens=None,window_size=(-1,-1)):
    if q.shape[0]!=1 or tuple(window_size)!=(-1,-1):
        raise ValueError("VEGA SDPA adapter expects one image and full-window attention")
    if k_lens is not None:
        end=int(k_lens[0])
        k=k[:,:end];v=v[:,:end]
    dtype=v.dtype if v.dtype in (torch.float16,torch.bfloat16) else torch.bfloat16
    qh,kh,vh=(x.transpose(1,2).to(dtype) for x in (q,k,v))
    return scaled_dot_product_attention(qh,kh,vh).transpose(1,2).to(q.dtype)
