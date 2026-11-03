"""VEGA trainable path with frozen encoders kept outside ZeRO partitioning."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
from qwen_vl.model.feature_fusion import FeatureFusionConfig,FeatureFusionModule
from vega_qwen3 import WanPatchMerger

class TrainableVega(nn.Module):
    def __init__(self,qwen,context_dim=1536):
        super().__init__();self.qwen=qwen
        width=qwen.config.text_config.hidden_size
        self.generative_merger=WanPatchMerger(context_dim,width,qwen.config.vision_config.spatial_merge_size)
        self.generative_fusion=FeatureFusionModule(FeatureFusionConfig(fusion_method='token_gated_residual',hidden_size=width))
        for block in [self.generative_merger,self.generative_fusion]:
            for layer in block.modules():
                if isinstance(layer,nn.Linear):
                    nn.init.normal_(layer.weight,0,qwen.config.text_config.initializer_range);nn.init.zeros_(layer.bias)
        self.requires_grad_(True)
        self.train()

    def fused_visual(self,visual,wan,grid):
        n,(_,h,w)=len(grid),grid[0].tolist()
        gen=self.generative_merger(wan.reshape(n,h,w,-1))
        q=torch.cat(visual.pooler_output).reshape_as(gen)
        fused=self.generative_fusion(q,gen).reshape(-1,gen.shape[-1])
        return BaseModelOutputWithDeepstackFeatures(
            pooler_output=fused.split([len(v) for v in visual.pooler_output]),
            deepstack_features=visual.deepstack_features,
        )

    def forward(self,input_ids,labels,image_grid_thw,visual,wan,**kwargs):
        mm={'image':self.fused_visual(visual,wan,image_grid_thw)}
        output=self.qwen.model(input_ids=input_ids,image_grid_thw=image_grid_thw,
            mm_encoder_outputs=mm,use_cache=False,**kwargs)
        # Only supervised positions enter the LM head. Same shifted CE, without
        # allocating logits for thousands of ignored visual/prompt positions.
        keep=labels[:,1:]!=-100
        logits=self.qwen.lm_head(output.last_hidden_state[:,:-1][keep])
        return F.cross_entropy(logits.float(),labels[:,1:][keep])


class FrozenEncoders:
    def __init__(self,visual,wan):
        self.visual=visual.requires_grad_(False).eval()
        self.wan=wan.requires_grad_(False).eval()

    def activate(self,device):
        self.visual.to(device)
        self.wan.tower._move_models_to_device(torch.device(device))

    @torch.no_grad()
    def encode(self,batch,device):
        grid=batch['image_grid_thw'].to(device)
        pixels=batch['pixel_values'].split(grid.prod(-1).tolist())
        pools=[];layers=[[] for _ in self.visual.deepstack_visual_indexes];wans=[]
        # Frames are independent images in official VEGA. Microbatch one limits
        # peak memory without dropping frames, tokens or DeepStack branches.
        for j,pixel in enumerate(pixels):
            out=self.visual(pixel.to(device=device,dtype=self.visual.dtype),grid_thw=grid[j:j+1],return_dict=True)
            pools.append(out.pooler_output)
            for dest,layer in zip(layers,out.deepstack_features):dest.append(layer)
            wans.append(self.wan.encode(batch['wan_images'][j:j+1].to(device),grid_thw=grid[j:j+1]).to(dtype=out.pooler_output.dtype))
        visual=BaseModelOutputWithDeepstackFeatures(
            pooler_output=tuple(pools),deepstack_features=[tuple(layer) for layer in layers],
        )
        return {'visual':visual,'wan':torch.cat(wans),**{k:batch[k].to(device,non_blocking=True) for k in ['input_ids','labels','attention_mask','mm_token_type_ids','image_grid_thw']}}
