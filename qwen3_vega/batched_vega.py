"""Right-padded independent QA rows. One supervised-token mean per example."""
import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
from train_model import TrainableVega


class BatchedVega(TrainableVega):
    def forward(self, examples=None, **single):
        if examples is None:
            return super().forward(**single)

        length = max(row['input_ids'].shape[1] for row in examples)
        tokens = {}
        for key, fill in [('input_ids', 0), ('attention_mask', 0), ('mm_token_type_ids', 0), ('labels', -100)]:
            tokens[key] = torch.cat([
                F.pad(row[key], (0, length - row[key].shape[1]), value=fill) for row in examples
            ])
        visual = [self.fused_visual(row['visual'], row['wan'], row['image_grid_thw']) for row in examples]
        merged = BaseModelOutputWithDeepstackFeatures(
            pooler_output=tuple(frame for output in visual for frame in output.pooler_output),
            deepstack_features=[
                tuple(frame for output in visual for frame in output.deepstack_features[layer])
                for layer in range(len(visual[0].deepstack_features))
            ],
        )
        labels = tokens.pop('labels')[:, 1:]
        hidden = self.qwen.model(**tokens, image_grid_thw=torch.cat([r['image_grid_thw'] for r in examples]),
                                 mm_encoder_outputs={'image': merged}, use_cache=False).last_hidden_state
        keep = labels != -100
        logits = self.qwen.lm_head(hidden[:, :-1][keep])
        token_losses = F.cross_entropy(logits.float(), labels[keep], reduction='none')
        counts = keep.sum(dim=1)
        owners = torch.arange(len(examples), device=labels.device).repeat_interleave(counts, output_size=token_losses.numel())
        return token_losses.new_zeros(len(examples)).scatter_add_(0, owners, token_losses) / counts
