import torch
import torch.nn as nn

class VisualPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.cfg = cfg
        n_ctx = cfg.n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        
        self.ctx = nn.Parameter(ctx_vectors)

    def forward(self, batch_size):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(batch_size, -1, -1)
        return ctx