"""
Full Definition of the GPT model with rotary embeddings and RMSNorm.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from .configuration_gpt import GPTConfig


################################
###         Layers           ###
################################

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class RMSNorm(nn.Module):
    """ Root Mean Square Normalization """
    def __init__(self, dim: int, weight: bool = False, bias: bool = False, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        if weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return self.weight * x / (norm + self.eps)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


################################
###          Model           ###
################################

class GPT(PreTrainedModel):
    config_class = GPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # GPT-2 style scaled init
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        tok_emb = self.transformer.wte(input_ids)
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)['logits']
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer='adamw'):
        """
        Create optimizer groups to handle weight decay for certain parameters.
        We also tag parameters for 'embedding', 'body', and 'head' so that the
        Muon optimizer can use different hyperparameters or schedules for each group
        (if desired). This follows the dimension-based weight decay logic used in
        the standard GPT class, but now organized by parameter type.

        Args:
            weight_decay (float): Weight decay for decayed parameters.
            learning_rate (float): Base learning rate.
            betas (tuple): Betas for Adam-type optimizers.
            device_type (str): e.g. 'cuda' or 'cpu'.
            optimizer (str): Optimizer type, one of ['adamw', 'adam', 'sgd', 'muon'].

        Returns:
            torch.optim.Optimizer or Muon: The configured optimizer instance.
        """
        # 1) Collect all parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # 2) Separate parameters by "type" to support Muon param_type:
        #    - "embedding" for word + position embeddings
        #    - "body" for transformer blocks and final layer norm
        #    - "head" for the language model head (lm_head)
        embedding_names = [n for n in param_dict if ('transformer.wte' in n or 'transformer.wpe' in n)]
        body_names = [
            n for n in param_dict
            if ('transformer.h.' in n or 'transformer.ln_f.' in n)
        ]
        head_names = [n for n in param_dict if 'lm_head' in n]

        embedding_params = [param_dict[n] for n in embedding_names]
        body_params = [param_dict[n] for n in body_names]
        head_params = [param_dict[n] for n in head_names]

        all_params = embedding_params + body_params + head_params

        # Summaries for printing
        num_params = sum(p.numel() for p in embedding_params + body_params + head_params)
        print(f"num parameter tensors: {len(embedding_params + body_params + head_params)}, "
            f"with {num_params:,} parameters")

        # 5) Create the optimizer
        import inspect

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = (fused_available and device_type == 'cuda')
        extra_args = {}
        # Only Adam/AdamW can use the fused version
        if optimizer in ['adamw', 'adam'] and use_fused:
            extra_args['fused'] = True

        if optimizer == 'adamw':
            opt = torch.optim.AdamW(all_params, lr=learning_rate, betas=betas, **extra_args)
        elif optimizer == 'adam':
            opt = torch.optim.Adam(all_params, lr=learning_rate, betas=betas, **extra_args)
        elif optimizer == 'sgd':
            opt = torch.optim.SGD(all_params, lr=learning_rate, momentum=0.9)
        elif optimizer == 'muon':
            try:
                from muon import Muon
            except ImportError:
                raise ImportError(
                    "Muon optimizer not installed. Please install it via:\n"
                    "pip install git+https://github.com/KellerJordan/Muon"
                )
            opt = Muon(body_params, lr=0.1*learning_rate, momentum=0.95,
                 adamw_params=embedding_params+head_params, adamw_lr=learning_rate, 
                 adamw_betas=betas, adamw_wd=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        print(f"Using optimizer: {optimizer}, fused support: {extra_args.get('fused', False)}")
        return opt


    def estimate_mfu(self, fwdbwd_per_iter, T, dt, accelerator_type='A40'):
        """
        Estimate model flops utilization (MFU).
        fwdbwd_per_iter: how many forward+backward passes we do each iteration
        T: sequence length
        dt: iteration time (seconds)
        """
        # number of parameters
        N = self.get_num_params()
        cfg = self.config
        L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head

        # approximate flops
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt  # per second

        # peak FLOPS for different accelerators
        if accelerator_type == 'A40':
            flops_promised = 149.7e12   # A40 bfloat16 TFLOPS
        elif accelerator_type == 'A100':
            flops_promised = 312e12    # A100 bfloat16 TFLOPS
        elif accelerator_type == 'H100':
            flops_promised = 756e12
        elif accelerator_type == 'TPUv4':
            flops_promised = 275e12
        elif accelerator_type == 'TPUv5e':
            flops_promised = 197e12
        else:
            raise ValueError(f"Unknown accelerator_type: {accelerator_type}")

        mfu = flops_achieved / flops_promised
        return mfu