"""
Configs for the GPT Languages model
"""
from transformers import PretrainedConfig

class GPTConfig(PretrainedConfig):
    model_type = "GenerationalPruning.GPT"

    def __init__(
        self,
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
        pos_init=None,
        **kwargs
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.pos_init = pos_init
        super().__init__(**kwargs)
