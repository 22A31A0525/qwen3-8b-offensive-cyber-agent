import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import bitsandbytes as bnb
from kaggle.train_qwen3_8b import patch_causal_lm_chunked_loss

class DummyBackbone(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_dim)
        self.last_kwargs = {}

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.last_kwargs = kwargs
        self.last_mask = attention_mask
        h = self.embed(input_ids)
        return type("Output", (), {"last_hidden_state": h})()

class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.model = DummyBackbone(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = type("Config", (), {"use_cache": True})()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(out.last_hidden_state)
        return CausalLMOutputWithPast(logits=logits)

def test_chunked_loss_and_use_cache_patch():
    model = DummyCausalLM()
    patch_causal_lm_chunked_loss(model, chunk_size=4)

    assert model.config.use_cache is False

    # 1. Unlabeled inference pass
    input_ids = torch.randint(0, 100, (1, 10))
    attn_mask = torch.ones(1, 10, dtype=torch.long)
    out_infer = model(input_ids=input_ids, attention_mask=attn_mask)
    assert out_infer.logits is not None
    assert out_infer.logits.shape == (1, 10, 100)

    # 2. Labeled training pass with all 1s attention mask -> model_mask must be None for SDPA memory efficiency
    labels = torch.randint(0, 100, (1, 10))
    labels[0, :3] = -100 # test assistant loss masking
    out_train = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    assert out_train.loss is not None
    assert out_train.loss.item() > 0
    assert model.model.last_kwargs.get("use_cache") is False
    assert model.model.last_mask is None # Attention mask should be None when all tokens valid

    # 3. Backward pass check
    out_train.loss.backward()
    assert model.lm_head.weight.grad is not None

def test_adamw_8bit_optimization():
    model = nn.Linear(32, 32)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    x = torch.randn(2, 32)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    sd = opt.state_dict()
    assert len(sd["state"]) > 0
    opt.load_state_dict(sd)

