"""Train only the added segmentation-token rows, keeping the vocabulary frozen."""
import torch
from torch import nn


class TokenRowEmbedding(nn.Module):
    def __init__(self, base, token_id):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.token_id = int(token_id)
        self.delta = nn.Parameter(base.weight.new_zeros(base.embedding_dim, dtype=torch.float32))
        self.num_embeddings, self.embedding_dim = base.num_embeddings, base.embedding_dim

    @property
    def weight(self):
        return self.base.weight

    def forward(self, token_ids):
        output = self.base(token_ids)
        return output + (token_ids == self.token_id).unsqueeze(-1) * self.delta.to(output.dtype)


class TokenRowLMHead(nn.Module):
    def __init__(self, base, token_id):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.token_id = int(token_id)
        self.delta = nn.Parameter(base.weight.new_zeros(base.in_features, dtype=torch.float32))
        self.in_features, self.out_features = base.in_features, base.out_features

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        logits = self.base(hidden)
        delta = (hidden.float() * self.delta).sum(-1).to(logits.dtype)
        # Clone avoids an in-place modification of a view used by autograd.
        logits = logits.clone()
        logits[..., self.token_id] = logits[..., self.token_id] + delta
        return logits
