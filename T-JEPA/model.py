# model.py - Fixed version
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, depth: int, num_heads: int, embed_dim: int, mlp_ratio: float, norm_layer: Type[nn.Module]):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        ffn_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = SwiGLU(dim=embed_dim, hidden_dim=ffn_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        normed_x = self.norm1(x)
        x = x + self.attn(normed_x, normed_x, normed_x, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TextJEPA(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_config = config['model']
        data_config = config['data']

        embed_dim = model_config['embed_dim']
        vocab_size = data_config['vocab_size']
        max_length = data_config['max_length']
        
        norm_layer = RMSNorm

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_length, embed_dim))

        self.context_encoder = self._create_encoder(model_config['context_encoder'], embed_dim, norm_layer)
        self.target_encoder = self._create_encoder(model_config['target_encoder'], embed_dim, norm_layer)
        self.predictor = self._create_encoder(model_config['predictor'], embed_dim, norm_layer)

        self._initialize_weights()
        self._sync_and_freeze_target_encoder()

    def _create_encoder(self, encoder_config: dict, embed_dim: int, norm_layer: Type[nn.Module]) -> nn.Module:
        layers = []
        for _ in range(encoder_config['depth']):
            layers.append(TransformerBlock(
                depth=encoder_config['depth'],
                num_heads=encoder_config['num_heads'],
                embed_dim=embed_dim,
                mlp_ratio=encoder_config['mlp_ratio'],
                norm_layer=norm_layer
            ))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.positional_embeddings, std=0.02)
        nn.init.trunc_normal_(self.token_embeddings.weight, std=0.02)
        self.apply(self._init_transformer_weights)

    def _init_transformer_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _sync_and_freeze_target_encoder(self):
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        momentum = self.config['model']['momentum']
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_k.data * momentum + param_q.data * (1. - momentum))

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        embeddings = self.token_embeddings(input_ids)
        return embeddings + self.positional_embeddings[:, :seq_len, :]

    def forward(self, context_tokens: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        batch_size, seq_len = context_tokens.shape
        
        # Embed inputs
        context_embeddings = self._embed(context_tokens)
        
        # Get target representations (no gradients)
        with torch.no_grad():
            self.target_encoder.eval()
            target_embeddings = self._embed(target_tokens)
            target_output = self.target_encoder(target_embeddings)
        
        # Get context representations
        context_output = self.context_encoder(context_embeddings)
        
        # Predict representations for masked positions
        predicted_representations = self.predictor(context_output)
        
        # Extract representations for masked positions only
        # Reshape to handle batch dimension properly
        mask_flat = mask.view(-1)
        target_flat = target_output.view(-1, target_output.size(-1))
        pred_flat = predicted_representations.view(-1, predicted_representations.size(-1))
        
        if mask_flat.sum() == 0:
            # If no tokens are masked, return zero loss
            loss = torch.tensor(0.0, device=context_tokens.device, requires_grad=True)
            return {"loss": loss, "predictions": pred_flat[mask_flat], "targets": target_flat[mask_flat]}
        
        # Only compute loss on masked positions
        target_masked = target_flat[mask_flat]
        pred_masked = pred_flat[mask_flat]
        
        loss = F.mse_loss(pred_masked, target_masked)

        return {"loss": loss, "predictions": pred_masked, "targets": target_masked}