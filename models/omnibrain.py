import torch
import torch.nn as nn
import torch.nn.functional as F

class OmniBrainModel(nn.Module):
    def __init__(self, 
                 vocab_size=32000, 
                 hidden_size=512, 
                 num_layers=6, 
                 num_heads=8,
                 num_experts=6,
                 max_position_embeddings=2048):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Token + Position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Transformer layers (simplified)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=num_heads, 
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Mixture of Experts router (simplified)
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Output layer
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        print(f"OmniBrainModel initialized with {num_experts} experts")

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        
        # Simple MoE routing (for now just average)
        router_logits = self.router(x)
        expert_weights = F.softmax(router_logits, dim=-1)
        
        # Final projection
        logits = self.lm_head(x)
        
        return logits
