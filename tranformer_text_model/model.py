# import torch
# import torch.nn as nn


# class SelfAttention(nn.Module):

#     def __init__(self, embed_size, heads):
#         super().__init__()

#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads

#         assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

#     def forward(self, values, keys, query, mask):

#         N = query.shape[0]
#         value_len = values.shape[1]
#         key_len = keys.shape[1]
#         query_len = query.shape[1]

#         # Split embedding into heads
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         queries = query.reshape(N, query_len, self.heads, self.head_dim)

#         values = self.values(values)
#         keys = self.keys(keys)
#         queries = self.queries(queries)

#         energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))

#         attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
#         out = torch.einsum("nhql,nlhd->nqhd", attention, values)
#         out = out.reshape(N, query_len, self.heads * self.head_dim)
#         out = self.fc_out(out)
#         return out


# class TransformerBlock(nn.Module):

#     def __init__(self, embed_size, heads, dropout, forward_expansion):
#         super().__init__()

#         self.attention = SelfAttention(embed_size, heads)
#         self.norm1 = nn.LayerNorm(embed_size)
#         self.norm2 = nn.LayerNorm(embed_size)

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_size, forward_expansion * embed_size),
#             nn.ReLU(),
#             nn.Linear(forward_expansion * embed_size, embed_size)
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, value, key, query, mask):

#         attention = self.attention(value, key, query, mask)
#         x = self.dropout(self.norm1(attention + query))
#         forward = self.feed_forward(x)
#         out = self.dropout(self.norm2(forward + x))
#         return out


# class TinyTransformer(nn.Module):

#     def __init__(
#         self,
#         vocab_size,
#         embed_size=128,
#         num_layers=2,
#         heads=4,
#         forward_expansion=4,
#         dropout=0.1,
#         max_length=200
#     ):
#         super().__init__()

#         self.embed_size = embed_size
#         self.max_length = max_length

#         self.word_embedding = nn.Embedding(vocab_size, embed_size)
#         self.position_embedding = nn.Embedding(max_length, embed_size)

#         self.layers = nn.ModuleList(
#             [
#                 TransformerBlock(embed_size, heads, dropout, forward_expansion)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.fc_out = nn.Linear(embed_size, vocab_size)
#         self.dropout = nn.Dropout(dropout)

#         # Causal mask for speed
#         mask = torch.tril(torch.ones(max_length, max_length))
#         mask = mask.unsqueeze(0).unsqueeze(0)
#         self.register_buffer("mask", mask)

#     def forward(self, x):

#         N, seq_length = x.shape
#         positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
#         out = self.dropout(
#             self.word_embedding(x) + self.position_embedding(positions)
#         )
#         mask = self.mask[:, :, :seq_length, :seq_length]

#         for layer in self.layers:
#             out = layer(out, out, out, mask)

#         logits = self.fc_out(out)
#         return logits


import torch
import torch.nn as nn

class TinyTransformer(nn.Module):

    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=2, seq_len=64):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(seq_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):

        B, T = x.shape

        positions = torch.arange(0, T, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.pos_embedding(positions)

        x = self.transformer(x)

        logits = self.fc(x)

        return logits
