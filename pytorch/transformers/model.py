import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, dim_model, vocab_size):
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim_model)

    def forward(self, x):
        return self.embedding * math.sqrt(self.dim_model)
    
class PositoinalEncoding(nn.Module):
    def __init__(self, dim_model, seq_len, dropout):
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = dropout

        pe = torch.zeros(self.seq_len, self.dim_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-math.log(10000)/dim_model))


        pe[:, 0::2] = torch.sin(position * denom)
        pe[:, 1::2] = torch.cos(position * denom)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        # save this along with the weight of the model

    def forward(self, x):
        x = x +  (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    def __init__(self, eps=10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / torch.sqrt( std+ self.eps)  + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, dim_model, dim_feedfor, dropout):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, dim_feedfor)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedfor, dim_model) # bias is True, so no need to added inthe linear layer

    def forward(self, x):
        return self.linear2(self.dropout(self.linear1(x)))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_model, h, dropout):
        super().__init__()
        self.dim_model = dim_model
        self.h = h
        assert dim_model % h == 0 # "dim_model should be divisible by h"
        self.d_k = dim_model // h

        self.w_q = nn.Linear(dim_model, dim_model)
        self.w_k = nn.Linear(dim_model, dim_model)
        self.w_v = nn.Linear(dim_model, dim_model)

        self.w_o = nn.Linear(self.h * self.d_k)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):

        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1))/ math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)

        # (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k)
        return (attention_scores @ value),  attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, dim_model) >> (batch, seq_len, dim_model)
        key = self.w_k(k)
        value = self.w_v(v)

        #(batch, seq_len, dim_model) >> (batch, seq_len, h, d_k) >> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) >> (batch, seq_len, dim_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h *self.d_k)
        # (batch, seq_len, dim_model) >> (batch, seq_len, dim_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.sublayer(self.norm(x))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.dropout = dropout
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block(x))
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, dim_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, target_embed: InputEmbeddings, src_pos:PositoinalEncoding, target_pos: PositoinalEncoding, proj: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.proj_layer = proj
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, targt, targt_mask):
        tgt = self.target_embed(tgt)
        tgt = self.target_pos(tgt)
        return self.decode(tgt, encoder_output, src_mask, targt_mask)
    
    def project(self, x):
        return self.project_layer(x)
    

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff = 2048):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositoinalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositoinalEncoding(d_model, tgt_seq_len, dropout)

    # encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block  = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,  decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # project layer
    project_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create a transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, project_layer)

    # initialize params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
