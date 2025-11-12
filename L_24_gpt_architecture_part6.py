# --------------------------------------------------------------
# GPT-2 Mini Mimarisi (Basitlestirilmis Uygulama)
# --------------------------------------------------------------
# Girdi → Token Embedding + Positional Embedding
#       → N × Transformer Blogu
#           ├─ LayerNorm1 → Masked Multi-Head Attention → Dropout → (+) Kisa yol baglantisi
#           └─ LayerNorm2 → Feed Forward (Linear → GELU → Linear) → Dropout → (+) Kisa yol baglantisi
#       → LayerNorm → Cikis (Output Projection)
# --------------------------------------------------------------

import torch
import torch.nn as nn

# --------------------------------------------------------------
# GPT-2 (124M) Konfigurasyon Parametreleri
# --------------------------------------------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Kelime dagarcigi (tokenizer kelime sayisi)
    "context_length": 1024,  # Maksimum baglam (token) uzunlugu
    "emb_dim": 768,          # Embedding boyutu
    "n_heads": 12,           # Attention baslik (head) sayisi
    "n_layers": 12,          # Transformer blok sayisi
    "drop_rate": 0.1,        # Dropout orani
    "qkv_bias": False        # Query-Key-Value bias kullanimi
}

# --------------------------------------------------------------
# Katman Normalizasyonu (Layer Normalization)
# --------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# --------------------------------------------------------------
# GELU Aktivasyon Fonksiyonu
# --------------------------------------------------------------
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# --------------------------------------------------------------
# Feed Forward Katmani
# --------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # Genisletme
            GELU(),                                        # Aktivasyon
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])  # Daraltma
        )

    def forward(self, x):
        return self.layers(x)

# --------------------------------------------------------------
# Multi-Head Self-Attention Mekanizmasi
# --------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Cok basli (multi-head) self-attention mekanizmasi.
    Her head girdiye farkli bir alt uzayda dikkat (attention) uygular.
    Sonucta tum head'lerin ciktilari birlestirilip (concat + linear projeksiyon)
    modele daha zengin baglamsal bilgi kazandirilir.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out degeri num_heads'e tam bolunebilir olmalidir."

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Gelecege bakan (ust ucgen) degerleri maskeleyen causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Query, Key, Value hesaplama
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Boyutlari head'lere ayirma
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Skor hesaplama (Q @ Kᵀ)
        attn_scores = queries @ keys.transpose(2, 3)

        # Mask uygulaniyor (gelecege bakmayi engeller)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Softmax ile normalizasyon ve dropout
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Agirlikli toplama (attention-weighted sum)
        context_vec = attn_weights @ values

        # Head'leri birlestirme
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

# --------------------------------------------------------------
# Transformer Blogu
# --------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention blogu
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # kisa yol baglantisi

        # Feed Forward blogu
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # kisa yol baglantisi
        return x

# --------------------------------------------------------------
# GPT Modeli
# --------------------------------------------------------------
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])   # Token embedding
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # Pozisyon embedding
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer bloklari (stack)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# --------------------------------------------------------------
# Test Asamasi
# --------------------------------------------------------------
if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # Ornek metinler
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    # Tokenize etme
    batch = torch.stack([
        torch.tensor(tokenizer.encode(txt1)),
        torch.tensor(tokenizer.encode(txt2))
    ], dim=0)

    print(batch)
    # tensor([[6109, 3626, 6100,  345],
    #         [6109, 1110, 6622,  257]])

    # Modeli calistir
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print("Sample logits:\n", logits)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    tied_params = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Parameters with weight tying: {tied_params:,}")
