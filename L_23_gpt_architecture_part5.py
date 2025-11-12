GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Kelime sayısı (GPT-2'nin tokenizer'ı için)
    "context_length": 1024,  # Maksimum context uzunluğu
    "emb_dim": 768,          # Embedding boyutu
    "n_heads": 12,           # Attention head sayısı
    "n_layers": 12,          # Transformer block sayısı
    "drop_rate": 0.1,        # Dropout oranı
    "qkv_bias": False        # Query-Key-Value bias kullanımı
}
import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased = False)
        norm_x = (x-mean)/ torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
# GELU aktivasyon fonksiyonu tanımı
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        )) 
        
# Feed Forward Katmanı        
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # Expansion
            GELU(),                                        # Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # Contraction
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    """
    Çok başlı (multi-head) self-attention mekanizmasının PyTorch implementasyonu.
    Her head, girdiye farklı bir alt uzayda dikkat uygular.
    Sonuçta tüm head’lerin çıktıları birleştirilip (concat + linear projeksiyon)
    modele daha zengin bir bağlamsal temsil kazandırılır.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, \
            "d_out değeri num_heads'e tam bölünebilir olmalıdır."

        # Temel boyutlar
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Her head’in embedding boyutu

        # Query, Key ve Value lineer dönüşümleri (öğrenilebilir ağırlıklar)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Head’lerin birleştirilmesinden sonra çıkış projeksiyonu
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout over attention weights
        self.dropout = nn.Dropout(dropout)

        # Üst üçgen (geleceğe bakan) değerleri maskeleyen causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        x: [batch_size, num_tokens, d_in]
        Output: [batch_size, num_tokens, d_out]
        """
        b, num_tokens, d_in = x.shape

        # 1️ Query-Key-Value hesaplama
        keys    = self.W_key(x)     # [b, num_tokens, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        # 2️ Her head için boyut ayrıştırma
        # [b, num_tokens, d_out] -> [b, num_tokens, num_heads, head_dim]
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 3️ Head eksenini öne al: [b, num_heads, num_tokens, head_dim]
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # 4️ Scaled Dot-Product Attention hesaplama
        # attn_scores = Q @ Kᵀ → [b, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(2, 3)

        # Maski kısalt ve boolean’a çevir
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 5️ Causal mask uygulanıyor: gelecekteki token’ları -inf ile maskeler
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 6️ Skorları normalize et (softmax + ölçekleme)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # overfitting önleme

        # 7️ Ağırlıklı toplama (attention-weighted sum)
        # [b, num_heads, num_tokens, head_dim]
        context_vec = attn_weights @ values

        # 8️ Head’leri birleştir: [b, num_tokens, num_heads * head_dim = d_out]
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        # 9️ Son lineer projeksiyon
        context_vec = self.out_proj(context_vec)

        return context_vec

# --------------------------------------------------------------
# Transformer Bloğu 
# --------------------------------------------------------------
# Girdi → LayerNorm1 → Masked Multi-Head Attention → Dropout → (+) Kısa yol bağlantısı
#       → LayerNorm2 → Feed Forward (Linear → GELU → Linear) → Dropout → (+) Kısa yol bağlantısı
# --------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out= cfg["emb_dim"],
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
        # short cut connection for attention blok
        # Girdi → LayerNorm1 → Masked Multi-Head Attention → Dropout → (+) Kısa yol bağlantısı
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) # shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut
        # feed forward block icin shortcut
        #       → LayerNorm2 → Feed Forward (Linear → GELU → Linear) → Dropout → (+) Kısa yol bağlantısı
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2,4,768) # A  -shape [batch_size, num_tokens, emb_size]
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape : " , x.shape)
    print("Output shape: " , output.shape)


        