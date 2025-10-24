# --------------------------------------------------------------
# LLM Lessons - Lesson 21: Multi-Head Self-Attention (Transformer Style)
# --------------------------------------------------------------
import torch
import torch.nn as nn

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
