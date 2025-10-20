import torch
import torch.nn as nn

# --------------------------------------------------------------
# Self-Attention Mekanizmasının Temel Uygulaması
# --------------------------------------------------------------
class SelfAttention_v1(nn.Module):
    """
    Basit Self-Attention katmanı.
    Her input embedding için Query, Key, Value vektörleri oluşturur,
    ardından scaled dot-product attention uygular:
    
        Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
    """

    def __init__(self, d_in, d_out):
        """
        d_in  : giriş embedding boyutu
        d_out : çıkış embedding boyutu
        """
        super().__init__()
        # Öğrenilebilir ağırlık matrisleri
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        """
        x: Girdi matrisi (batch_size x seq_len x d_in)
        """
        # 1. Q, K, V matrislerini oluştur
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value

        # 2. Attention skorlarını hesapla (QKᵀ)
        attn_scores = queries @ keys.T

        # 3. Skorları ölçekle ve softmax ile normalize et
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1
        )

        # 4. Ağırlıklarla Value'ları topla (context vektörleri)
        context_vec = attn_weights @ values

        return context_vec

# --------------------------------------------------------------
# Self-Attention Katmanı (nn.Linear ile)
# --------------------------------------------------------------
import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    """
    Self-Attention mekanizmasının Linear katmanlar ile uygulanmış versiyonu.
    Her token embedding'i için Query, Key ve Value vektörleri oluşturur.
    Ardından scaled dot-product attention uygular:

        Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        Parametreler:
        --------------
        d_in : int
            Giriş embedding boyutu (örneğin 768)
        d_out : int
            Çıkış embedding boyutu (örneğin 512)
        qkv_bias : bool
            Linear katmanlarda bias kullanılacak mı (varsayılan: False)
        """
        super().__init__()
        # Linear katmanlar: W_query, W_key ve W_value matrislerini öğrenir
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        x : torch.Tensor
            Girdi matrisi (örneğin: [token_sayısı, d_in])
        Dönüş :
            Context vektörleri (her token için zenginleştirilmiş temsil)
        """
        # 1️⃣ Query, Key, Value matrislerini oluştur
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # 2️⃣ Dot-product attention skorlarını hesapla
        attn_scores = queries @ keys.T  # QKᵀ

        # 3️⃣ Skorları √dₖ ile ölçekle ve softmax ile normalize et
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1
        )

        # 4️⃣ Ağırlıklarla Value’ları topla → context vektörleri
        context_vec = attn_weights @ values

        return context_vec

class CausalAttention(nn.Module):
    """
    Causal (masked) self-attention katmanı.
    Bu versiyonda, her token yalnızca KENDİSİNDEN ÖNCEKİ token’lara bakabilir.
    (Yani gelecek token’lar maskelenir.)
    
    Kullanıldığı yer: GPT, decoder-only transformer modelleri.
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        Parametreler:
        --------------
        d_in : int
            Giriş embedding boyutu
        d_out : int
            Çıkış embedding boyutu
        context_length : int
            Maksimum sequence (context) uzunluğu — mask boyutunu belirler
        dropout : float
            Attention ağırlıklarına uygulanacak dropout oranı
        qkv_bias : bool
            Linear katmanlarda bias eklensin mi (varsayılan: False)
        """
        super().__init__()
        self.d_out = d_out

        # Query, Key, Value dönüşümleri için Linear katmanlar
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout: overfitting’i azaltmak için uygulanır
        self.dropout = nn.Dropout(dropout)

        # Üst üçgen mask (upper triangular mask)
        # Üst üçgen kısım 1 → gelecek token’ları gizler (causal attention)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        x : torch.Tensor
            Girdi tensörü (boyut: [batch_size, num_tokens, d_in])
        Dönüş :
            context_vec : torch.Tensor
                Çıktı tensörü (boyut: [batch_size, num_tokens, d_out])
        """
        # Batch boyutunu da içeren giriş şekli
        b, num_tokens, d_in = x.shape

        # 1 Query, Key, Value matrislerini oluştur
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # 2 Attention skorlarını hesapla (QKᵀ)
        attn_scores = queries @ keys.transpose(1, 2)

        # 3 Gelecekteki token’ları maskele
        # mask.bool()[:num_tokens, :num_tokens] → yalnızca mevcut sequence kadarını alır
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        # 4 Ölçekle, softmax uygula ve dropout ekle
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # 5 Ağırlıklarla Value vektörlerini topla → context vektörleri
        context_vec = attn_weights @ values

        return context_vec
