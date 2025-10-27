# --------------------------------------------------------------
# Dummy GPT Model
# --------------------------------------------------------------
# Bu örnek, GPT mimarisinin basitleştirilmiş bir sürümünü gösterir.
# Modelin yapısı doğru, fakat Transformer blokları ve LayerNorm “dummy” (boş) olarak bırakılmıştır.
# Amaç: GPT’nin tokenization → embedding → transformer → output head akışını adım adım anlamak.

import torch
import torch.nn as nn

# --------------------------------------------------------------
# 1️ Dummy GPT Model Tanımı
# --------------------------------------------------------------
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Token embedding tablosu: vocab_size × emb_dim
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Pozisyon embedding tablosu: context_length × emb_dim
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Embedding katmanından sonra dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer block katmanları (dummy versiyon)
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Son LayerNorm (dummy)
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # Çıkış katmanı: her token için vocab_size boyutunda olasılık (logit)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # seq_len = token sayısı

        # Token embedding’leri
        tok_embeds = self.tok_emb(in_idx)  # [batch, seq_len, emb_dim]

        # Pozisyon embedding’leri
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # [seq_len, emb_dim]

        # Token + Pozisyon bilgisi birleştirilir
        x = tok_embeds + pos_embeds

        # Dropout uygulanır
        x = self.drop_emb(x)

        # Transformer block’lara gönderilir
        x = self.trf_blocks(x)

        # Son katman normalizasyonu (dummy)
        x = self.final_norm(x)

        # Her token için logit değerleri (vocab_size uzunluğunda)
        logits = self.out_head(x)

        return logits


# --------------------------------------------------------------
# 2️ Dummy Transformer Block (Boş)
# --------------------------------------------------------------
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Bu class, gerçek TransformerBlock yapısına placeholder olarak eklenmiştir.

    def forward(self, x):
        # Şu an hiçbir işlem yapmaz, sadece girdiyi döner.
        return x


# --------------------------------------------------------------
# 3️ Dummy LayerNorm (Boş)
# --------------------------------------------------------------
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # Gerçek LayerNorm yapısının arayüzünü taklit etmek için bırakılmıştır.

    def forward(self, x):
        # Şu an hiçbir işlem yapmaz, sadece girdiyi döner.
        return x


# --------------------------------------------------------------
# 4️ GPT-2 Small (124M) Benzeri Konfigürasyon
# --------------------------------------------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Kelime sayısı (GPT-2'nin tokenizer'ı için)
    "context_length": 1024,  # Maksimum context uzunluğu
    "emb_dim": 768,          # Embedding boyutu
    "n_heads": 12,           # Attention head sayısı
    "n_layers": 12,          # Transformer block sayısı
    "drop_rate": 0.1,        # Dropout oranı
    "qkv_bias": False        # Query-Key-Value bias kullanımı
}

# --------------------------------------------------------------
# 5️ TOKENIZATION AŞAMASI
# --------------------------------------------------------------
# GPT-2 tokenizer kullanılarak metinler token ID’lerine dönüştürülür.

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# Metinler tokenize edilir
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

# Batch olarak birleştirilir (2 örnek, her biri 4 token)
batch = torch.stack(batch, dim=0)
print(batch)

# output:
# tensor([[6109, 3626, 6100,  345],     # first batch token IDs
#         [6109, 1110, 6622,  257]])    # second batch token IDs
#
# forward metodu içindeki seq_len = token sayısı = 4


# --------------------------------------------------------------
# 6️ MODEL OLUŞTURMA ve İLERİ (FORWARD) GEÇİŞ
# --------------------------------------------------------------
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)

print("Output shape:", logits.shape)
print(logits)

# output:
# Output shape: torch.Size([2, 4, 50257])
#
# tensor([[[-1.2034,  0.3201, -0.7130,  ..., -1.5548, -0.2390, -0.4667],
#          [-0.1192,  0.4539, -0.4432,  ...,  0.2392,  1.3469,  1.2430],
#          [ 0.5307,  1.6720, -0.4695,  ...,  1.1966,  0.0111,  0.5835],
#          [ 0.0139,  1.6754, -0.3388,  ...,  1.1586, -0.0435, -1.0400]],
#
#         [[-1.0908,  0.1798, -0.9484,  ..., -1.6047,  0.2439, -0.4530],
#          [-0.7860,  0.5581, -0.0610,  ...,  0.4835, -0.0077,  1.6621],
#          [ 0.3567,  1.2698, -0.6398,  ..., -0.0162, -0.1296,  0.3717],
#          [-0.2407, -0.7349, -0.5102,  ...,  2.0057, -0.3694,  0.1814]]],
#        grad_fn=<UnsafeViewBackward0>)
#
#   Açıklama:
# - Çıkan değerler şu anda rastgele, çünkü model eğitilmedi.
# - Her token için “vocab_size (50257)” uzunluğunda bir olasılık vektörü (logit) üretilir.
# - En yüksek logit değeri olan indeks, tahmin edilen sonraki token ID’sini verir.
# - Bu ID, tokenizer.decode() ile geri kelimeye çevrilebilir.


# --------------------------------------------------------------
# 7️ SONUÇ ÖZETİ
# --------------------------------------------------------------
# - DummyGPTModel, GPT-2 mimarisinin iskeletini oluşturur.
# - Tokenization → Embedding → (Transformer) → Output Head akışını gösterir.
# - Bu modelde hiçbir öğrenme veya attention işlemi yoktur.
# - Ama aynı boyutlarda gerçek GPT modelinin çalışma mantığını birebir simüle eder.
