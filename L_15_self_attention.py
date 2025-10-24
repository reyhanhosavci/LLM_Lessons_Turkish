# --------------------------------------------------------------
# LLM Lessons - Lesson 15: Scaled Dot-Product Attention (Türkçe Açıklamalı)
# --------------------------------------------------------------

import torch

# --------------------------------------------------------------
# 1. Örnek giriş (6 token, her biri 3 boyutlu embedding)
# --------------------------------------------------------------
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your     (x¹)
   [0.55, 0.87, 0.66],  # journey  (x²)
   [0.57, 0.85, 0.64],  # starts   (x³)
   [0.22, 0.58, 0.33],  # with     (x⁴)
   [0.77, 0.25, 0.10],  # one      (x⁵)
   [0.05, 0.80, 0.55]]  # step     (x⁶)
)

# A → ikinci input (x²)
# B → input embedding boyutu (d_in = 3)
# C → output embedding boyutu (d_out = 2)

x_2 = inputs[1]              # A: ikinci token ("journey")
d_in = inputs.shape[1]       # B: giriş boyutu
d_out = 2                    # C: çıkış boyutu

# --------------------------------------------------------------
# 2. Key, Query ve Value için ağırlık matrisleri
# --------------------------------------------------------------
# Bu matrisler öğrenilebilir parametrelerdir.
# (Burada requires_grad=False → eğitim yapılmadığı için)
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# --------------------------------------------------------------
# 3. Tek bir token (x²) için Query, Key, Value vektörleri
# --------------------------------------------------------------
query_2 = x_2 @ W_query
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value

print("query_2:", query_2)  # İkinci token’a ait query vektörü

# --------------------------------------------------------------
# 4. Tüm tokenlar için Key, Query ve Value vektörleri
# --------------------------------------------------------------
# Adım 1: Input embedding’lerin K, Q, V uzayına projeksiyonu
keys    = inputs @ W_key
queries = inputs @ W_query
values  = inputs @ W_value

print("keys.shape:", keys.shape, "values.shape:", values.shape, "queries.shape:", queries.shape)
# Boyutlar → 6x2 olmalı (6 token, 2 boyutlu vektör)

# --------------------------------------------------------------
# 5. Attention skorlarının hesaplanması
# --------------------------------------------------------------
# Adım 2: Query ve Key vektörleri arasındaki benzerlikler (dot-product)
keys_2 = keys[1]
attn_score_22 = query_2.dot(key_2)
print("2. token’ın kendisiyle skoru:", attn_score_22)  # tensor(1.8524)

# 2. token’ın tüm token’lara karşı attention skorları
attn_score_2 = query_2 @ keys.T
print("2. token attention skorları:", attn_score_2)
# Örnek çıktı:
# tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])
# Yani "journey" token'ı, "starts" token'ına en çok dikkat ediyor (~1.81)

# Tüm skor matrisi: QKᵀ
attn_scores = queries @ keys.T
print("Tüm attention skor matrisi:\n", attn_scores)
# Boyut: 6x6

# --------------------------------------------------------------
# 6. Ölçekleme ve normalizasyon (Softmax)
# --------------------------------------------------------------
# d_k = embedding boyutu → skor varyansını normalize etmek için kullanılır.
# (√d_k ile bölme: scaled dot-product attention)
d_k = keys.shape[-1]

# Yalnızca 2. token için normalizasyon
attn_weights_2 = torch.softmax(attn_score_2 / d_k**0.5, dim=-1)
print("2. token attention ağırlıkları:", attn_weights_2)
print("d_k:", d_k)
# Çıktı:
# tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

# Tüm token’lar için attention ağırlıkları
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

# --------------------------------------------------------------
# 7. Context vektörlerinin hesaplanması
# --------------------------------------------------------------
# Context vektörü → attention ağırlıklarıyla Value vektörlerinin ağırlıklı ortalaması
context_vec_2 = attn_weights_2 @ values
print("2. token için context vektörü:", context_vec_2)
# Çıktı: tensor([0.3061, 0.8210])

# --------------------------------------------------------------
# 9. Aynı işlemi sınıf (SelfAttention_v1) kullanarak gerçekleştirme
# --------------------------------------------------------------
# Artık manuel olarak yaptığımız tüm adımlar, SelfAttention_v1 sınıfının içinde tanımlı.
# Bu sınıf Query, Key, Value matrislerini oluşturur ve scaled dot-product attention uygular.

from self_attention import SelfAttention_v1, SelfAttention_v2

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)  # key, query ve value ağırlık matrisleri oluşturulur

# Girdi embedding'leri (inputs) modele gönderildiğinde
# sınıf içindeki forward() metodu çağrılır ve context vektörleri oluşturulur.
context_vec_class = sa_v1(inputs)
print("SelfAttention_v1 ile oluşturulan context vektörleri:\n", context_vec_class)
# Çıktı:
# SelfAttention_v1 ile oluşturulan context vektörleri:
#  tensor([[0.2996, 0.8053],
#         [0.3061, 0.8210],
#         [0.3058, 0.8203],
#         [0.2948, 0.7939],
#         [0.2927, 0.7891],
#         [0.2990, 0.8040]], grad_fn=<MmBackward0>)

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# inputs: 6 token, her biri 3 boyutlu embedding
# forward() otomatik olarak çalışır ve context vektörleri döner.
context_vec_v2 = sa_v2(inputs)
print("SelfAttention_v2 ile oluşturulan context vektörleri:\n", context_vec_v2)
# Çıktı:
# SelfAttention_v2 ile oluşturulan context vektörleri:
#  tensor([[-0.0739,  0.0713],
#         [-0.0748,  0.0703],
#         [-0.0749,  0.0702],
#         [-0.0760,  0.0685],
#         [-0.0763,  0.0679],
#         [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
