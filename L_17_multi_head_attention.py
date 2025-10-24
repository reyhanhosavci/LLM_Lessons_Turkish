import torch 
from self_attention import MultiHeadAttentionWrapper

# --------------------------------------------------------------
# 1️ MultiHeadAttentionWrapper (Basit Yapı)
# --------------------------------------------------------------
# Bu sınıf, birden fazla CausalAttention head'ini paralel olarak çalıştırır
# ve her head’in ürettiği context vektörlerini birleştirir (concatenate).
# Böylece model, farklı “dikkat pencereleri” üzerinden öğrenebilir.
# Dropout = 0 olarak ayarlanmıştır, böylece sonuçlar deterministik olur.

torch.manual_seed(123)

# Örnek input embedding'leri (6 kelime, her biri 3 boyutlu)
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your     (x¹)
   [0.55, 0.87, 0.66],  # journey  (x²)
   [0.57, 0.85, 0.64],  # starts   (x³)
   [0.22, 0.58, 0.33],  # with     (x⁴)
   [0.77, 0.25, 0.10],  # one      (x⁵)
   [0.05, 0.80, 0.55]]  # step     (x⁶)
)

# Aynı input'u iki defa stack’leyerek batch boyutu = 2 oluşturuyoruz
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]  # token sayısı = 6 
d_in, d_out = 3, 2 

# 2 başlı (num_heads=2) multi-head attention
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape: ", context_vecs.shape)

# çıktı:
#     tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
#          [-0.5874,  0.0058,  0.5891,  0.3257],
#          [-0.6300, -0.0632,  0.6202,  0.3860],
#          [-0.5675, -0.0843,  0.5478,  0.3589],
#          [-0.5526, -0.0981,  0.5321,  0.3428],
#          [-0.5299, -0.1081,  0.5077,  0.3493]],

#         [[-0.4519,  0.2216,  0.4772,  0.1063],
#          [-0.5874,  0.0058,  0.5891,  0.3257],
#          [-0.6300, -0.0632,  0.6202,  0.3860],
#          [-0.5675, -0.0843,  0.5478,  0.3589],
#          [-0.5526, -0.0981,  0.5321,  0.3428],
#          [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
# context_vecs.shape:  torch.Size([2, 6, 4])

#   Açıklama:
# - Her head, giriş token dizisini kendi parametreleriyle işler.
# - Çıktılar son boyutta birleştirilir (d_out * num_heads = 2 * 2 = 4).
# - Dropout kullanılmadığı için sonuç deterministiktir.
# - Her token için 4 boyutlu bir context vektörü elde edilir.


# --------------------------------------------------------------
# 2️ MultiHeadAttention (Transformer Tarzı Gerçek MHA)
# --------------------------------------------------------------
# Aşağıdaki örnek, "multi_head_attention.py" dosyasındaki daha gelişmiş sınıfı test eder.
# Bu yapı gerçek Transformer mimarilerinde kullanıldığı şekilde:
# scaled dot-product attention + causal mask + dropout + linear projection içerir.

from multi_head_attention import MultiHeadAttention
torch.manual_seed(123)

# --------------------------------------------------------------
# Girdi Tanımı
# --------------------------------------------------------------
# 3 token (satır) × 6 boyutlu embedding
inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # token 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # token 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # token 3
)

# Aynı input’tan 2 adet kopya oluşturularak batch=2 elde ediliyor
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
# çıktı:
#   torch.Size([2, 3, 6])
#  batch = 2 (örnek sayısı)
#  context_length = 3 (token sayısı)
#  d_in = 6 (her token’ın embedding boyutu)

batch_size, context_length, d_in = batch.shape
d_out = 6

# 2 head'li Multi-Head Self-Attention örneği
mhsa = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2) # dropout=0

context_vecs = mhsa(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# çıktı:
#   tensor([[[ 0.1569, -0.0873,  0.0210,  0.0215, -0.3243, -0.2518],
#          [ 0.1117, -0.0547,  0.0406, -0.0213, -0.3251, -0.2993],
#          [ 0.1196, -0.0491,  0.0318, -0.0635, -0.2788, -0.2578]],

#         [[ 0.1569, -0.0873,  0.0210,  0.0215, -0.3243, -0.2518],
#          [ 0.1117, -0.0547,  0.0406, -0.0213, -0.3251, -0.2993],
#          [ 0.1196, -0.0491,  0.0318, -0.0635, -0.2788, -0.2578]]],
#        grad_fn=<ViewBackward0>)
# context_vecs.shape: torch.Size([2, 3, 6])

#  Açıklama:
# - Her head, token’lar arası dikkat (self-attention) uygular.
# - Causal mask sayesinde her token yalnızca kendisi ve önceki token’lara bakar.
# - Softmax öncesi ölçekleme (1/√d_k) işlemi yapılır.
# - Tüm head çıktıları birleştirilir ve son lineer katmanla d_out boyutuna projekte edilir.
# - Elde edilen context_vecs tensorü, her token’ın “bağlama göre yeniden ifade edilmiş” halidir.
# - Transformer mimarisinde bu yapı residual bağlantı ve LayerNorm ile tamamlanır.
