import torch 
from self_attention import SelfAttention_v2, CausalAttention

# --------------------------------------------------------------
# 1 Örnek input token embedding'leri
# --------------------------------------------------------------
# 6 kelimelik bir cümleye ait örnek embedding vektörleri (her biri 3 boyutlu).
# Gerçek modellerde bu değerler, embedding katmanından öğrenilir.
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your     (x¹)
   [0.55, 0.87, 0.66],  # journey  (x²)
   [0.57, 0.85, 0.64],  # starts   (x³)
   [0.22, 0.58, 0.33],  # with     (x⁴)
   [0.77, 0.25, 0.10],  # one      (x⁵)
   [0.05, 0.80, 0.55]]  # step     (x⁶)
)

from self_attention import SelfAttention_v2
d_in = inputs.shape[1]   # giriş boyutu (embedding boyutu)
d_out = 2                # çıkış boyutu

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# inputs: 6 token, her biri 3 boyutlu embedding
queries = sa_v2.W_query(inputs)
keys    = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T  # QK^T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

print(attn_weights)
# çıktı:
#     tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
#         [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],      
#         [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],      
#         [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],      
#         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],      
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]], 


# --------------------------------------------------------------
# 2️ Alt üçgen mask (lower-triangular)
# --------------------------------------------------------------
# Attention weight’lerin üst üçgen kısmındaki (gelecekteki token’lar)
# ağırlıkları sıfırlıyoruz. Böylece her token yalnızca kendisi ve
# kendinden önceki token’lara dikkat edebilir (causal structure).
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
# çıktı:
#     tensor([[1., 0., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1., 1.]])

masked_simple = attn_weights * mask_simple
print(masked_simple)
# çıktı:
# tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],      
#         [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],      
#         [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],      
#         [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],      
#         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],      
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],     
#        grad_fn=<MulBackward0>)


# --------------------------------------------------------------
# 3️ Yeniden normalizasyon
# --------------------------------------------------------------
# Mask uygulandıktan sonra, her bir satırın toplamı 1 olacak şekilde
# yeniden normalize edilir. Bu, attention dağılımını korur.
rows_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / rows_sums
print(masked_simple_norm)
# çıktı:
#     tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],      
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],      
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],      
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],      
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],      
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],     
#        grad_fn=<DivBackward0>)


# --------------------------------------------------------------
# 4️ Softmax öncesi mask uygulanması (-inf ile)
# --------------------------------------------------------------
# Üst üçgen sıfırlansa bile, attention skorları hâlâ tüm token’lardan etkilenir.
# Bu nedenle softmax öncesinde -inf maskesi uygulanır.
print(attn_scores)
# çıktı: orijinal attention skorları
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
# çıktı:
#     tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
#         [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
#         [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
#         [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
#         [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
#         [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]], grad_fn=<MaskedFillBackward0>)

# Bundan sonra softmax uygulandığında -inf olan değerler sıfıra dönüşür.
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)
# çıktı:
#     tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]], grad_fn=<SoftmaxBackward0>)


# --------------------------------------------------------------
# 5️ Dropout işlemi
# --------------------------------------------------------------
# Dropout, aşırı öğrenmeyi (overfitting) önlemek için bazı ağırlıkları
# rastgele sıfırlar. Aktif kalan değerler 1/p oranında ölçeklenir.
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # %50 dropout
example = torch.ones(6, 6)
print(dropout(example))
# çıktı:
#  tensor([[2., 2., 0., 2., 2., 0.],
#         [0., 0., 0., 2., 0., 2.],
#         [2., 2., 2., 2., 0., 2.],
#         [0., 2., 2., 0., 0., 2.],
#         [0., 2., 0., 2., 0., 2.],
#         [0., 2., 2., 2., 2., 0.]])

torch.manual_seed(123)
print(dropout(attn_weights))
# çıktı:
#   tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
#         [0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]], grad_fn=<MulBackward0>)


# --------------------------------------------------------------
# 6️ Compact Causal Attention sınıfı testi (batch destekli)
# --------------------------------------------------------------
# Bu sınıf, dropout ve causal masklamayı tek bir yapıda birleştirir.
# Ayrıca batch boyutu (örneğin 2 farklı cümle) ile çalışabilir.
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 
# çıktı :
#   torch.Size([2, 6, 3])  2 => batch sayısı, 6 => token sayısı, 3 => embedding boyutu

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape: ", context_vecs.shape)
# çıktı:
#   context_vecs.shape:  torch.Size([2, 6, 2])
print(context_vecs)
# çıktı:
#   tensor([[[-0.4519,  0.2216],
#          [-0.5874,  0.0058],
#          [-0.6300, -0.0632],
#          [-0.5675, -0.0843],
#          [-0.5526, -0.0981],
#          [-0.5299, -0.1081]],

#         [[-0.4519,  0.2216],
#          [-0.5874,  0.0058],
#          [-0.6300, -0.0632],
#          [-0.5675, -0.0843],
#          [-0.5526, -0.0981],
#          [-0.5299, -0.1081]]], grad_fn=<UnsafeViewBackward0>)

