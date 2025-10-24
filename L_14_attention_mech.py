import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP hatası önleme

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------
# 1. Örnek token vektörleri (6 kelimelik basit bir cümle)
# --------------------------------------------------------------
inputs = torch.tensor(
  [[0.43, 0.15, 0.89],  # Your     (x¹)
   [0.55, 0.87, 0.66],  # journey  (x²)
   [0.57, 0.85, 0.64],  # starts   (x³)
   [0.22, 0.58, 0.33],  # with     (x⁴)
   [0.77, 0.25, 0.10],  # one      (x⁵)
   [0.05, 0.80, 0.55]]  # step     (x⁶)
)

words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

# --------------------------------------------------------------
# 2. "journey" kelimesi query seçilir (x²)
# --------------------------------------------------------------
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

# Her token için dot-product benzerlik hesaplama
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
# Örnek çıktı: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
# "journey" ve "starts" vektörleri birbirine yakın → yüksek skor (1.4754)

# --------------------------------------------------------------
# 3. Normalizasyon: skorları ağırlıklara dönüştürme
# --------------------------------------------------------------
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Toplam:", attn_weights_2_tmp.sum())
# Bu yöntem basit ama kararlı değildir. Softmax tercih edilir.

# --------------------------------------------------------------
# 4. Softmax ile normalizasyon
# --------------------------------------------------------------
def softmax_naive(x):
    """Naive softmax fonksiyonu: e^x / Σ(e^x)"""
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Softmax (manuel):", attn_weights_2_naive)
print("Toplam:", attn_weights_2_naive.sum())

# PyTorch’un dahili softmax’i (daha kararlı)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Softmax (PyTorch):", attn_weights_2)
print("Toplam:", attn_weights_2.sum())
# "journey" ve "starts" token'ları en yüksek dikkat ağırlığına sahip (~%23)

# --------------------------------------------------------------
# 5. Context vektörünün oluşturulması (2. token için)
# --------------------------------------------------------------
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("Context vektörü (x²):", context_vec_2)

# --------------------------------------------------------------
# 6. Tüm token’lar için attention matrisi (6x6)  
# --------------------------------------------------------------
attn_scores = inputs @ inputs.T  # dot product (hızlı yöntem)
print("Attention skor matrisi:\n", attn_scores)

# Softmax ile satır bazlı normalizasyon (her satırın toplamı = 1)
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Attention ağırlıkları:\n", attn_weights)

# --------------------------------------------------------------
# 7. Her token için context vektörlerinin hesaplanması
# --------------------------------------------------------------
all_context_vecs = attn_weights @ inputs
print("Context vektörleri (6x3):\n", all_context_vecs)


