# --------------------------------------------------------------
# Transformer Bloğu 
# --------------------------------------------------------------
# Girdi → *LayerNorm1 → Masked Multi-Head Attention → Dropout → (+) Kısa yol bağlantısı
#       → LayerNorm2 → Feed Forward (Linear → GELU → Linear) → Dropout → (+) Kısa yol bağlantısı
# --------------------------------------------------------------
# Bu dosya, özellikle **Layer Normalization** kavramını örnekle açıklamaktadır.
# Transformer bloklarında her alt katmandan (attention / feedforward) önce LayerNorm uygulanır.
# --------------------------------------------------------------

import torch
import torch.nn as nn

# --------------------------------------------------------------
#  1. Basit örnek: Normalizasyon öncesi ve sonrası
# --------------------------------------------------------------
torch.manual_seed(123)

# (batch_size=2, features=5)
batch_example = torch.randn(2, 5)

# Basit bir linear + ReLU ağı
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# output:
# tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
#       grad_fn=<ReluBackward0>)

# --------------------------------------------------------------
#  2. Ortalama ve varyansın hesaplanması
# --------------------------------------------------------------
mean = out.mean(dim=-1, keepdim=True)  # her örnek (satır) için ortalama
var = out.var(dim=-1, keepdim=True)    # her örnek (satır) için varyans
print("Mean:\n", mean)
print("Variance:\n", var)

# output:
# Mean:
#  tensor([[0.1324],
#         [0.2170]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[0.0231],
#         [0.0398]], grad_fn=<VarBackward0>)

# --------------------------------------------------------------
#  3. Manuel Layer Normalization
# --------------------------------------------------------------
out_norm = (out - mean) / torch.sqrt(var)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

# output:
# Normalized layer outputs:
#  tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
#         [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
#        grad_fn=<DivBackward0>)
# Mean:
#  tensor([[0.0000],
#         [0.0000]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)

torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# --------------------------------------------------------------
#  4. LayerNorm sınıfının manuel implementasyonu
# --------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # sayısal kararlılık için küçük sabit
        self.scale = nn.Parameter(torch.ones(emb_dim))   # γ (öğrenilebilir çarpan)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # β (öğrenilebilir kaydırma)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift  # öğrenilebilir ölçek ve kaydırma uygulanır


# --------------------------------------------------------------
#  5. Uygulama ve kontrol
# --------------------------------------------------------------
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)

print("Mean:\n", mean)
print("Variance:\n", var)

# output:
# Mean:
#  tensor([[0.0000],
#         [0.0000]], grad_fn=<MeanBackward1>)
# Variance:
#  tensor([[1.0000],
#         [1.0000]], grad_fn=<VarBackward0>)

# --------------------------------------------------------------
# Sonuç:
# LayerNorm işlemi, her token vektörünü kendi içinde normalize eder.
# Böylece modelin eğitimi daha kararlı hale gelir.
# Transformer bloklarında bu işlem, hem attention hem de feed-forward katmanlarından önce uygulanır.
# --------------------------------------------------------------
