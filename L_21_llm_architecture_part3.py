# --------------------------------------------------------------
# Transformer Bloğu 
# --------------------------------------------------------------
# Girdi → LayerNorm1 → Masked Multi-Head Attention → Dropout → (+) Kısa yol bağlantısı
#       → LayerNorm2 → Feed Forward (Linear → GELU → Linear) → Dropout → (+) Kısa yol bağlantısı
# --------------------------------------------------------------
# Bu dosya, özellikle Feed Forward (Linear → GELU → Linear) yapısını örnekle açıklar.
# Bu yapı, Transformer bloklarında Attention katmanından sonra yer alır.
# Her token bağımsız olarak işlenir ve embedding boyutu genişletilip tekrar daraltılır.
# --------------------------------------------------------------

import torch
import torch.nn as nn

# GPT-2 Small (124M) yapılandırma parametreleri
GPT_CONFIG_124M = {
    "vocab_size": 50257,    
    "context_length": 1024,  
    "emb_dim": 768,          
    "n_heads": 12,           
    "n_layers": 12,          
    "drop_rate": 0.1,        
    "qkv_bias": False        
}

# GELU (Gaussian Error Linear Unit) aktivasyon fonksiyonu
# ReLU’ya göre daha yumuşak bir doğrusal olmayan fonksiyondur.
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        )) 

# Feed Forward Katmanı
# Transformer içindeki ikinci alt katmandır.
# 1. Linear (emb_dim → 4×emb_dim): Boyut genişletilir.
# 2. GELU: Aktivasyon uygulanır.
# 3. Linear (4×emb_dim → emb_dim): Boyut tekrar indirgenir.
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), 
            GELU(),                                      
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# Uygulama örneği
print(GPT_CONFIG_124M["emb_dim"])
# output:
#     768

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)
# output:
#     torch.Size([2, 3, 768])

# FeedForward katmanı, Attention katmanından gelen çıktıyı işler.
# Boyutu önce 4 katına çıkarır, sonra tekrar indirir.
# Bu işlem modelin daha karmaşık ilişkileri öğrenmesini sağlar.
# Her token diğerlerinden bağımsız olarak işlenir.
