
# --------------------------------------------------------------
# Transformer Bloğu 
# --------------------------------------------------------------
# Girdi → *LayerNorm1 → Masked Multi-Head Attention → Dropout → (+) Kısa yol bağlantısı
#       → LayerNorm2 → Feed Forward (Linear → GELU → Linear) → Dropout → (+) Kısa yol bağlantısı
# --------------------------------------------------------------
# Bu dosya, özellikle **Feed Forward (Linear → GELU → Linear)** kavramını örnekle açıklamaktadır.
# --------------------------------------------------------------
import torch.nn as nn
import torch

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Kelime sayısı (GPT-2'nin tokenizer'ı için)
    "context_length": 1024,  # Maksimum context uzunluğu
    "emb_dim": 768,          # Embedding boyutu
    "n_heads": 12,           # Attention head sayısı
    "n_layers": 12,          # Transformer block sayısı
    "drop_rate": 0.1,        # Dropout oranı
    "qkv_bias": False        # Query-Key-Value bias kullanımı
}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        )) 
       
# Feedforward modülü 2 lineer layer ve gelu aktivasyon fonksiyonu içeren küçük bir sinir ağı        
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # genişletme
            GELU(), # aktivasyon
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # daraltma
        )

    def forward(self, x):
        return self.layers(x)

print(GPT_CONFIG_124M["emb_dim"])
# output:
#     768

ffn=FeedForward(GPT_CONFIG_124M)
x=torch.rand(2,3,768)
out=ffn(x)
print(out.shape)
# output:
    # torch.Size([2, 3, 768])