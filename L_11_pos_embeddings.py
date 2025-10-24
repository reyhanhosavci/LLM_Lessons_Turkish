# Token embedding örneği (BPE ile)
import torch
from data_loader import create_dataloader_v1 

# Metni yükleme
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Token embedding parametreleri
vocab_size = 50257     # GPT-2 sözlük boyutu
output_dim = 256       # her token 256 boyutlu vektöre dönüştürülür

# Token embedding matrisi (vocab_size x output_dim)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# --------------------------------------------------------------
# Veri örnekleme (sliding window)
# --------------------------------------------------------------
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)   # [8, 4]

# --------------------------------------------------------------
# Token embedding oluşturma
# --------------------------------------------------------------
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)              # [8, 4, 256]

# --------------------------------------------------------------
# Konum (pozisyon) embedding katmanı
# --------------------------------------------------------------
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)                # [4, 256]

# --------------------------------------------------------------
# Token + pozisyon embeddinglerinin toplanması
# --------------------------------------------------------------
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)              # [8, 4, 256]
