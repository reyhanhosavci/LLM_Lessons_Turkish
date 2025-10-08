# Metin dosyasını okuma
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# --------------------------------------------------------------
# DataLoader oluşturma ve ilk batch'i inceleme
# --------------------------------------------------------------
import torch
from data_loader import create_dataloader_v1

print("PyTorch version:", torch.__version__)

# DataLoader ayarları:
# batch_size=1 → her seferinde 1 örnek
# max_length=4 → her örnek 4 token uzunluğunda
# stride=1 → her adımda 1 token kayarak ilerler (overlap’li)
# stride=4 → overlap olmadan ilerler
# shuffle=False → sıralı veri
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

# DataLoader’ı Python iterator’ına çevirme
data_iter = iter(dataloader)

# İlk batch’i alma
first_batch = next(data_iter)
print(first_batch[0][0])
# Örnek: tensor([[  40,  367, 2885, 1464]])

# --------------------------------------------------------------
# Token ID’leri çözümleme (decode)
# --------------------------------------------------------------
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# İlk batch’in hedef (output) kısmını çözümleme
print(tokenizer.decode(first_batch[1][0].tolist()))
