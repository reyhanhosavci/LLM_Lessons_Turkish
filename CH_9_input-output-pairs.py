import tiktoken

# GPT-2 BPE tokenizer yükleme
tokenizer = tiktoken.get_encoding("gpt2")

# --------------------------------------------------------------
# Metni okuma ve tokenize etme
# --------------------------------------------------------------
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Metni BPE tokenlarına dönüştürme
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))  
# 5145 → metindeki toplam token sayısı (vocab büyüklüğü değil)

# --------------------------------------------------------------
# Basit input–output çiftleri oluşturma
# --------------------------------------------------------------
enc_sample = enc_text[50:]     # örnek bölüm
context_size = 4               # her giriş 4 token içerir

# Giriş (x) ve hedef (y) dizileri
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"x: {x}")
print(f"y: {y}")

# --------------------------------------------------------------
# Her adım için hedef token tahmini gösterimi
# --------------------------------------------------------------
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# Örnek çıktı:
# [290] -----> 4920
# [290, 4920] -----> 2241
# [290, 4920, 2241] -----> 287
# [290, 4920, 2241, 287] -----> 257
