import re

# Metin dosyasını okuma
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

# --------------------------------------------------------------
# Tokenization örnekleri
# --------------------------------------------------------------
"""
re.split() ile metin boşluk ve noktalama işaretlerine göre bölünür.
item.strip() boşlukları temizler.
"""

# Örnek: re.split(r"([,.:;?_!\"()\']|--|\s)", text)
# Çıktı: ['Hello', ',', 'world', '.', 'This', '--', 'is', 'a', 'test', '?']

# Gerçek metni parçalama
preprocessed = re.split(r"([,.:;?_!\"()\']|--|\s)", raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

# --------------------------------------------------------------
# Benzersiz token’lar ve sözlük oluşturma
# --------------------------------------------------------------
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)  # 1130

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 5:
        break

# --------------------------------------------------------------
# Tokenizer V1 testi
# --------------------------------------------------------------
from simple_tokenizer import SimpleTokenizerV1

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

# --------------------------------------------------------------
# Özel tokenlar ekleme
# --------------------------------------------------------------
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))  # 1132

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# --------------------------------------------------------------
# Tokenizer V2 testi
# --------------------------------------------------------------
from simple_tokenizer import SimpleTokenizerV2

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace"
text = " <|endoftext|> ".join((text1, text2))

print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
