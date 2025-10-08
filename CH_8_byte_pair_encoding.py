import importlib
import importlib.metadata
import tiktoken

# tiktoken sürümünü kontrol etme
# print("tiktoken version:", importlib.metadata.version("tiktoken"))  # 0.9.0

# GPT-2 tokenizer’ını yükleme
tokenizer = tiktoken.get_encoding("gpt2")

# Metin örnekleri
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknowPlace")
text2 = "Akwirw ier"  # modelin sözlüğünde olmayan kelimeler

# Metni token ID'lerine dönüştürme
integers = tokenizer.encode(text2, allowed_special={"<|endoftext|>"})
print(integers)
# Örnek: [33901, 86, 343, 86, 220, 959]

# ID’leri tekrar metne dönüştürme
strings = tokenizer.decode(integers)
print(strings)
# Çıktı: Akwirw ier
# Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknowPlace
# Akwirw ier
