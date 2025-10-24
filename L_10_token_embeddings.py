""" 
import gensim.downloader as api  # pyright: ignore[reportMissingImports] 
model = api.load("word2vec-google-news-300")
word_vector = model
"""

""" 
# örnek: 'computer' kelimesine ait vektör
print(word_vector['computer'])  

print(word_vector['cat'].shape)  # vektör boyutu (300)
# benzerlik örneği: vektör aritmetiği
print(word_vector.most_similar(positive=['king', 'woman'], negative=['man']))

# kelimeler arası benzerlik ölçümü
print(word_vector.similarity('woman', 'man'))
print(word_vector.similarity('king', 'queen'))
print(word_vector.similarity('boy', 'girl'))

import numpy as np

word1 = 'man'
word2 = 'woman'
word3 = 'semiconductor'
word4 = 'earthworm'

# iki kelime arasındaki vektör farkını hesaplama
vector_difference1 = model[word1] - model[word2]
vector_difference2 = model[word3] - model[word4]

# fark vektörlerinin büyüklüğünü hesaplama
magnitude_of_difference1 = np.linalg.norm(vector_difference1)
magnitude_of_difference2 = np.linalg.norm(vector_difference2)

# sonuçları yazdırma
print("The magnitude of the difference between '{}' and '{}' is {:.2f}"
      .format(word1, word2, magnitude_of_difference1))
"""

# --------------------------------------------------------------
# Token embedding örneği
# --------------------------------------------------------------
import torch

# 4 token ID'den oluşan örnek giriş
input_ids = torch.tensor([2, 3, 5, 1])

# küçük bir örnek sözlük boyutu ve çıktı boyutu
vocab_size = 6        # (örnek) GPT-3'te bu 50,257 olur
output_dim = 3        # her token 3 boyutlu vektöre çevrilir

torch.manual_seed(123)  # tekrarlanabilir sonuçlar için sabit seed

# Embedding katmanını tanımlama
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# embedding ağırlıklarını görüntüleme
print(embedding_layer.weight)

# belirli bir token ID için embedding vektörü
print(embedding_layer(torch.tensor([3])))

# tüm input dizisi için embedding çıktısı
print(embedding_layer(input_ids))
