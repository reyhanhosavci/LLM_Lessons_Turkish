import re  # "re" modülü, metinleri düzenli ifadelerle (regex) parçalamamıza yardımcı olur.

# 🔹 Bu sınıf, çok basit bir "tokenizer" örneği.
# Yani metni kelimelere veya sembollere ayırıyor (encode),
# sonra da o sayılardan tekrar metin oluşturabiliyor (decode).

class SimpleTokenizerV1:
    def __init__(self, vocab):
        # vocab: kelimelerle (ya da sembollerle) sayılar arasındaki eşleşmeyi içeriyor.
        self.str_to_int = vocab
        # ters eşleşme oluşturuluyor: sayıdan kelimeye ulaşmak için
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        # Metni, noktalama işaretlerine ve boşluklara göre parçalıyoruz.
        preprocessed = re.split(r"([,.:;?_!\"()\']|--|\s)", text)
        # Boşluklardan ve gereksiz boş stringlerden kurtuluyoruz.
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Her kelimeyi veya sembolü, vocab içindeki sayısal karşılığına çeviriyoruz.
        ids = [self.str_to_int[s] for s in preprocessed]
        
        # Sonuç olarak sayılardan oluşan bir liste dönüyor.
        return ids

    def decode(self, ids):
        # Sayı listesini tekrar kelimelere çeviriyoruz.
        text = " ".join([self.int_to_str[i] for i in ids])
        # Noktalama işaretlerinden önceki gereksiz boşlukları temizliyoruz.
        text = re.sub(r'\s+([,.:;?_!\"()\'])', r'\1', text)
        return text


# 🔹 Bu sınıf, ilkine çok benziyor.
# Tek farkı: sözlükte olmayan kelimeleri yakalayıp "<|unk|>" (bilinmeyen) etiketiyle değiştiriyor.
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r"([,.:;?_!\"()\']|--|\s)", text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Eğer bir kelime vocab içinde yoksa "<|unk|>" olarak işaretliyoruz.
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!\"()\'])', r'\1', text)
        return text
