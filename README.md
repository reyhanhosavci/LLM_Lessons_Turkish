# LLM_Lessons_Turkish
These implementations are based on the YouTube tutorial series LLM Lessons (https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu) and the source materials shared publicly by the original author on Google Drive.
# LLM Lessons (Türkçe Açıklamalı Kodlar)  
**LLM Lessons — Turkish Annotated Implementations**

---

## 🇹🇷 Türkçe Açıklama

Bu depo, [LLM Lessons YouTube serisi](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)  
videolarında anlatılan örnek kodların **Türkçe açıklamalı versiyonlarını** içermektedir.  

Kodlar, orijinal içerik üreticisinin Google Drive üzerinden **herkese açık olarak paylaştığı** eğitim materyallerine dayanmaktadır.  
Bu çalışma **öğrenme, öğretme ve araştırma amaçlıdır**; ticari bir kullanım veya dağıtım hedeflenmemektedir.  

Her dosyada, orijinal kod yapısına sadık kalınarak **Türkçe açıklamalar, yorum satırları ve açıklayıcı notlar** eklenmiştir.  
Amaç, büyük dil modelleri (LLM) konusunu Türkçe olarak daha anlaşılır hale getirmektir.

📚 İçerik Başlıkları — LLM Lessons Serisi
| Ders              | Dosya Adı                      | Başlık                                                 |
| ----------------- | ------------------------------ | ------------------------------------------------------ |
| **Lesson 7**      | `L_7_simple_tokenizer.py`      |  Basit Tokenizer Mantığı ve Uygulaması               |
| **Lesson 8**      | `L_8_byte_pair_encoding.py`    |  Byte Pair Encoding (BPE) Algoritması                |
| **Lesson 9**      | `L_9_dataloader.py`            |  DataLoader ile Veri Yükleme ve Batch İşleme         |
| **Lesson 9 (Ek)** | `L_9_input-output-pairs.py`    |  Input–Output Çiftlerinin Oluşturulması              |
| **Lesson 10**     | `L_10_token_embeddings.py`     |  Token Embedding Katmanı                             |
| **Lesson 11**     | `L_11_pos_embeddings.py`       |  Positional Embeddings (Pozisyon Bilgisi)            |
| **Lesson 14**     | `L_14_attention_mech.py`       |  Attention Mekanizmasının Temelleri                  |
| **Lesson 15**     | `L_15_self_attention.py`       |  Self-Attention (Kendine Dikkat) Yapısı              |
| **Lesson 16**     | `L_16_causal_attention.py`     |  Causal (Maskeli) Attention                          |
| **Lesson 17**     | `L_17_multi_head_attention.py` |  Multi-Head Attention (Çok Başlı Dikkat)             |
| **Lesson 19**     | `L_19_llm_architecture.py`     |  LLM (Large Language Model) Mimarisi Genel Yapısı   |
| **Ek Modül**      | `self_attention.py`            |  Self-Attention Modül Tanımı                         |
| **Ek Modül**      | `multi_head_attention.py`      |  Multi-Head Attention Modül Tanımı                   |
| **Ek Modül**      | `data_loader.py`               |  Dataset ve DataLoader Yardımcı Fonksiyonları        |
| **Uygulama**      | `DummyGPTModel.py`             |  GPT Mimarisinin Basitleştirilmiş (Dummy) Uygulaması |
| **Veri**          | `the-verdict.txt`              |  Eğitim/Test İçin Kullanılan Örnek Metin             |


### ⚠️ Telif ve Lisans Notu
Bu proje, orijinal video serisinin sahibi tarafından paylaşılan içeriklere dayanmaktadır.  
Orijinal içerik sahibine ait tüm haklar saklıdır.  
Benzer çalışmalar yapmak isteyenler için kaynak belirtilmesi önerilir.

---

## 🇬🇧 English Description

This repository contains **Turkish-annotated implementations** of the examples from the  
[LLM Lessons YouTube series](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu).

The codes are based on educational materials **publicly shared via Google Drive** by the original author.  
This project is intended **solely for learning, teaching, and research purposes** —  
no commercial redistribution or monetization is intended.  

Each file includes **Turkish commentary, code explanations, and docstrings**  
to make Large Language Model (LLM) concepts more accessible to Turkish learners.

📚 Contents — LLM Lessons Serie
| Lesson               | File Name                      | Topic                                                                     |
| -------------------- | ------------------------------ | ------------------------------------------------------------------------- |
| **Lesson 7**         | `L_7_simple_tokenizer.py`      |  Building a Simple Tokenizer                                            |
| **Lesson 8**         | `L_8_byte_pair_encoding.py`    |  Byte Pair Encoding (BPE) Algorithm                                     |
| **Lesson 9**         | `L_9_dataloader.py`            |  Creating a Custom DataLoader and Batching                              |
| **Lesson 9 (Extra)** | `L_9_input-output-pairs.py`    |  Preparing Input–Output Pairs                                           |
| **Lesson 10**        | `L_10_token_embeddings.py`     |  Token Embedding Layer                                                  |
| **Lesson 11**        | `L_11_pos_embeddings.py`       |  Positional Embeddings (Adding Order Information)                       |
| **Lesson 14**        | `L_14_attention_mech.py`       |  Fundamentals of the Attention Mechanism                                |
| **Lesson 15**        | `L_15_self_attention.py`       |  Self-Attention (Understanding Context Within a Sequence)               |
| **Lesson 16**        | `L_16_causal_attention.py`     |  Causal (Masked) Attention — Preventing Information Leakage             |
| **Lesson 17**        | `L_17_multi_head_attention.py` |  Multi-Head Attention Explained                                         |
| **Lesson 19**        | `L_19_llm_architecture.py`     |  LLM Architecture Overview — Assembling Model Components               |
| **Module**           | `self_attention.py`            |  Self-Attention Class Implementation                                    |
| **Module**           | `multi_head_attention.py`      |  Multi-Head Attention Class Implementation                              |
| **Module**           | `data_loader.py`               |  Data Preparation Utilities (Dataset & DataLoader)                      |
| **Example**          | `DummyGPTModel.py`             |  Simplified GPT Model (Token + Positional Embedding + Transformer Flow) |
| **Data**             | `the-verdict.txt`              |  Sample Text File Used for Training and Testing                         |


### ⚖️ License and Credits
All rights for the original materials belong to their respective creator(s).  
This repository exists for educational demonstration only.  
Please cite or reference the original YouTube playlist when reproducing or sharing.

---

### 👩‍💻 Author / Katkıda Bulunan
**Reyhan Hoşavcı**  
FSMVÜ — Computer Engineering PhD Student  
AI Researcher | NLP & Vision-Language Models  
🔗 [GitHub Profile](https://github.com/reyhanhosavci)

---

### 🪶 License
MIT License (for the annotations and added comments only).  
Original code may be subject to separate terms by its author.
