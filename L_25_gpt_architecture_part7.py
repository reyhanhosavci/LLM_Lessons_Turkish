from L_24_gpt_architecture_part6 import GPTModel, GPT_CONFIG_124M
import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Basit bir metin uretim fonksiyonu (greedy decoding).
    Parametreler:
        model: GPT modeli
        idx: Token indekslerini iceren tensor (batch, num_tokens)
        max_new_tokens: Uretilecek token sayisi
        context_size: Modelin maksimum baglam uzunlugu
    Donus:
        Orijinal ve uretilecek tokenlari iceren tensor
    """
    for _ in range(max_new_tokens):
        # Son context_size kadar tokeni girdi olarak al
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            # Modelin cikisi: [batch, n_tokens, vocab_size]
            logits = model(idx_cond)

        # Her batch icin son tokenin logit degerini al
        logits = logits[:, -1, :]  # [batch, vocab_size]

        # Logitleri olasiliklara donustur
        probas = torch.softmax(logits, dim=-1)

        # En yuksek olasiliga sahip tokeni sec (greedy secim)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # [batch, 1]

        # Secilen tokeni mevcut dizinin sonuna ekle
        idx = torch.cat((idx, idx_next), dim=1)  # [batch, n_tokens + 1]

    return idx


if __name__ == "__main__":
    start_context = "Hello, I am"
    import tiktoken

    # Baslangic metnini tokenlara donustur
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    print("Encoded: ", encoded)

    # Tensor formatina cevir ve batch boyutu ekle
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("Encoded_tensor.shape", encoded_tensor.shape)
    # Encoded: [15496, 11, 314, 716]
    # Encoded_tensor.shape: torch.Size([1, 4])

    # Modeli olustur ve degerlendirme moduna al
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    # Metin uret
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output: ", out)
    print("Output length: ", len(out[0]))
    # Output: tensor([[15496, 11, 314, 716, 39450, 27424, 40337, 30272, 21898, 49132]])
    # Output length: 10

    # Tokenlari metne geri donustur
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
    # Ornek cikti: Hello, I am036 Enlight ventured Construction overthrowtur 
    # Her calistirmada farkli sonuc verebilir cunku model egitilmemistir
