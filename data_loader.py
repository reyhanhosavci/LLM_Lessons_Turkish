from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch

# --------------------------------------------------------------
# GPT için veri kümesi sınıfı
# --------------------------------------------------------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        # Tüm metni tokenize et
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # Sliding window ile metni üst üste binen parçalara ayır
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            
            # Giriş ve hedef dizilerini tensora dönüştür
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# --------------------------------------------------------------
# DataLoader oluşturma fonksiyonu
# --------------------------------------------------------------
def create_dataloader_v1(
    txt, batch_size=4, max_length=256, stride=128,
    shuffle=True, drop_last=True, num_workers=0
):
    # Tokenizer başlatma (GPT-2 BPE)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Dataset oluşturma
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # PyTorch DataLoader oluşturma
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader
