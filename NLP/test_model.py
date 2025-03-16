import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer
import time
import sys

# Modül yolunu ekle
sys.path.append('.')

# eng.py'den model sınıfını içe aktar
from eng import GPTLanguageModel

# Pozisyon kodlaması için sınıf
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Temel GPT benzeri dil modeli
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(GPTLanguageModel, self).__init__()
        self.d_model = d_model
        
        # Token ve pozisyon embeddinglari
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Basitleştirilmiş model - Transformer yerine LSTM kullanıyoruz
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Çıkış katmanı
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Parametreleri başlat
        self._init_weights()
    
    def _init_weights(self):
        # Embedding ve çıkış katmanlarını başlatalım
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, src, attention_mask=None):
        # src boyutu: [batch_size, seq_len]
        
        # Embedding ve pozisyon kodlaması
        src = self.token_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # LSTM katmanı
        output, _ = self.lstm(src)
        
        # Çıkış katmanı
        output = self.output_layer(output)
        
        return output

# GPU kullanılabilirliğini kontrol et
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cihaz: {device}")

# Tokenizer'ı yükle
print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
print(f"Tokenizer kelime dağarcığı boyutu: {len(tokenizer)}")

# Model parametrelerini ayarla
use_smaller_model = False
if use_smaller_model:
    print("Küçük model kullanılıyor...")
    model = GPTLanguageModel(
        vocab_size=len(tokenizer),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512
    )
else:
    print("Tam boyutlu model kullanılıyor...")
    model = GPTLanguageModel(
        vocab_size=len(tokenizer),
        d_model=384,
        nhead=12,
        num_layers=6,
        dim_feedforward=1536
    )

# Modeli yükle
print("Model yükleniyor...")
model.load_state_dict(torch.load("best_english_language_model.pt", map_location=device))
model = model.to(device)
model.eval()

# Metin üretme fonksiyonu
def generate_text(prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
    print(f"\nTest: '{prompt}'")
    print(f"  Parametreler: max_length={max_length}, temperature={temperature}, top_k={top_k}, top_p={top_p}, device={device}")
    
    # Başlangıç metnini kodla
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]  # Başlangıç metninin uzunluğunu kaydet
    
    # Attention mask oluştur
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Manuel metin üretme döngüsü
    with torch.no_grad():
        for i in range(max_length):
            # Model çıktısını al
            outputs = model(input_ids, attention_mask)
            
            # Son token için tahminlere bak
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Son tokeni koru
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., :1] = 0  # En yüksek olasılıklı tokeni koru
                
                # Orijinal indeksleri kullanarak maskeleme
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Token olasılıklarını hesapla
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sonraki tokeni örnekle
            next_token = torch.multinomial(probs, num_samples=1)
            
            # EOS tokeni gelirse dur
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Yeni tokeni ekle
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token).to(device)], dim=-1)
    
    # Sonucu çöz - sadece başlangıç metninden sonraki kısmı göster
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    original_text = tokenizer.decode(input_ids[0, :input_length], skip_special_tokens=True)
    generated_text = tokenizer.decode(input_ids[0, input_length:], skip_special_tokens=True)
    
    print(f"  Girdi metni: {original_text}")
    print(f"  Üretilen ek metin: {generated_text}")
    print(f"  Tam metin: {full_text}")
    print(f"  Toplam token sayısı: {len(input_ids[0])}")
    
    return full_text

# Test metinleri
test_texts = [
    "The quick brown fox",
    "In the beginning",
    "Once upon a time",
    "Artificial intelligence",
    "The meaning of life",
    "Bir varmış bir yokmuş",
    "Merhaba dünya",
    "Yapay zeka",
    "Türkiye'nin başkenti"
]

# Her test metni için metin üret
print("\nMETİN ÜRETME TESTLERİ BAŞLIYOR...")
print("="*50)

for test_text in test_texts:
    generate_text(test_text, max_length=100, temperature=1.0, top_k=50, top_p=0.95)

print("\nTüm testler tamamlandı!") 