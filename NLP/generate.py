import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
from transformers import AutoTokenizer

# Çıktı tamponlamasını devre dışı bırak
sys.stdout.reconfigure(line_buffering=True)

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

# Metin üretme fonksiyonu
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.95, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Metin üretme parametreleri:")
    print(f"- Maksimum uzunluk: {max_length} token")
    print(f"- Sıcaklık: {temperature}")
    print(f"- Top-k: {top_k}")
    print(f"- Top-p: {top_p}")
    print(f"- Cihaz: {device}")
    
    model.eval()
    
    # Başlangıç metnini kodla
    print(f"Başlangıç metni kodlanıyor: '{prompt}'")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Attention mask oluştur
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Manuel metin üretme döngüsü
    print(f"Metin üretiliyor...")
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
                print(f"EOS token'ı üretildi, üretim durduruluyor.")
                break
            
            # İlerleme göster
            if i % 10 == 0:
                print(f"{i}/{max_length} token üretildi...", end="\r")
            
            # Yeni tokeni ekle
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token).to(device)], dim=-1)
    
    # Sonucu çöz
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Metin üretimi tamamlandı! ({len(input_ids[0])} token)")
    return generated_text

def main():
    print("\n" + "="*70)
    print("KAYDEDILMIŞ MODEL İLE METİN ÜRETİMİ")
    print("="*70)
    
    # GPU kullanılabilirliğini kontrol et
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\nGPU KULLANIMI ETKİN!")
        print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\nGPU BULUNAMADI! CPU kullanılacak.")
    
    # Tokenizer'ı yükle
    print("\n1. Tokenizer yükleniyor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
        print("   Tokenizer başarıyla yüklendi!")
    except:
        print("   Kaydedilmiş tokenizer bulunamadı! GPT-2 tokenizer kullanılacak.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Özel token ekleme
        special_tokens = {"pad_token": "[PAD]", "eos_token": "[EOS]"}
        tokenizer.add_special_tokens(special_tokens)
    
    print(f"   Tokenizer kelime dağarcığı boyutu: {len(tokenizer)} token")
    
    # Modeli yükle
    print("\n2. Model yükleniyor...")
    try:
        # Önce model boyutunu kontrol et
        use_small_model = True  # Eğer küçük model kullandıysanız True yapın
        
        if use_small_model:
            # Küçük model parametreleri
            model = GPTLanguageModel(
                vocab_size=len(tokenizer),
                d_model=128,
                nhead=4,
                num_layers=2,
                dim_feedforward=512
            )
            print("   Küçük model yapılandırması kullanılıyor...")
        else:
            # Normal model parametreleri
            model = GPTLanguageModel(
                vocab_size=len(tokenizer),
                d_model=256,
                nhead=8,
                num_layers=6,
                dim_feedforward=1024
            )
            print("   Normal model yapılandırması kullanılıyor...")
        
        # Kaydedilmiş model ağırlıklarını yükle
        model.load_state_dict(torch.load("english_language_model.pt", map_location=device))
        model = model.to(device)
        print("   Model başarıyla yüklendi!")
    except Exception as e:
        print(f"   Hata: {e}")
        print("   Kaydedilmiş model yüklenemedi!")
        sys.exit(1)
    
    # Metin üretimi
    while True:
        print("\n" + "-"*70)
        prompt = input("Başlangıç metni girin (çıkmak için 'q'): ")
        
        if prompt.lower() == 'q':
            break
        
        if not prompt:
            prompt = "The quick brown fox"
            print(f"Boş giriş! Varsayılan metin kullanılıyor: '{prompt}'")
        
        # Metin üretme parametreleri
        max_length = int(input(f"Maksimum uzunluk (varsayılan: 50): ") or "50")
        temperature = float(input(f"Sıcaklık (0.1-2.0, varsayılan: 1.0): ") or "1.0")
        
        # Metin üret
        start_time = time.time()
        generated = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        
        # Sonuçları göster
        print("\nÜretilen metin:")
        print("-" * 40)
        print(generated)
        print("-" * 40)
        
        # Süreyi göster
        end_time = time.time()
        print(f"Üretim süresi: {end_time - start_time:.2f} saniye")
    
    print("\n" + "="*70)
    print("PROGRAM SONLANDI")
    print("="*70)

if __name__ == "__main__":
    main() 