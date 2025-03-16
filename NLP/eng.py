import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import time
import sys

# Çıktı tamponlamasını devre dışı bırak
sys.stdout.reconfigure(line_buffering=True)

# GPU bellek kullanımını izleme fonksiyonu
def print_gpu_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Bellek Kullanımı:")
        print(f"  Ayrılan: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"  Önbellek: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
        print(f"  Maksimum Ayrılan: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
        # Belleği temizle
        torch.cuda.empty_cache()
    else:
        print("GPU mevcut değil, bellek kullanımı izlenemiyor.")

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

# Verileri modele uygun formata getirme
class CausalLanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __getitem__(self, idx):
        # Dataset nesnesi için doğrudan erişim kullanıyoruz
        input_ids = torch.tensor(self.encodings[idx]['input_ids'])
        attention_mask = torch.tensor(self.encodings[idx]['attention_mask']) if 'attention_mask' in self.encodings.column_names else None
        
        # Giriş ve çıkış için aynı uzunlukta diziler oluşturalım
        # Giriş: tüm tokenlar son token hariç
        input_ids_x = input_ids[:-1]
        # Çıkış: tüm tokenlar ilk token hariç
        labels = input_ids[1:]
        
        # Attention mask'i de aynı şekilde kısaltalım
        if attention_mask is not None:
            attention_mask = attention_mask[:-1].bool()
        
        return {
            "input_ids": input_ids_x,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def __len__(self):
        return len(self.encodings)

# Modeli eğitme fonksiyonu
def train_language_model(model, train_dataset, val_dataset=None, 
                        batch_size=16, epochs=3, learning_rate=5e-5, 
                        device="cuda" if torch.cuda.is_available() else "cpu"):
    
    print(f"\n{'='*50}")
    print(f"Eğitim başlıyor...")
    print(f"Cihaz: {device}")
    print(f"Batch boyutu: {batch_size}")
    print(f"Epoch sayısı: {epochs}")
    print(f"Öğrenme oranı: {learning_rate}")
    print(f"Eğitim veri seti boyutu: {len(train_dataset)}")
    if val_dataset:
        print(f"Doğrulama veri seti boyutu: {len(val_dataset)}")
    print(f"{'='*50}\n")
    
    # Toplam eğitim süresini hesapla
    total_batches = len(train_dataset) // batch_size * epochs
    print(f"Toplam batch sayısı: {total_batches}")
    
    # GPU bellek kullanımını kontrol et
    if device == "cuda":
        print("Başlangıç GPU bellek durumu:")
        print_gpu_memory_usage()
    
    # Veri yükleyicilerini oluştur
    print("Veri yükleyicileri oluşturuluyor...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    
    # Model ve optimizasyonu ayarla
    print("Model ve optimizasyon ayarlanıyor...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Öğrenme oranı zamanlayıcısı ekle
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches)
    
    criterion = nn.CrossEntropyLoss()
    
    # Eğitim istatistiklerini takip et
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Eğitim döngüsü
    for epoch in range(epochs):
        print(f"\n{'-'*50}")
        print(f"Epoch {epoch+1}/{epochs} başlıyor...")
        model.train()
        total_loss = 0
        batch_count = len(train_loader)
        
        # Epoch başlangıç zamanı
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # İlerleme durumunu göster
            if batch_idx % 10 == 0 or batch_idx == batch_count - 1:
                print(f"Batch {batch_idx+1}/{batch_count} işleniyor... ({(batch_idx+1)/batch_count*100:.1f}%)", end="\r")
            
            # Batchi cihaza taşı
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Attention mask'i kontrol et ve gerekirse oluştur
            if batch["attention_mask"] is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
            else:
                attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # İlk batch için boyutları yazdır
            if epoch == 0 and batch_idx == 0:
                print(f"\nİlk batch boyutları:")
                print(f"Giriş (input_ids) boyutu: {input_ids.shape}")
                print(f"Çıkış (outputs) boyutu: {outputs.shape}")
                print(f"Etiket (labels) boyutu: {labels.shape}")
                
                # GPU kullanılıyorsa bellek durumunu göster
                if device == "cuda":
                    print("\nİlk forward pass sonrası GPU bellek durumu:")
                    print_gpu_memory_usage()
            
            # Boyutların eşleştiğinden emin olalım
            if outputs.size(1) != labels.size(1):
                # Boyutları eşleştirelim (daha kısa olanı kullan)
                min_len = min(outputs.size(1), labels.size(1))
                outputs = outputs[:, :min_len, :]
                labels = labels[:, :min_len]
                
                # Boyut uyumsuzluğunu bildir
                if epoch == 0 and batch_idx == 0:
                    print(f"\nBoyut uyumsuzluğu tespit edildi ve düzeltildi:")
                    print(f"Yeni çıkış boyutu: {outputs.shape}")
                    print(f"Yeni etiket boyutu: {labels.shape}")
            
            # Loss hesaplama
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping ekle
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Öğrenme oranını güncelle
            
            total_loss += loss.item()
            
            # Her 100 batch'te bir GPU bellek durumunu göster (GPU kullanılıyorsa)
            if device == "cuda" and batch_idx > 0 and batch_idx % 100 == 0:
                print(f"\nBatch {batch_idx} sonrası GPU bellek durumu:")
                print_gpu_memory_usage()
                
            # Her 500 batch'te bir ara doğrulama yap
            if val_loader and batch_idx > 0 and batch_idx % 500 == 0:
                print(f"\nBatch {batch_idx} sonrası ara doğrulama yapılıyor...")
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        
                        if val_batch["attention_mask"] is None:
                            val_attention_mask = torch.ones_like(val_input_ids, dtype=torch.bool, device=device)
                        else:
                            val_attention_mask = val_batch["attention_mask"].to(device)
                        
                        val_outputs = model(val_input_ids, val_attention_mask)
                        
                        # Boyutların eşleştiğinden emin olalım
                        if val_outputs.size(1) != val_labels.size(1):
                            min_len = min(val_outputs.size(1), val_labels.size(1))
                            val_outputs = val_outputs[:, :min_len, :]
                            val_labels = val_labels[:, :min_len]
                        
                        val_loss += criterion(val_outputs.reshape(-1, val_outputs.shape[-1]), val_labels.reshape(-1)).item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f"Ara doğrulama kaybı: {avg_val_loss:.4f}")
                
                # Eğitime devam et
                model.train()
        
        # Epoch sonuçlarını göster
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Epoch bitiş zamanı ve süre hesaplama
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{epochs} tamamlandı, Süre: {epoch_duration:.2f} saniye")
        print(f"Ortalama Eğitim Kaybı: {avg_train_loss:.4f}")
        
        # Güncel öğrenme oranını göster
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Güncel öğrenme oranı: {current_lr:.6f}")
        
        # GPU bellek durumunu göster
        if device == "cuda":
            print(f"Epoch {epoch+1} sonrası GPU bellek durumu:")
            print_gpu_memory_usage()
        
        # Doğrulama
        if val_loader:
            print("Doğrulama yapılıyor...")
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    if batch["attention_mask"] is None:
                        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
                    else:
                        attention_mask = batch["attention_mask"].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    
                    # Boyutların eşleştiğinden emin olalım
                    if outputs.size(1) != labels.size(1):
                        min_len = min(outputs.size(1), labels.size(1))
                        outputs = outputs[:, :min_len, :]
                        labels = labels[:, :min_len]
                    
                    loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Doğrulama Kaybı: {avg_val_loss:.4f}")
            
            # En iyi modeli kaydet
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"Yeni en iyi model! (Doğrulama kaybı: {best_val_loss:.4f})")
                torch.save(model.state_dict(), "best_english_language_model.pt")
                print("En iyi model kaydedildi: best_english_language_model.pt")
    
    print(f"\n{'='*50}")
    print("Eğitim tamamlandı!")
    
    # Eğitim istatistiklerini göster
    print("\nEğitim istatistikleri:")
    print(f"Başlangıç eğitim kaybı: {train_losses[0]:.4f}")
    print(f"Son eğitim kaybı: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Başlangıç doğrulama kaybı: {val_losses[0]:.4f}")
        print(f"Son doğrulama kaybı: {val_losses[-1]:.4f}")
        print(f"En iyi doğrulama kaybı: {best_val_loss:.4f}")
    
    # Son GPU bellek durumunu göster
    if device == "cuda":
        print("Eğitim sonrası GPU bellek durumu:")
        print_gpu_memory_usage()
    
    print(f"{'='*50}")
    
    return model

# Metin üretme fonksiyonu
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.95, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"   Metin üretme parametreleri:")
    print(f"   - Maksimum uzunluk: {max_length} token")
    print(f"   - Sıcaklık: {temperature}")
    print(f"   - Top-k: {top_k}")
    print(f"   - Top-p: {top_p}")
    print(f"   - Cihaz: {device}")
    
    model.eval()
    
    # Başlangıç metnini kodla
    print(f"   Başlangıç metni kodlanıyor: '{prompt}'")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Attention mask oluştur
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Generation parametreleri
    gen_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature
    }
    
    # Manuel metin üretme döngüsü
    print(f"   Metin üretiliyor...")
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
                print(f"   EOS token'ı üretildi, üretim durduruluyor.")
                break
            
            # İlerleme göster
            if i % 10 == 0:
                print(f"   {i}/{max_length} token üretildi...", end="\r")
            
            # Yeni tokeni ekle
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token).to(device)], dim=-1)
    
    # Sonucu çöz
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"   Metin üretimi tamamlandı! ({len(input_ids[0])} token)")
    return generated_text

# Örnek kullanım için ana kod
def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    print("\n" + "="*70)
    print("İNGİLİZCE DİL MODELİ EĞİTİMİ BAŞLIYOR")
    print("="*70)
    
    # GPU kullanılabilirliğini kontrol et
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\nGPU KULLANIMI ETKİN!")
        print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        print(f"\nGPU BULUNAMADI! CPU kullanılacak.")
        print("Eğitim daha yavaş olacak. GPU kullanmak için CUDA yüklü bir GPU'ya sahip olduğunuzdan emin olun.")
    
    # Başlangıç zamanını kaydet
    start_time = time.time()
    
    # Veri seti yükleme
    print("\n1. Veri seti yükleniyor...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Eğitim sürecini hızlandırmak için veri setinin daha küçük bir kısmını kullanabiliriz
    # Bu seçenek, eğitim süresini önemli ölçüde azaltır
    use_smaller_dataset = False  # Tüm veri setini kullanmak için False yapıldı
    if use_smaller_dataset:
        print("   Daha hızlı eğitim için küçük veri seti kullanılıyor...")
        # Eğitim setinin %10'unu kullan
        train_size = len(dataset['train']) // 10
        val_size = len(dataset['validation']) // 10
        
        # Veri setlerini küçült
        dataset['train'] = dataset['train'].select(range(train_size))
        dataset['validation'] = dataset['validation'].select(range(val_size))
        
        print(f"   Küçültülmüş eğitim seti boyutu: {len(dataset['train'])} örnek")
        print(f"   Küçültülmüş doğrulama seti boyutu: {len(dataset['validation'])} örnek")
    else:
        print(f"   Tam eğitim seti kullanılıyor!")
        print(f"   Eğitim seti boyutu: {len(dataset['train'])} örnek")
        print(f"   Doğrulama seti boyutu: {len(dataset['validation'])} örnek")
    
    print(f"   Test seti boyutu: {len(dataset['test'])} örnek")
    
    # Tokenizer yükleme
    print("\n2. Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Özel token ekleme
    print("   Özel tokenler ekleniyor...")
    special_tokens = {"pad_token": "[PAD]", "eos_token": "[EOS]"}
    tokenizer.add_special_tokens(special_tokens)
    print(f"   Tokenizer kelime dağarcığı boyutu: {len(tokenizer)} token")
    
    # Veri setini tokenize etme
    print("\n3. Veri seti tokenize ediliyor...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
    
    print("   Tokenizasyon işlemi başladı...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("   Tokenizasyon işlemi tamamlandı!")
    
    # Causal LM veri setini oluştur
    print("\n4. Dil modeli veri seti oluşturuluyor...")
    train_dataset = CausalLanguageModelingDataset(tokenized_datasets["train"])
    val_dataset = CausalLanguageModelingDataset(tokenized_datasets["validation"])
    print("   Veri setleri hazır!")
    
    # Model oluşturma
    print("\n5. Model oluşturuluyor...")
    
    # Eğitim sürecini hızlandırmak için daha küçük bir model kullanabiliriz
    use_smaller_model = False  # Daha büyük model kullanmak için False yapıldı
    if use_smaller_model:
        print("   Daha hızlı eğitim için küçük model kullanılıyor...")
        model = GPTLanguageModel(
            vocab_size=len(tokenizer),
            d_model=128,  # Daha küçük gizli katman boyutu
            nhead=4,      # Daha az attention head
            num_layers=2, # Daha az katman
            dim_feedforward=512  # Daha küçük feedforward boyutu
        )
        print("   Model parametreleri (küçültülmüş):")
        print(f"   - Kelime dağarcığı boyutu: {len(tokenizer)}")
        print(f"   - Gizli katman boyutu: 128")
        print(f"   - Attention head sayısı: 4")
        print(f"   - Katman sayısı: 2")
        print(f"   - Feedforward boyutu: 512")
    else:
        print("   Tam boyutlu model kullanılıyor...")
        model = GPTLanguageModel(
            vocab_size=len(tokenizer),
            d_model=384,  # Daha büyük gizli katman boyutu
            nhead=12,     # Daha fazla attention head
            num_layers=6, # Daha fazla katman
            dim_feedforward=1536  # Daha büyük feedforward boyutu
        )
        print("   Model parametreleri (tam boyut):")
        print(f"   - Kelime dağarcığı boyutu: {len(tokenizer)}")
        print(f"   - Gizli katman boyutu: 384")
        print(f"   - Attention head sayısı: 12")
        print(f"   - Katman sayısı: 6")
        print(f"   - Feedforward boyutu: 1536")
    
    # Modeli eğit
    print("\n6. Model eğitimi başlıyor...")
    train_language_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,  # Daha büyük batch boyutu
        epochs=5,       # Daha fazla epoch
        learning_rate=3e-5,  # Daha düşük öğrenme oranı
        device=device  # Burada device parametresini geçiyoruz
    )
    
    # Modeli ve tokenizer'ı kaydet
    print("\n7. Model ve tokenizer kaydediliyor...")
    torch.save(model.state_dict(), "english_language_model.pt")
    tokenizer.save_pretrained("tokenizer")
    print("   Model kaydedildi: english_language_model.pt")
    print("   Tokenizer kaydedildi: tokenizer/")
    
    # Metin üretme
    print("\n8. Örnek metin üretiliyor...")
    generated = generate_text(model, tokenizer, "The quick brown fox", max_length=50)
    print(f"   Girdi: 'The quick brown fox'")
    print(f"   Üretilen metin: {generated}")
    
    # Bitiş zamanını hesapla
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*70)
    print(f"İŞLEM TAMAMLANDI! Toplam süre: {int(hours)}s {int(minutes)}d {int(seconds)}sn")
    print("="*70)

if __name__ == "__main__":
    main()