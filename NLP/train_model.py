import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import argparse
import logging
import numpy as np
from tqdm import tqdm

# Argüman parser'ı oluştur
parser = argparse.ArgumentParser(description="B2 seviyesinde İngilizce konuşma modeli eğitimi")
parser.add_argument("--epochs", type=int, default=3, help="Eğitim epoch sayısı")
parser.add_argument("--batch_size", type=int, default=8, help="Batch boyutu")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Öğrenme oranı")
parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup adım sayısı")
parser.add_argument("--model_name", type=str, default="gpt2", help="Temel model adı")
parser.add_argument("--output_dir", type=str, default="./english_conversation_model", help="Model kayıt dizini")
parser.add_argument("--tokenizer_dir", type=str, default="./saved_tokenizer", help="Tokenizer dizini")
parser.add_argument("--dataset_dir", type=str, default="./tokenized_dailydialog", help="Tokenize edilmiş veri seti dizini")
args = parser.parse_args()

# Logging ayarları
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Eğitim cihazı: {device}")

# Tokenizer ve veri seti yükleme
logger.info(f"Tokenizer yükleniyor: {args.tokenizer_dir}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

logger.info(f"Tokenize edilmiş veri seti yükleniyor: {args.dataset_dir}")
tokenized_datasets = load_from_disk(args.dataset_dir)

# Model oluşturma
logger.info(f"Model oluşturuluyor: {args.model_name}")

# Önce modeli doğrudan yükle, sonra token boyutunu ayarla
model = GPT2LMHeadModel.from_pretrained(
    args.model_name,
    ignore_mismatched_sizes=True  # Boyut uyuşmazlıklarını görmezden gel
)

# Tokenizer boyutuna göre model token embedding'lerini yeniden boyutlandır
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Data collator oluşturma
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 için causal language modeling kullanıyoruz
)

# DataLoader oluşturma
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    batch_size=args.batch_size,
    collate_fn=data_collator,
    shuffle=True
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=args.batch_size,
    collate_fn=data_collator
)

# Optimizer ve scheduler oluşturma
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps
)

# Eğitim fonksiyonu
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Eğitim")
    
    for batch in progress_bar:
        # Batch'i cihaza taşı
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer ve scheduler adımları
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Loss güncelleme
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)

# Değerlendirme fonksiyonu
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Değerlendirme"):
            # Batch'i cihaza taşı
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Loss güncelleme
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Eğitim döngüsü
logger.info("Eğitim başlıyor...")
best_val_loss = float("inf")

for epoch in range(args.epochs):
    logger.info(f"Epoch {epoch + 1}/{args.epochs}")
    
    # Eğitim
    train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
    logger.info(f"Eğitim kaybı: {train_loss:.4f}")
    
    # Değerlendirme
    val_loss = evaluate(model, eval_dataloader, device)
    logger.info(f"Doğrulama kaybı: {val_loss:.4f}")
    
    # En iyi modeli kaydet
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.info(f"En iyi model kaydediliyor: {args.output_dir}")
        
        # Dizin oluştur
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Modeli kaydet
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # En iyi modeli ayrıca kaydet
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

logger.info(f"Eğitim tamamlandı! En iyi doğrulama kaybı: {best_val_loss:.4f}")

# Test veri seti üzerinde değerlendirme
logger.info("Test veri seti üzerinde değerlendirme yapılıyor...")
test_dataloader = DataLoader(
    tokenized_datasets["test"],
    batch_size=args.batch_size,
    collate_fn=data_collator
)
test_loss = evaluate(model, test_dataloader, device)
logger.info(f"Test kaybı: {test_loss:.4f}")

# Örnek cümle oluşturma
logger.info("Örnek cümleler oluşturuluyor...")
model.eval()

# Örnek konuşma başlangıçları
prompts = [
    "[USER] How was your weekend? [SEP] ",
    "[USER] Can you tell me about your favorite book? [SEP] ",
    "[USER] What do you think about climate change? [SEP] ",
    "[USER] I'm planning to visit London next month. Any recommendations? [SEP] "
]

for prompt in prompts:
    # Prompt'u tokenize et
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Cümle oluştur
    output = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Çıktıyı decode et
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Oluşturulan cevap: {generated_text}")
    logger.info("-" * 50)

logger.info("İşlem tamamlandı!") 