from transformers import AutoTokenizer
from datasets import load_dataset
import os

# DailyDialog veri setini yükleyelim (İngilizce konuşma veri seti)
print("DailyDialog veri seti yükleniyor...")
dataset = load_dataset("daily_dialog", trust_remote_code=True)
print(f"Eğitim seti boyutu: {len(dataset['train'])}")
print(f"Doğrulama seti boyutu: {len(dataset['validation'])}")
print(f"Test seti boyutu: {len(dataset['test'])}")

# Veri örneklerini inceleyelim
print("\nÖrnek eğitim diyaloğu:")
print(dataset['train'][100]['dialog'])

# Tokenizer model oluşturalım - GPT2 tabanlı tokenizer kullanıyoruz
# B2 seviyesi İngilizce için uygun bir seçim
print("\nTokenizer oluşturuluyor...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Konuşma modeli için özel tokenler ekleme
special_tokens = {
    "pad_token": "[PAD]",
    "eos_token": "[EOS]",
    "bos_token": "[BOS]",
    "sep_token": "[SEP]",  # Konuşmacıları ayırmak için
    "additional_special_tokens": ["[USER]", "[ASSISTANT]"]  # Konuşmacı rolleri
}
tokenizer.add_special_tokens(special_tokens)

# Diyalogları tokenize etmek için fonksiyon
def tokenize_dialog(examples):
    # Her diyaloğu tek bir metin olarak birleştir
    conversations = []
    for dialog in examples["dialog"]:
        # Diyalog turlarını konuşmacı etiketleriyle birleştir
        formatted_dialog = ""
        for i, utterance in enumerate(dialog):
            speaker = "[USER]" if i % 2 == 0 else "[ASSISTANT]"
            formatted_dialog += f"{speaker} {utterance} {tokenizer.sep_token} "
        formatted_dialog += tokenizer.eos_token
        conversations.append(formatted_dialog)
    
    # Tokenize et
    tokenized = tokenizer(
        conversations,
        padding="max_length",
        truncation=True,
        max_length=512,  # Daha uzun konuşmalar için
        return_tensors="pt"
    )
    
    return tokenized

print("\nVeri seti tokenize ediliyor...")
tokenized_datasets = dataset.map(
    tokenize_dialog,
    batched=True,
    batch_size=16,
    remove_columns=dataset["train"].column_names
)

# Tokenize edilmiş veri setini kontrol edelim
print("\nTokenize edilmiş veri seti özellikleri:")
print(tokenized_datasets["train"].features)

# Bir örnek görelim
print("\nTokenize edilmiş bir örnek:")
print(tokenized_datasets["train"][0])

# Tokenizer'ı kaydedelim
save_path = "./saved_tokenizer"
if not os.path.exists(save_path):
    os.makedirs(save_path)

tokenizer.save_pretrained(save_path)
print(f"\nTokenizer başarıyla kaydedildi: {save_path}")

# Tokenize edilmiş veri setini de kaydedelim
tokenized_datasets.save_to_disk("./tokenized_dailydialog")
print("\nTokenize edilmiş veri seti başarıyla kaydedildi: ./tokenized_dailydialog")

print("\nB2 seviyesinde İngilizce konuşma modeli için tokenizer hazır!")