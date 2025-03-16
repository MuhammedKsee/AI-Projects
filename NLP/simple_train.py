import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")
print(f"Tokenizer yüklendi. Kelime dağarcığı boyutu: {len(tokenizer)}")

print("\nVeri seti yükleniyor...")
try:
    tokenized_datasets = load_from_disk("./tokenized_dailydialog")
    print(f"Veri seti yüklendi. Eğitim seti boyutu: {len(tokenized_datasets['train'])}")
except Exception as e:
    print(f"Veri seti yüklenirken hata oluştu: {e}")
    exit(1)

print("\nModel oluşturuluyor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim cihazı: {device}")

try:
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print("Model başarıyla oluşturuldu.")
except Exception as e:
    print(f"Model oluşturulurken hata oluştu: {e}")
    exit(1)

print("\nData collator oluşturuluyor...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

print("\nDataLoader oluşturuluyor...")
batch_size = 2  # Çok küçük bir batch size
train_dataloader = DataLoader(
    tokenized_datasets["train"].select(range(10)),  # Sadece ilk 10 örnek
    batch_size=batch_size,
    collate_fn=data_collator,
    shuffle=True
)
print(f"DataLoader oluşturuldu. Batch sayısı: {len(train_dataloader)}")

print("\nOptimizer oluşturuluyor...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("\nEğitim başlıyor...")
model.train()
num_steps = 5  # Sadece 5 adım eğitim

for step in range(num_steps):
    print(f"\nAdım {step+1}/{num_steps}")
    
    # Batch al
    batch = next(iter(train_dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Optimizer adımı
    optimizer.step()
    optimizer.zero_grad()

print("\nBasit eğitim tamamlandı!")

print("\nÖrnek cümle oluşturuluyor...")
model.eval()

prompt = "[USER] How was your weekend? [SEP] "
print(f"Prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=1,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(f"Oluşturulan cevap: {generated_text}")

print("\nİşlem tamamlandı!") 