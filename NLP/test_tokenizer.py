from transformers import AutoTokenizer
import torch

# Tokenizer'ı yükle
print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")
print(f"Tokenizer yüklendi. Kelime dağarcığı boyutu: {len(tokenizer)}")

# Özel tokenleri kontrol et
print("\nÖzel tokenler:")
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"Ek özel tokenler: {tokenizer.additional_special_tokens}")

# Örnek cümleler
print("\nÖrnek cümleler tokenize ediliyor:")
examples = [
    "[USER] How was your weekend? [SEP] [ASSISTANT] It was great! I went hiking with my friends. [SEP]",
    "[USER] Can you tell me about your favorite book? [SEP] [ASSISTANT] My favorite book is 'To Kill a Mockingbird' by Harper Lee. [SEP]",
    "[USER] What do you think about climate change? [SEP] [ASSISTANT] Climate change is a serious issue that requires global cooperation. [SEP]"
]

for i, example in enumerate(examples):
    # Tokenize et
    tokens = tokenizer.tokenize(example)
    token_ids = tokenizer.encode(example)
    
    print(f"\nÖrnek {i+1}:")
    print(f"Metin: {example}")
    print(f"Token sayısı: {len(tokens)}")
    print(f"İlk 10 token: {tokens[:10]}")
    print(f"İlk 10 token ID: {token_ids[:10]}")
    
    # Decode et
    decoded = tokenizer.decode(token_ids)
    print(f"Decode edilmiş metin: {decoded}")

# Konuşma formatını test et
print("\nKonuşma formatı testi:")
conversation = [
    {"role": "user", "content": "How was your weekend?"},
    {"role": "assistant", "content": "It was great! I went hiking with my friends."},
    {"role": "user", "content": "That sounds fun! Where did you go hiking?"},
    {"role": "assistant", "content": "We went to the mountains near the city. The views were amazing!"}
]

# Konuşmayı formatlı metne dönüştür
formatted_text = ""
for turn in conversation:
    if turn["role"] == "user":
        formatted_text += f"[USER] {turn['content']} [SEP] "
    else:
        formatted_text += f"[ASSISTANT] {turn['content']} [SEP] "
formatted_text += tokenizer.eos_token

print(f"Formatlı metin: {formatted_text}")

# Tokenize et
tokens = tokenizer.tokenize(formatted_text)
token_ids = tokenizer.encode(formatted_text)

print(f"Token sayısı: {len(tokens)}")
print(f"İlk 10 token: {tokens[:10]}")
print(f"İlk 10 token ID: {token_ids[:10]}")

# Decode et
decoded = tokenizer.decode(token_ids)
print(f"Decode edilmiş metin: {decoded}")

print("\nTokenizer testi tamamlandı!") 