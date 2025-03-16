import torch
import sys

# Çıktıyı hem dosyaya hem de ekrana yazdır
print(f"PyTorch Versiyonu: {torch.__version__}")
print(f"CUDA Kullanılabilir: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Versiyonu: {torch.version.cuda}")
    print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024} GB")
else:
    print("CUDA Versiyonu: Yok")

# Aynı bilgileri dosyaya da yaz
with open('cuda_info_new.txt', 'w') as f:
    f.write(f"PyTorch Versiyonu: {torch.__version__}\n")
    f.write(f"CUDA Kullanılabilir: {torch.cuda.is_available()}\n")
    if torch.cuda.is_available():
        f.write(f"CUDA Versiyonu: {torch.version.cuda}\n")
        f.write(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024} GB\n")
    else:
        f.write("CUDA Versiyonu: Yok\n")

print("CUDA bilgileri cuda_info_new.txt dosyasına yazıldı.") 