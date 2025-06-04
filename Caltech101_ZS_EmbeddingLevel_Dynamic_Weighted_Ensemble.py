import torch
import clip
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
import os

# Cihaz ayari
device = "cuda" if torch.cuda.is_available() else "cpu"

# Iki CLIP modelini yukle (embedding boyutlari ayni: 512)
model1, preprocess1 = clip.load("ViT-B/32", device=device)
model2, preprocess2 = clip.load("ViT-B/16", device=device)


#Caltech101 klasor yolu
data_path = "/content/drive/MyDrive/THESIS/Kod/Ensemble_VLM/DATA/caltech-101/101_ObjectCategories"

# Dataset (ImageFolder kullanarak)
dataset = datasets.ImageFolder(
    root=data_path,
    transform=preprocess1
)

# Sinif adlari
class_names = dataset.classes
print(f"Toplam sinif: {len(class_names)}")

# CLIP formatinda prompt'lar olustur
text_prompts = [f"a photo of a {c}" for c in class_names]
text_tokens = clip.tokenize(text_prompts).to(device)

# Text embeddingleri cikar
with torch.no_grad():
    text_features1 = model1.encode_text(text_tokens)
    text_features2 = model2.encode_text(text_tokens)

text_features1 = F.normalize(text_features1, dim=-1)
text_features2 = F.normalize(text_features2, dim=-1)

# DataLoader
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Dogruluk olcumu
correct = 0
total = 0

# Ilk 500 ornekte test
for images, labels in tqdm(loader, total=500):
    images = images.to(device)
    label = labels.item()

    with torch.no_grad():
        image1 = model1.encode_image(images)
        image2 = model2.encode_image(images)

    image1 = F.normalize(image1, dim=-1)
    image2 = F.normalize(image2, dim=-1)

    # Her modelin kendi metin embedding'ine gore benzerlik skorlari
    sim1 = (image1 @ text_features1.T)
    sim2 = (image2 @ text_features2.T)

    score1 = sim1.softmax(dim=-1)
    score2 = sim2.softmax(dim=-1)

    # Dinamik agirlik: En yuksek skora gore
    w1 = score1.max().item()
    w2 = score2.max().item()
    alpha = w1 / (w1 + w2)

    # Ensemble image embedding
    ensemble_image = F.normalize(alpha * image1 + (1 - alpha) * image2, dim=-1)

    # Ensemble text embedding (sabit)
    ensemble_text = F.normalize((text_features1 + text_features2) / 2, dim=-1)

    # Skorlar
    logits = ensemble_image @ ensemble_text.T
    pred = logits.argmax(dim=-1).item()

    if pred == label:
        correct += 1
    total += 1

    if total == 500:
        break

print(f"Top-1 Accuracy (Zero-Shot + Dynamic Weighted Ensemble): {correct / total:.4f}")


