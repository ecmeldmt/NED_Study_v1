import torch
import clip
from PIL import Image
import os
import json
from google.colab import files
from tqdm import tqdm

data_root = "/content/drive/MyDrive/THESIS/Kod/Ensemble_VLM/DATA/caltech-101/101_ObjectCategories"
split_json_path = "/content/drive/MyDrive/THESIS/Kod/Ensemble_VLM/DATA/caltech-101/split_zhou_Caltech101.json"

# === CLIP MODELLERINI YUKLE ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_names = ["ViT-B/32", "ViT-B/16"]

def load_models(model_names, device):
    models, preprocesses = [], []
    for name in model_names:
        model, preprocess = clip.load(name, device=device)
        models.append(model)
        preprocesses.append(preprocess)
    return models, preprocesses

models, preprocesses = load_models(model_names, device)

for i, model in enumerate(models):
    print(f"Model {i+1}: {model_names[i]}")
    print(f" - Architecture: {model.visual.__class__.__name__}")
    print(f" - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

# === ENSEMBLE FONKSİYONLARI ===
def normalize(x): return x / x.norm(dim=-1, keepdim=True)

def get_logits(model, preprocess, image_path, text_tokens, text_features=None):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = normalize(image_features)
        if text_features is None:
            text_features = model.encode_text(text_tokens)
            text_features = normalize(text_features)
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        confidence = probs.max()
    return logits, confidence

def compute_ensemble_logits(models, preprocesses, image_path, text_tokens, all_text_features):
    logits_list = []
    confs = []
    for model, pre, text_feat in zip(models, preprocesses, all_text_features):
        logits, confidence = get_logits(model, pre, image_path, text_tokens, text_feat)
        logits_list.append(logits)
        confs.append(confidence)
    weights = torch.softmax(torch.tensor(confs), dim=0)
    ensemble_logits = sum(w * l for w, l in zip(weights, logits_list))
    return ensemble_logits, weights

# === SINIF ADLARI VE PROMPTLARI AYARLA ===
with open(split_json_path, 'r') as f:
    split = json.load(f)

test_split = split["test"]

class_names = sorted(list({entry[2] for entry in test_split}))
texts = [f"a photo of a {cls.replace('_', ' ')}" for cls in class_names]
class_to_idx = {cls: i for i, cls in enumerate(class_names)}


# === TEXT OZELLİKLERİNİ ONCEDEN HESAPLA ===
text_tokens = clip.tokenize(texts).to(device)
all_text_features = []
with torch.no_grad():
    for model in models:
        text_features = model.encode_text(text_tokens)
        text_features = normalize(text_features)
        all_text_features.append(text_features)

# === DOĞRULUK ÖLÇ ===
# === 8. TEST SPLIT ÜZERİNDE ENSEMBLE TAHMIN ===
correct = 0
total = 0

print("📊 CLIP Ensemble Test Başladı...\n")

for image_rel_path, _, class_name in tqdm(test_split):
    image_path = os.path.join(data_root, image_rel_path)
    if not os.path.exists(image_path):
        continue

    logits, weights = compute_ensemble_logits(models, preprocesses, image_path, text_tokens, all_text_features)
    pred = logits.argmax().item()
    gt = class_to_idx[class_name]
    correct += int(pred == gt)
    total += 1

    #print(f"{os.path.basename(image_path):<25} → Tahmin: {texts[pred]:<30} | Doğru: {texts[gt]}")
    #print(f"Ağırlıklar: {[round(w.item(), 3) for w in weights]}\n")

accuracy = correct / total if total > 0 else 0
print(f"\n✅ Toplam Test Doğruluğu: {correct}/{total} = {accuracy:.2%}")
