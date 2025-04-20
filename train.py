import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  # ✅ CSV için pandas ekledik

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB için normalize
])

dataset_root = 'C:/Users/yusuf/PycharmProjects/araba_logo_tahmin/arabadataset'
train_data = datasets.ImageFolder(os.path.join(dataset_root, 'train'), transform=transform)
val_data = datasets.ImageFolder(os.path.join(dataset_root, 'valid'), transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

model = models.resnet18(pretrained=True)
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📊 Loss'ları saklayacağımız liste
train_losses = []

# Eğitim döngüsü
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"📘 Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

# ✅ Modeli kaydet
torch.save(model.state_dict(), 'araba_logosu_model.pth')
print("✅ Eğitim tamamlandı. Model kaydedildi.")

# 📈 Loss grafiğini çiz
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', color='blue', label='Train Loss')
plt.title("Eğitim Süresince Kayıp (Loss) Değeri")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("train_loss_grafik.png")
plt.show()

# ✅ Loss değerlerini CSV olarak kaydet
df = pd.DataFrame({
    'Epoch': list(range(1, epochs + 1)),
    'Train Loss': train_losses
})
df.to_csv('train_losses.csv', index=False)
print("📁 train_losses.csv başarıyla oluşturuldu.")
