import torch
from torchvision import transforms, models
from PIL import Image
import os

def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB için normalize
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

def main():
    # 🔧 Ayarlar
    dataset_root = 'C:/Users/yusuf/PycharmProjects/araba_logo_tahmin/arabadataset'
    model_path = 'araba_logosu_model.pth'
    image_path = 'C:/Users\yusuf\PycharmProjects/araba_logo_tahmin\deneme2.png'  # Test etmek istediğin tam görsel yolu

    # 🎯 Sınıf isimlerini al
    class_names = os.listdir(os.path.join(dataset_root, 'train'))

    # 📦 Modeli yükle
    model = load_model(model_path, len(class_names))

    # 🔮 Tahmin
    predicted_class = predict_image(model, image_path, class_names)
    print(f"📸 Tahmin edilen araba markası: **{predicted_class}**")

if __name__ == '__main__':
    main()
