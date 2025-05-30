from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


classes = ['plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('Build_model/resnet18_cifar10.pth', map_location=device))
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

image_path = 'Image_input' 
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device) 

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]

print(f'Ảnh {image_path} được dự đoán là: {predicted_class}')
# Hiển thị ảnh với nhãn dự đoán
plt.imshow(image)
plt.title(f'Dự đoán: {predicted_class}')
plt.axis('off')
plt.show()