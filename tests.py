import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Load a pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess an image
image = Image.open('/home/mashalimay/DL_project/Diffusion-Model-Latent-Space-Manipulation/data/original/Airplane/000000084752.jpg')
transform = transforms.Compose([
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    prediction = model(image)

# Visualize the result
plt.figure(figsize=(8,8))
plt.imshow(image.squeeze(0).permute(1, 2, 0))
for element in prediction[0]['masks']:
    mask = element[0].numpy()
    plt.imshow(mask, alpha=0.5, cmap='gray')
plt.axis('off')
plt.savefig('output.png')
plt.close()
