from dataset import MNIST
import matplotlib.pyplot as plt 
import torch 
from model import ViT
import torch.nn.functional as F

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" 

dataset = MNIST() 
model = ViT().to(DEVICE) 
model.load_state_dict(torch.load('model.pth'))

model.eval() 

image,label=dataset[4567]
print('正确分类:',label)
plt.imshow(image.permute(1,2,0))
plt.show()

logits=model(image.unsqueeze(0).to(DEVICE))
print('预测分类:',logits.argmax(-1).item())