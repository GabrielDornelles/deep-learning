from model import CRNN
from dataset import Flickr
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
        transforms.Resize((299, 299)),
        #transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

dataset = Flickr()
states = torch.load("crnn_and_vocab.pth_old.tar")
vocabulary = states["vocabulary"]
state_dict = states["model_state_dict"]

model = CRNN(embed_size=256, hidden_size=256, vocab_size=len(vocabulary), num_layers=2)
model.load_state_dict(state_dict)
model.eval()
 
for idx in range(1102,1302,10): # 1000,1100,10
    image = dataset[idx][0]
    sample = transform(dataset[idx][0]).unsqueeze(0)
    caption = model.caption_image(sample,vocabulary)
    caption = " ".join(caption)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title(caption)
    plt.axis(False)
    plt.grid(False)
    plt.show()