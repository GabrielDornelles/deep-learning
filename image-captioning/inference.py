from models.cnn_lstm import CRNN
from models.cnn_transformer import ImageCaptioningModel
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

states = torch.load("crnn_and_vocab_interrupted.pth.tar")
vocabulary = states["vocabulary"]
state_dict = states["model_state_dict"]

#print(vocabulary)
#model = CRNN(embed_size=256, hidden_size=256, vocab_size=len(vocabulary), num_layers=2)
model = ImageCaptioningModel(vocab_size=len(vocabulary), 
        embedding_dim=320, 
        hidden_dim=320, 
        num_layers=2, 
        num_heads=4, 
        dropout=0.2
    )
#model.load_state_dict(state_dict)
model.eval()
 
for idx in range(0,3000,100): # 1000,1100,10
    image = dataset[idx][0]
    sample = transform(dataset[idx][0]).unsqueeze(0)
    caption = model.sample_2(sample,vocabulary,device=torch.device("cpu"))
    
    caption = " ".join(caption)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title(caption)
    plt.axis(False)
    plt.grid(False)
    
    plt.show()
    #break