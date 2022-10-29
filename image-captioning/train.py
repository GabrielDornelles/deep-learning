import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomCollate, Flickr
import torchvision.transforms as transforms
from model import CRNN
from rich.progress import track
import time
from datetime import datetime 

def main():
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        #transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    dataset = Flickr(transform=transform)
    vocab_size = len(dataset.vocabulary)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        collate_fn=CustomCollate(pad_idx=0) # our mapped <PAD> token is idx 0
    )
    
    epoches = 100
    device = torch.device("cuda")
    lr = 3e-4

    model = CRNN(embed_size=256, hidden_size=256, vocab_size=vocab_size, num_layers=1)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocabulary.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    try:
        for epoch in range(epoches):
            print(f"{datetime.now().time().replace(microsecond=0)} - Epoch: {epoch}/{epoches - 1}")
            
            losses = []
            for images, captions in track(dataloader, description="Training..."):

                images, captions = images.to(device), captions.to(device)
                outputs = model(images, captions[:-1]) # we want the model to predict <EOS> token, so we dont send it (if we did it could relate the end token to specific words or scenarios)

                # criterion expects 2d tensor (RuntimeError: Expected target size [padded_len, vocab_size], got [padded_len, batch_size])
                # outputs is [padded_len, batch_size, vocab_size] we transform in -> [padded_len * batch_size, vocab_size]
                # captions is [padded_len, batch_size] we transform in -> [padded_len * batch_size]
                # which is pretty much just stack the whole batch of captions into one dimension
                outputs = outputs.reshape(-1, outputs.shape[2])
                captions = captions.reshape(-1)
        
                loss = criterion(outputs, captions)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward() # backward(loss)?
                optimizer.step()
            
            print(f"Epoch loss: {sum(losses)/len(losses)}")
        
        states = {
            "model_state_dict": model.state_dict(),
            "vocabulary": dataset.vocabulary,
        }
        torch.save(states, "crnn_and_vocab.pth.tar") # save vocab and model weights         
        print("Finished training")
    except KeyboardInterrupt:
        states = {
            "model_state_dict": model.state_dict(),
            "vocabulary": dataset.vocabulary,
        }
        torch.save(states, "crnn_and_vocab_interrupted.pth.tar")
    

if __name__ == "__main__":
    main()