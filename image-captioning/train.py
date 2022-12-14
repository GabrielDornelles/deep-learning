import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomCollate, Flickr
import torchvision.transforms as transforms
from models.cnn_lstm import CRNN
from models.cnn_transformer import ImageCaptioningModel
from rich.progress import track
import time
from datetime import datetime 
import wandb
import sys
from rich.console import Console
console = Console()


def main():
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        #transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    sample_dataset = Flickr()
    dataset = Flickr(transform=transform)
    vocab_size = len(dataset.vocabulary)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        collate_fn=CustomCollate(pad_idx=0) # our mapped <PAD> token is idx 0
    )
    
    epoches = 150
    device = torch.device("cuda")
    lr = 3e-4

    #model = CRNN(embed_size=256, hidden_size=256, vocab_size=vocab_size, num_layers=1)
    model = ImageCaptioningModel(vocab_size=vocab_size, 
        embedding_dim=320,#512, 
        hidden_dim=320, 
        num_layers=4, 
        num_heads=5, 
        dropout=0.2
    )
    model.to(device)

    #model = torch.compile(model)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocabulary.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.8, patience=5, verbose=True
        )

    experiment = wandb.init(project='image-captioning-cnn-transformer', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epoches, batch_size=16, learning_rate=lr, amp=True))
    global_step = 0
    try:
        for epoch in range(epoches):
            print(f"{datetime.now().time().replace(microsecond=0)} - Epoch: {epoch}/{epoches - 1}")
            losses = []
            model.train()
            for images, captions in track(dataloader, description="Training..."):
                
                with torch.cuda.amp.autocast():
                    images, captions = images.to(device), captions.to(device)
                    # TODO: when batching, captions batch_size dimension is the second (why?)
                    #print(captions.shape)
                    #print(captions[:-1])
                    outputs = model(images, captions[:-1]) # captions[:-1] we want the model to predict <EOS> token, so we dont send it (if we did it could relate the end token to specific words or scenarios)

                # criterion expects 2d tensor (RuntimeError: Expected target size [padded_len, vocab_size], got [padded_len, batch_size])
                # outputs is [padded_len, batch_size, vocab_size] we transform in -> [padded_len * batch_size, vocab_size]
                # captions is [padded_len, batch_size] we transform in -> [padded_len * batch_size]
                # which is pretty much just stack the whole batch of captions into one dimension

                

                #outputs = outputs.view(-1, outputs.shape[-1])
                outputs = outputs.reshape(-1, outputs.shape[2])
                captions = captions.reshape(-1)
                # print(outputs.shape)
                # print(captions.shape)
                
                loss = criterion(outputs, captions) * 20
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                global_step += 1
                experiment.log({
                    'loss': loss.item(),
                    'global_step': global_step,
                    'epoch': epoch
                })
                
            
            model.eval()
            idxs = list(range(0,3000,300))
            for idx in idxs:
                sample_image = sample_dataset[idx][0]
                image = transform(sample_image)
                image = image[None, ...].cuda()
                caption = model.sample_2(image,dataset.vocabulary,device=torch.device("cuda"))
                
                caption = " ".join(caption)

                experiment.log({ 
                    'Sample': wandb.Image(sample_image, caption=caption),
                    
                })
            epoch_loss = sum(losses)/len(losses)
            print(f"Epoch loss: {epoch_loss}")
            experiment.log({
                "epoch_loss": epoch_loss
            })
            scheduler.step(epoch_loss)

        
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

    except Exception:
        console.print_exception()
        sys.exit()

    

if __name__ == "__main__":
    main()