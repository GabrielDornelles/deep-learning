from torch.utils.data import DataLoader
from dataset import TatoebaDataset
from transforms import collate_fn
from engine import train_epoch, evaluate
import torch.optim as optim
import torch
from model.init_model import init_transformer
from timeit import default_timer as timer


BATCH_SIZE = 64
EPOCHES = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    train_iter = TatoebaDataset(split="train")
    val_iter = TatoebaDataset(split="val")
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = init_transformer()
    #model.load_state_dict(torch.load("./en_to_pt_transformer-18.pth"))
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    train_start_time = timer()
    for epoch in range(EPOCHES):
        start_time = timer()
        train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            device=DEVICE
        )
        end_time = timer()
        val_loss = evaluate(
            model=model,
            val_dataloader=val_dataloader,
            device=DEVICE
        )
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    train_end_time = timer()
    torch.save(model.state_dict(), "en_to_pt_transformer.pth")
    print(f"Finished Training in {(train_end_time - train_start_time):.3f}s")

