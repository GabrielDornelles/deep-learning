
import torch
import torch.nn as nn
from utils import create_mask
from rich.progress import track


PAD_IDX = 1
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train_epoch(model, optimizer, train_dataloader, device):
    model.train()
    losses = 0
    
    #with torch.cuda.amp.autocast():
    for src, tgt in track(train_dataloader, description="Training..."):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model, val_dataloader, device):
    model.eval()
    losses = 0
    
    with torch.no_grad():
        for src, tgt in track(val_dataloader, description="Validating..."):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

    return losses / len(val_dataloader)