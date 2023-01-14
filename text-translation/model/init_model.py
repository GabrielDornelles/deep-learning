import torch
from model.model import Seq2SeqTransformer
import torch.nn as nn
from transforms import vocab_transforms

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'pt'
vocab_transform = vocab_transforms()

PAD_IDX = 1
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 64 #128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

torch.manual_seed(0)


def init_transformer():
    transformer = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, 
        NUM_DECODER_LAYERS, 
        EMB_SIZE,
        NHEAD, 
        SRC_VOCAB_SIZE, 
        TGT_VOCAB_SIZE, 
        FFN_HID_DIM
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
