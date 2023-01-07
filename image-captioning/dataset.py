import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy
import numpy as np
# Originally written by: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/image_captioning/get_loader.py
# I just added comments to ensure its understandable without any further explanations.


spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    """
    The Vocabulary class is responsable by creating the dictionaries of
    'String to Index (stoi)' and 'Index to String (itos)'. The dicts are
    initialized with the following Tokens:
    - <PAD> : A pad token is a token that is used to pad a tokenized sentence to a fixed length.
    - <SOS>: This token defines the 'Start of Sentence'
    - <EOS>: This token defines the 'End of Sentence'
    - <UNK>: We convert words to this token when the infered word is not found in the dictionaries.
    We use the freq_threshold to define which words are in our dicts, that is, words that appear less than
    this threshold are not mapped to our dicts.

    After that, we start mapping words that appear in our dataset (at least freq_threshold times) 
    starting by the index 4 (Flickr dataset). 
    We use spacy english language tokenizer.
    """
    def __init__(self, freq_threshold : int = 5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        """
        Sentence list is a list of every sentence in our dataset
        """
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, caption):
        """
        Receives a caption, transform to tokenized list of strings,
        then transform list of strings into list of indexes. 
        The indexes are used as input in the embedding layer.
        """
        tokenized_text = self.tokenizer_eng(caption)
        numericalized = [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] 
            for token in tokenized_text]
        return numericalized

class CustomCollate:
    """
    Custom collate functionality to pad tokenized targets to be same length 
    (it has to be if we want to train in batches).
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


class Flickr(Dataset):

    def __init__(self, transform=None) -> None:
        super().__init__()
        self.root_dir = "flickr8k/"
        csv_file = pd.read_csv("flickr8k/captions.txt", delimiter= ',')
        self.images = csv_file["image"]
        self.anns = csv_file["caption"]
        self.transform = transform
        self.vocabulary = Vocabulary(freq_threshold=5)
        self.vocabulary.build_vocabulary(self.anns.tolist())
        self.fixed_length = 100
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(f"{self.root_dir}/images/{self.images[idx]}")
        anns = self.anns[idx]

        # apply transform to the image
        if self.transform is not None:
            image = self.transform(image)

        # image = np.array(image).transpose(2,0,1)
        # image = torch.tensor(image, dtype=torch.float32)#torch.tensor(np.array(image.transpose(2,0,1)), dtype=torch.float32)
        #image = torch.tensor(np.array(image.transpose(2,0,1))).float()
        # transform words into tensor of indexes
        tokenized_caption = [self.vocabulary.stoi["<SOS>"]] # Start the sentence
        tokenized_caption += self.vocabulary.numericalize(anns) # add the sentence
        tokenized_caption.append(self.vocabulary.stoi["<EOS>"]) # End the sentence
        tokenized_caption += [0] * int(self.fixed_length - len(tokenized_caption)) # pad to fixed length with token 0 (<PAD>)
        tokenized_caption = torch.tensor(tokenized_caption)
        #print(len(tokenized_caption))
        return image, tokenized_caption
        