import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from dataset import Flickr


class Encoder(nn.Module):

    def __init__(self, embed_size, freeze_weights : bool = True) -> None:
        super().__init__()
        self.convnet = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.convnet.classifier[1] = nn.Linear(in_features=1280, out_features=embed_size, bias=True)
        self.dropout = nn.Dropout(0.5)

        if freeze_weights:
            for name, parameter in self.convnet.named_parameters():
                if "classifier" in name:
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False

    def forward(self, x):
        features = self.convnet(x) # torch.Size([1, embed_size])
        return self.dropout(F.relu(features))

class Print(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, layer_name):
        print(f"{layer_name} shape: {x.shape}")
        return x


class Decoder(nn.Module):
    """
    Functionality 
    ---

    embed_size is the amount of neurons we choose to represent a word (in nn.embedding), and is the neural net task to learn
    the best weights to represent this words. Thats why nn.embedding is torch.size(vocab_size, embed_size), which in our case
    is (2994, 256). What that means is that we have 256 numbers to represent each word in our vocabulary. What we do is
    that we plug that embed vector where each word is a 256d vector into a block that process this sequence (rnn, lstm, transformer etc...)
    and then connect it back to a linear layer that will have our vocabulary as a probability distribuition. Having all that, we input
    our embed vector with words and image features into the (in this case) LSTM, it outputs a vector that will do matmul with a 
    linear layer that outputs a vocab_size vector of probabilities for the word, we take softmax and argmax to see which 
    word was predicted, and we use [this word] + [image feature vector again] as the input again to predict the next. 
    At test time, the model predicts which word comes next while also considering the image feature vector at every step.
    We do it until we find the <EOS> token. 
    This model task is to learn good representations between words and the image feature vector, 
    thats why we concatenate words and image features as the input to LSTM.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.encoderCNN = Encoder(embed_size)
        self.decoderRNN = Decoder(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = (output).argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]


if __name__ == "__main__":
    dataset = Flickr()
    vocab_size = len(dataset.vocabulary)
    print(f"Vocabulary size: {vocab_size}")
    model = CRNN(embed_size=256, hidden_size=256, vocab_size=vocab_size, num_layers=2) # Encoder(embed_size=256) #
    sample = dataset[50][0]
    sample = sample[None,...]
    captions = dataset[50][1]
    output = model(sample, captions)
    #print("\nModel prediction:")
    #print(model.caption_image(sample,dataset.vocabulary))
    