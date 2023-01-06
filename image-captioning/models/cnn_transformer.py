import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from dataset import Flickr
import torchvision.transforms as transforms


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


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Image feature encoder
        self.cnn_encoder = Encoder(embedding_dim)

        # Embedding layer to convert word indices to word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer encoder to process image features
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, 1024 * 1, dropout),
            num_layers
        )
        
        # Transformer decoder to generate captions
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_heads, 1024 * 1, dropout), 
            num_layers
        )
        
        # Linear layer to map output of decoder to word embeddings
        self.output_linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, image, captions):
        image_features = self.cnn_encoder(image)

        # Convert word indices to word embeddings
        embedded_captions = self.embedding(captions)
        embedded_captions = torch.cat((image_features.unsqueeze(0), embedded_captions), dim=0)

        # Pass image features through encoder
        thought_vectors = self.encoder(embedded_captions)

        # Pass thought vectors and embedded captions through decoder
        decoder_output = self.decoder(thought_vectors, embedded_captions)
        
        # Map decoder output to word embeddings
        logits = self.output_linear(decoder_output)
        return logits
    
    def sample(self, image, vocabulary, max_length=20, device=torch.device("cuda")):
        caption = []
        next_word = torch.tensor([1], dtype=torch.int).to(device) # First token is 1: <SOS> Start of sentence
        
        with torch.no_grad():
            image_features = self.cnn_encoder(image)
    
            for _ in range(max_length):
                embedded_captions = self.embedding(next_word)
         
                embedded_captions = torch.cat((image_features, embedded_captions), dim=0)

                thought_vectors = self.encoder(embedded_captions)
                # Pass thought vectors and embedded captions through decoder
                decoder_output = self.decoder(thought_vectors, embedded_captions)#[-1] # take the last step, encoder produces two steps (I dont understand yet)
                #print(decoder_output.shape)
                # Map decoder output to word embeddings
                logits = self.output_linear(decoder_output)
                print(logits.shape)
                
                # Sample next word using predicted logits
                next_word = self.sample_from_logits(logits)
                next_word = torch.tensor([int(next_word)], dtype=torch.int).to(device)
                #print(next_word)
                caption.append(next_word.cpu())
              
                # Stop if end token is generated
                if vocabulary.itos[int(next_word)] == "<EOS>":
                    return [vocabulary.itos[int(idx)] for idx in caption]
                #cur_sentence = [vocabulary.itos[int(idx)] for idx in next_word]
                # if "<EOS>" in cur_sentence:
                #     return cur_sentence

        
        return [vocabulary.itos[int(idx)] for idx in caption]
    
    def sample_2(self, image, vocabulary, max_length=50):
        # Initialize the caption with the start token
        caption = [1]
        image_features = self.cnn_encoder(image).unsqueeze(0)
        states = None

        for _ in range(max_length):

            output = self.decoder()

        

    @staticmethod
    def sample_from_logits(logits):
        # Sample from logits and return the index of the sampled word
        return torch.argmax(logits, dim=-1)
    


# if __name__ == "__main__":
#     transform = transforms.Compose([
#         transforms.Resize((299, 299)),
#         #transforms.RandomCrop((299, 299)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
#     )
#     dataset = Flickr(transform=transform)
#     vocab_size = len(dataset.vocabulary)
#     print(f"Vocabulary size: {vocab_size}")
#     model = ImageCaptioningModel(vocab_size=vocab_size, 
#         embedding_dim=256, 
#         hidden_dim=256, 
#         num_layers=2, 
#         num_heads=2, 
#         dropout=0.1
#     )
#     sample = dataset[50][0]
#     sample = sample[None,...]
#     #print(sample.shape)
#     print(sample.shape)
#     captions = dataset[1][1]
#     print(captions)
#     print(captions.shape)
#     print(captions.dtype)
#     output = model(sample, captions)
#     # print(output.shape)