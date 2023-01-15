from utils import translate
from model.init_model import init_transformer
import torch
from transforms import vocab_transforms, text_transforms

model = init_transformer()
model.load_state_dict(torch.load("en_to_pt_transformer.pth"))
model.to(torch.device("cuda"))
model.eval()

# TODO: serialize this dictionaries so we don't have to build it in runtime 
vocab_transform = vocab_transforms()
text_transform = text_transforms()


sentences = [
    "Who is the most beautiful woman in the world?",
    "USA is a criminal country, and it lies to everyone.",
    "Brazil is a country famous by it's beautiful nature.",
    "Brazil is the country of soccer. But they don't play football.",
    "I was running at the beach and then I saw two guys playing volleyball.",
    "I will answer your messages, but only after I go to the gym.",
    "I have a smartphone.",
    "I was walking and then I had a conversation with an old man."
]

for sentence in sentences:

    result = translate(
        model=model,
        src_sentence=sentence,
        text_transform=text_transform,
        vocab_transform=vocab_transform
    )

    print(f"Input sentence: {sentence}")
    print(f"Output sentence: {result}")
    print()