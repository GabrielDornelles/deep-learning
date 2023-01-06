# image-captioning

## CNN + LSTM

- Pretrained EfficientNet-B0 in ImageNet
- Word Embedding with Flickr8k vocabulary
- 1 layer LSTM

It was trained for 100 epoches (CNN weights were frozen) and the vocabulary was built with words that appear at least 5 times in the Flickr8k dataset.

![image](https://user-images.githubusercontent.com/56324869/198848257-d981dd83-d362-491a-bbf0-f7ec305798ee.png)


## CNN + Transformer
Not working yet
### TODO:
- Mask encoder while training so it doesnt have acess to future words  (everthing above the upper diag of matrix is zeroed)
- ```sample``` method is implemented in a wrong way.         
Transformer decoder works with memory and not with hidden states. Understand this and rewrite the sample method.
- 