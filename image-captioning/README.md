# image-captioning

This repository contains an image captioning system that is composed of:

- Pretrained EfficientNet-B0 in ImageNet
- Word Embedding with Flickr8k vocabulary
- 1 layer LSTM

It was trained for 100 epoches (CNN weights were frozen) and the vocabulary was built with words that appear at least 5 times in the Flickr8k dataset.

![image](https://user-images.githubusercontent.com/56324869/198848257-d981dd83-d362-491a-bbf0-f7ec305798ee.png)

