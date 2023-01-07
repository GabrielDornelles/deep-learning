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
- Mask encoder while training so it doesnt have acess to future words  (everthing above the upper diag of matrix is zeroed). ✅ (I believe).

- ```sample``` method is implemented in a wrong way. ✅ (Implemented one token at a time, appending the token to the next decoder prediction).

- Transformer decoder works with memory and not with hidden states. Understand this and rewrite the sample method. ✅

- Make it work. Right now after a lot of training, the current model just predicts a lot of words without meaning:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/56324869/211156154-13c0d5ff-6aa5-45dd-8329-16f419225401.png" alt="drawing" width="250"/>
    </p>

    ```
    <SOS> biting expression first campsite dusk spout appear darkened crosses policeman afternoon dolphins female bee fallen step dirty bicycler area finish skeleton women are restaurant sky mother yellow looking tournament cop beige ankle ambulance wings handbag batman electronic yellow canoes grins perform tracksuit over rv underneath neon meet cheerleader cries standS
    ```