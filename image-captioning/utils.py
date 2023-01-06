import matplotlib.pyplot as plt

def log_some_examples(dataset, transform, model, vocabulary):
    for idx in range(0,3000,300): # 1000,1100,10
        image = dataset[idx][0]
        sample = transform(dataset[idx][0]).unsqueeze(0)
        caption = model.sample(sample,vocabulary)
        
        caption = " ".join(caption)
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.title(caption)
        plt.axis(False)
        plt.grid(False)
        #plt.show()