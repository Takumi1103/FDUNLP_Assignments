import numpy as np
import pandas as pd
from BagOfWord import BagOfWords
from SoftmaxRegression import softmaxRegression

if __name__ == "__main__":
    train_df = pd.read_csv("../datasets/train.tsv", sep='\t')
    data_x = train_df['Phrase'].values
    data_y = train_df['Sentiment'].values
    bagOfWord = BagOfWords()
    features = bagOfWord.fit_transform(data_x)
    model = softmaxRegression(n_epoch=100, learning_rate=1e-3)
    model.fit(features, data_y, n_epoch=10000, num_class=5)

    print(features,data_y)
