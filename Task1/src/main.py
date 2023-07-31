import pandas as pd
import numpy as np
from BagOfWord import BagOfWords

if __name__ == "__main__":
    train_df = pd.read_csv("../datasets/train.tsv", sep = '\t')
    print(train_df)
    train_data = train_df['Phrase'].values
    bagOfWords = BagOfWords()
    features = bagOfWords.fit_transform(train_data)
    print(features)