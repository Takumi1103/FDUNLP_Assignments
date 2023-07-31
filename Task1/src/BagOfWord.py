import numpy as np
class BagOfWords():
    def __init__(self, do_lower_case=False):
        self.vocab = []
        self.do_lower_case = do_lower_case
    def fit_transform(self, sentence_list):
        # make vocabs
        for sentence in sentence_list:
            for word in sentence.strip().split(" "):
                if self.do_lower_case:
                    if word not in self.vocab:
                        self.vocab.append(word.lower())
                else:
                    if word not in self.vocab:
                        self.vocab.append(word)
        # make features
        features = np.zeros((len(sentence_list), len(self.vocab)))
        for idx, sent in enumerate(sentence_list):
            if self.do_lower_case:
                sent = sent.lower()
            for word in sent.strip().split(" "):
                features[idx][self.vocab.index(word)] += 1
        return features


            








        