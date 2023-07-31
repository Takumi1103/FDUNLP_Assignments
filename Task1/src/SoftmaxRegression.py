import numpy as np

class softmaxRegression():
    def __init__(self, n_epoch=10, learning_rate=1e-3):
        self.epoches = n_epoch
        self.learning_rate = learning_rate
        self.batch_size = None
        self.w = None
        self.num_class = None
        self.n_features = None
    
    def fit(self, data_x, data_y, n_epoch, num_class):
        '''
        :param data_x: [batch_size, n_features]
        :param data_y: [batch_size, 1]
        :param w:      [num_class, n_features]
        '''
        self.batch_size, self.n_features = data_x.shape
        self.epoches = n_epoch
        self.num_class = num_class
        self.w = np.random.randn(self.num_class, self.n_features)
        y_one_hot = np.zeros((self.batch_size, self.num_class))

        for i in range(self.batch_size):
            y_one_hot[i][data_y[i]] = 1

        loss_history = []

        # cal batch_loss
        for k in range(self.epoches):
            loss = 0
            probs = data_x.dot(self.w.T)
            probs = softmax(probs)
            for i in range(self.batch_size):
                loss -= np.log(probs[i][data_y[i]])
            loss /= self.batch_size
        
            # update weight
            weight_update = np.zeros_like(self.w)
            for i in range(self.batch_size):
                # [n_features, num_class]
                weight_update -= data_x[i].reshape(1, self.n_features).T.dot((data_y[i] - probs[i]).reshape(1, self.num_class)).T
            self.w -= self.learning_rate * weight_update / self.batch_size

            loss_history.append(loss)
            if k % 10 == 0:
                print(f'epoch{k} loss {loss}')

        return loss_history
    
    def predict(self, X):
        probs = softmax(X.dot(self.w.T)) 
        return probs.argmax(axis=1)

    def score(self, X, Y):
        pred = self.predict(X)
        return np.sum(pred.argmax(axis=1) == Y) / Y.shape[0]

        




def softmax(x):
    # 对每一行softmax
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    x /= np.sum(x, axis=1, keepdims=True)
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


                



    