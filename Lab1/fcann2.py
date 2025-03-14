import numpy as np
import data
from data import sample_gmm_2d, graph_surface, graph_data
import matplotlib.pyplot as plt


class Fcann2:
    def __init__(self, param_niter=1000, param_delta=0.5, param_lambda=1e-3, D_hidden=5):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        self.param_niter = param_niter
        self.param_delta = param_delta
        self.param_lambda = param_lambda
        self.D_hidden = D_hidden
        
    def stable_softmax(self, x):
        exp_x_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)
        return probs

    def fcann2_train(self, X, Y_):
        N, D_in = X.shape
        C = np.max(Y_) + 1
        
        if self.W1 is None and self.b1 is None:
            self.W1 = np.random.randn(D_in, self.D_hidden)
            self.b1 = np.zeros((1, self.D_hidden))
            
        if self.W2 is None and self.b2 is None:
            self.W2 = np.random.randn(self.D_hidden, C)
            self.b2 = np.zeros((1, C))
            
        for i in range(self.param_niter):
            # racunamo unaprijedni prolaz
            # model D_in -> D_hidden -> C
            s1 = np.dot(X, self.W1) + self.b1 # N x D_hidden
            h1 = np.maximum(0, s1) # N x D_hidden
            
            s2 = np.dot(h1, self.W2) + self.b2 # N x C
            
            probs = self.stable_softmax(s2) # N x C 
        
            # logaritmirane vjerojatnosti razreda
            logprobs = np.log(probs)  # N x C
            
            # gubitak
            loss = -np.mean(np.sum(data.class_to_onehot(Y_)
                            * logprobs, axis=1))

            # dijagnostički ispis
            if i % 500 == 0:
                print("iteration {}: loss {}".format(i, loss))

            # derivacije komponenata gubitka po mjerama
            dL_ds2 = probs - data.class_to_onehot(Y_)    # N x C
            dL_ds2 /= N     # uprosjecivanje gradijenata
            
            # gradijenti parametara
            grad_W2 = np.dot(dL_ds2.T, h1) + self.param_lambda * self.W2.T # C x D_hidden
            grad_b2 = np.sum(dL_ds2, axis=0).reshape(-1, 1) # C x 1 
        
            dL_dh1 = np.dot(dL_ds2, self.W2.T) # N x D_hidden
            dL_ds1 = dL_dh1 * (s1>0).astype(float) # N x D_hidden
            
            grad_W1 = np.dot(dL_ds1.T, X) + self.param_lambda * self.W1.T   # D_hidden x D
            grad_b1 = np.sum(dL_ds1, axis=0).reshape(-1, 1) # D_hidden x 1 

            # poboljšani parametri
            self.W1 += -self.param_delta * grad_W1.T
            self.b1 += -self.param_delta * grad_b1.T
            self.W2 += -self.param_delta * grad_W2.T
            self.b2 += -self.param_delta * grad_b2.T


    def fcann2_classify(self, X):
        s1 = np.dot(X, self.W1) + self.b1
        h1 = np.maximum(0, s1)
        s2 = np.dot(h1, self.W2) + self.b2
        probs = self.stable_softmax(s2)
        return np.argmax(probs, axis=1)


if __name__ == '__main__':
    np.random.seed(100)
    K, C, N = 6, 2, 10
    param_niter, param_delta, param_lambda = int(1e5), 0.05, 1e-3
    hidden_dim = 5
    X, Y_ = sample_gmm_2d(K, C, N)
    fcann2 = Fcann2(param_niter=param_niter, param_delta=param_delta, param_lambda=param_lambda, D_hidden=hidden_dim)
    fcann2.fcann2_train(X, Y_)
    Y = fcann2.fcann2_classify(X)
    acc, pr, M = data.eval_perf_multi(Y, Y_)
    print("Accuracy: ", acc)
    print("Precision: ", pr)
    print("Confusion matrix: ")
    bbox = (np.min(X, axis=0) - 0.1, np.max(X, axis=0) + 0.1)
    decfun = fcann2.fcann2_classify
    graph_surface(decfun, bbox, offset=0.5)
    graph_data(X, Y_, Y)
    plt.show()
