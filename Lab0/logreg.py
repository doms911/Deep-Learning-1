import numpy as np
import data

class LogReg:
    
    def __init__(self, param_niter=1000, param_delta=0.5):
        self.W = None
        self.b = None
        self.param_niter = param_niter
        self.param_delta = param_delta
        
      # stabilni softmax
    def stable_softmax(self, x):
        exp_x_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)
        return probs
    
    def logreg_train(self, X, Y_):
        N, D = X.shape
        C = np.max(Y_) + 1
        if self.W is None:
            self.W = np.random.normal(0,1,(C, D))
            self.b = np.zeros((C, 1))
            
        for i in range(self.param_niter):
            # eksponencirane klasifikacijske mjere
            # pri računanju softmaksa obratite pažnju
            # na odjeljak 4.1 udžbenika
            # (Deep Learning, Goodfellow et al)!
            scores =  np.dot(X, self.W.T) + self.b.T  # N x C
            expscores = np.exp(scores - scores.max(axis=1, keepdims=True)) # N x C

            # # nazivnik sofmaksa
            sumexp = np.sum(expscores, axis=1, keepdims=True)    # N x 1

            # # logaritmirane vjerojatnosti razreda 
            probs = np.divide(expscores, sumexp)     # N x C
            logprobs = np.log(probs)  # N x C

            # # gubitak
            loss = -np.mean(np.sum(data.class_to_onehot(Y_) * logprobs, axis=1))

            # dijagnostički ispis
            if i % 10 == 0:
                print("iteration {}: loss {}".format(i, loss))

            # derivacije komponenata gubitka po mjerama
            dL_ds = probs - data.class_to_onehot(Y_)    # N x C
            # gradijenti parametara
            grad_W = 1/N * np.dot(dL_ds.T, X)   # C x D (ili D x C)
            grad_b = 1/N * np.sum(dL_ds, axis=0).reshape(-1, 1)    # C x 1 (ili 1 x C)

            # poboljšani parametri
            self.W += -self.param_delta * grad_W
            self.b += -self.param_delta * grad_b
        
        return self.W, self.b
    
    def logreg_classify(self, X, W, b):
        '''
            Argumenti
                X:    podatci, np.array NxD
                W, b: parametri logističke regresije 

            Povratne vrijednosti
                probs: vjerojatnosti razreda c1
        '''
        return self.stable_softmax(np.dot(X, W.T) + b.T)

    def logreg_decfun(self, W, b):
        def classify(self, X):
            probs = self.logreg_classify(X, W, b)
            return np.argmax(probs, axis=1) 
        return classify
    