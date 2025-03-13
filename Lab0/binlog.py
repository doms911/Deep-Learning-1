import numpy as np

class BinLogReg:
    def __init__(self, param_niter=1000, param_delta=0.5):
        self.w = None
        self.b = 0.0
        self.param_niter = param_niter
        self.param_delta = param_delta
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binlogreg_train(self, X,Y_):
        '''
            Argumenti
            X:  podatci, np.array NxD
            Y_: indeksi razreda, np.array Nx1

            Povratne vrijednosti
            w, b: parametri logističke regresije
        '''
        N, D = X.shape
        if self.w is None:
            self.w = np.random.normal(0, 1, D)
            
        # gradijentni spust (param_niter iteracija)
        for i in range(self.param_niter):
            # klasifikacijske mjere
            scores = np.dot(X, self.w) + self.b    # N x 1
            # vjerojatnosti razreda c_1
            probs = self.sigmoid(scores)      # N x 1

            # gubitak
            loss  = np.mean(np.sum(- Y_ * np.log(probs) - (1 - Y_) * np.log(1 - probs)))     # scalar

            # dijagnostički ispis
            if i % 10 == 0:
                print("iteration {}: loss {}".format(i, loss))
            # derivacije gubitka po klasifikacijskim mjerama
            dL_dscores = probs - Y_     # N x 1
  
            # gradijenti parametara
            grad_w = (1/N) * np.dot(X.T, dL_dscores)     # D x 1
            grad_b = (1/N) * np.sum(dL_dscores)     # 1 x 1

            # poboljšani parametri
            self.w += -self.param_delta * grad_w
            self.b += -self.param_delta * grad_b
        return self.w, self.b
            
    def binlogreg_classify(self, X, w, b):
        '''
            Argumenti
                X:    podatci, np.array NxD
                w, b: parametri logističke regresije 

            Povratne vrijednosti
                probs: vjerojatnosti razreda c1
        '''
        return self.sigmoid(np.dot(X, w) + b)

def binlogreg_decfun(w,b):
    def classify(X):
        probs = np.dot(X, w) + b
        return np.array([1 if p >= 0.5 else 0 for p in probs])
    return classify

