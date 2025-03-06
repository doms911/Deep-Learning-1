import numpy as np

class BinLogReg:
    def __init__(self, param_niter=100, param_delta=0.1):
        self.w = np.normal(0, 1, 2)     # 2 because binary
        self.b = 0
        self.param_niter = param_niter
        self.param_delta = param_delta

    def binlogreg_train(self, X,Y_):
        '''
            Argumenti
            X:  podatci, np.array NxD
            Y_: indeksi razreda, np.array Nx1

            Povratne vrijednosti
            w, b: parametri logističke regresije
        '''
        # gradijentni spust (param_niter iteracija)
        for i in range(self.param_niter):
            # klasifikacijske mjere
            scores = ...    # N x 1

            # vjerojatnosti razreda c_1
            probs = ...     # N x 1

            # gubitak
            loss  = ...     # scalar

            # dijagnostički ispis
            if i % 10 == 0:
                print("iteration {}: loss {}".format(i, loss))
            # derivacije gubitka po klasifikacijskim mjerama
            dL_dscores = ...     # N x 1

            # gradijenti parametara
            grad_w = ...     # D x 1
            grad_b = ...     # 1 x 1

            # poboljšani parametri
            w += -self.param_delta * grad_w
            b += -self.param_delta * grad_b