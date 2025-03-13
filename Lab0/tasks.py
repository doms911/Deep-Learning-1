import numpy as np
import matplotlib.pyplot as plt
from binlog import BinLogReg, binlogreg_decfun
from logreg import LogReg
import data
from typing import Tuple

def generate_data(nclasses: int, nsamples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data for classification."""
    np.random.seed(seed)
    X, Y_ = data.sample_gauss_2d(nclasses, nsamples)
    return X, Y_

def train_binlogreg(X: np.ndarray, Y_: np.ndarray, param_niter: int, param_delta: float) -> tuple[np.ndarray, float]:
    """Trains a binary logistic regression model."""
    binlog = BinLogReg(param_niter, param_delta)
    w, b = binlog.binlogreg_train(X, Y_)
    return w, b

def train_multiclass_logreg(X: np.ndarray, Y_: np.ndarray, param_niter: int, param_delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Trains a multi-class logistic regression model."""
    logreg = LogReg(param_niter, param_delta)
    W, b = logreg.logreg_train(X, Y_)
    return W, b



def second():
    # get the training dataset
    X, Y_ = generate_data(2, 100, 100)
    
    binlog = BinLogReg()
    
    # train the model
    w,b = binlog.binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlog.binlogreg_classify(X, w,b)
    Y = [1 if p >= 0.5 else 0 for p in probs]

    # report performance
    precision, recall, acc = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("AP: ", AP)

def fifth():
    # instantiate the dataset
    X,Y_ = generate_data(nclasses=2, nsamples=100, seed=100)

    # train the logistic regression model
    binlog = BinLogReg()
    w,b = binlog.binlogreg_train(X, Y_)
    probs = binlog.binlogreg_classify(X, w, b)

    # evaluate the model on the train set
    data.eval_AP(Y_[probs.argsort()])

    # recover the predicted classes Y
    Y = [1 if p > 0.5 else 0 for p in probs]

    # evaluate and print performance measures
    precision, recall, acc = data.eval_perf_binary(Y, Y_)
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)   

    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
    
def sixth():
    # instantiate the dataset
    X,Y_ = generate_data(nclasses=3, nsamples=100, seed=100)

    # train the logistic regression model
    logreg = LogReg()
    W,b = logreg.logreg_train(X, Y_)
    probs = logreg.logreg_classify(X, W, b)

    # recover the predicted classes Y
    Y = np.argmax(probs, axis=1)

    # evaluate and print performance measures
    precision, recall, conf_matrix = data.eval_perf_multi(Y, Y_)
    print("Conf matrix: ", conf_matrix)
    print("Precision: ", precision)
    print("Recall: ", recall)   

    # graph the decision surface
    decfun = logreg.logreg_decfun(W,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
    
if __name__ == "__main__":
    print("\n###########################################################\nSecond task:\n")
    second()
    print("\n###########################################################\nFifth task:\n")
    fifth()
    print("\n###########################################################\nSixth task:\n")
    sixth()