import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:
    """Random bivariate normal distribution sampler

    Hardwired parameters:
        d0min,d0max: horizontal range for the mean
        d1min,d1max: vertical range for the mean
        scalecov: controls the covariance range

    Methods:
        __init__: creates a new distribution

        get_sample(n): samples n datapoints

    """
    minx = 0
    maxx = 10
    miny = 0
    maxy = 10

    def __init__(self):
        dx = self.maxx - self.minx
        dy = self.maxy - self.miny
        self.mean = np.array(
            [np.random.random_sample() * dx, np.random.random_sample() * dy])
        D = np.diag([(np.random.random_sample()*dx/5)**2,
                    (np.random.random_sample()*dy/5)**2])
        phi = np.random.random_sample()*2*np.pi
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
        self.cov = np.dot(np.dot(R.T, D), R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)


def sample_gauss_2d(C, N):
    Gs = []
    Ys = []
    for i in range(C):
        Gs.append(Random2DGaussian())
        Ys.append(i)
    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_ = np.hstack([[Y]*N for Y in Ys])
    return X, Y_

def class_to_onehot(Y):
    Yoh=np.zeros((len(Y),max(Y)+1))
    Yoh[range(len(Y)),Y] = 1
    return Yoh

def eval_perf_binary(Y, Y_):
    Y = np.array(Y)
    Y_ = np.array(Y_)
    tp = np.sum(np.logical_and(Y == 1, Y_ == 1))
    tn = np.sum(np.logical_and(Y == 0, Y_ == 0))
    fp = np.sum(np.logical_and(Y == 1, Y_ == 0))
    fn = np.sum(np.logical_and(Y == 0, Y_ == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, accuracy


def eval_AP(Yr):
    sum = 0
    for i in range(len(Yr)):
        Yi = [1 if j >= i else 0 for j in range(len(Yr))]
        pi, _, _ = eval_perf_binary(Yi, Yr)
        sum += pi * Yr[i]
    return sum / np.sum(Yr)

def graph_data(X, Y_, Y, special=[]):
    """Creates a scatter plot (visualize with plt.show)

    Arguments:
        X:       datapoints
        Y_:      groundtruth classification indices
        Y:       predicted class indices
        special: use this to emphasize some points

    Returns:
        None
    """
    # colors of the datapoint markers
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified datapoints
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad],
                s=sizes[bad], marker='s', edgecolors='black')

def graph_surface(function, rect, offset=0.5, width=256, height=256):
  """Creates a surface plot (visualize with plt.show)

  Arguments:
    function: surface to be plotted
    rect:     function domain provided as:
              ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot

  Returns:
    None
  """

  lsw = np.linspace(rect[0][1], rect[1][1], width) 
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

  #get the values and reshape them
  values=function(grid).reshape((width,height))
  
  # fix the range and offset
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  
  # draw the surface and the offset
  plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval)
    
  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])
    
def eval_perf_multi(Y, Y_):
  pr = []
  n = max(Y_)+1
  M = np.bincount(n * Y_ + Y, minlength=n*n).reshape(n, n)
  for i in range(n):
    tp_i = M[i,i]
    fn_i = np.sum(M[i,:]) - tp_i
    fp_i = np.sum(M[:,i]) - tp_i
    tn_i = np.sum(M) - fp_i - fn_i - tp_i
    recall_i = tp_i / (tp_i + fn_i)
    precision_i = tp_i / (tp_i + fp_i)
    pr.append( (recall_i, precision_i) )
  
  accuracy = np.trace(M)/np.sum(M)
  
  return accuracy, pr, M

def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores

# if __name__=="__main__":
#     np.random.seed(100)
#     G=Random2DGaussian()
#     X=G.get_sample(100)
#     plt.scatter(X[:,0], X[:,1])
#     plt.show()

# if __name__ == "__main__":
#     print(eval_AP([0,0,0,1,1,1]))    # Expect 1.0
#     print(eval_AP([0,0,1,0,1,1]))    # Expect 0.9166666666666666
#     print(eval_AP([0,1,0,1,0,1]))    # Expect 0.7555555555555555
#     print(eval_AP([1,0,1,0,1,0]))    # Expect 0.5


if __name__=="__main__":
    np.random.seed(100)
  
    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)
  
    # get the class predictions
    Y = myDummyDecision(X)>0.5
  
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, rect, offset=0)
    
    graph_data(X, Y_, Y) 

  
    # show the results
    plt.show()