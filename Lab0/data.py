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
        self.mean = np.array([np.random.random_sample() * dx, np.random.random_sample() * dy])
        D = np.diag([(np.random.random_sample()*dx/5)**2,
                    (np.random.random_sample()*dy/5)**2])
        phi = np.random.random_sample()*2*np.pi
        R = np.array([[np.cos(phi), -np.sin(phi)], 
                      [np.sin(phi), np.cos(phi)]])
        self.cov = np.dot(np.dot(R.T, D), R)
        
        
    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

# if __name__=="__main__":
#     np.random.seed(100)
#     G=Random2DGaussian()
#     X=G.get_sample(100)
#     plt.scatter(X[:,0], X[:,1])
#     plt.show()