import numpy as np
from scipy.special import expit, logit
from scipy.stats import multivariate_normal as mvn
from collections import Counter

class LogReg():
    """Logistic regression model estimation"""
    def __init__(self, mean, Sigma):
        self.mean = mean                         # mean value
        self.invSigma = np.linalg.inv(Sigma)     # inverse of covariance
        self.mean_log = []                       # log of mean values
        self.Sigma_log = []                      # log of covariances
        self.nu = 0.                             # hyperparameter nu
        self.brier_sum = 0.                      # summing of Brier score
        self.brier_score_log = []                # log of Brier scores
        self.preds = Counter()                   # internal predictions
        self.raw_preds = []                      # raw predictions in [0,1]
        self.binary_preds = []                   # predictions {0,1}
        self.true_vals = []                      # True values
        
#TODO spis nez pouzivat counter by mohlo byt dobre to pocitat
#TODO ex post z self.mean_log - pak by melo jit i vykreslit
#TODO ROC graf pro ruzne cut-off hodnoty, nejen 0.5.

    def update(self, y, X):
        """Bayesian update - korekce stareho kodu"""
        yhat = expit(X.dot(self.mean))
        # 1 Newton step
        Dl = X.T.dot(y - yhat)
        R = yhat*(1. - yhat)
        if not np.isscalar(R):
            R = np.diag(R)
        if len(X.shape)  == 1:
            X = X[np.newaxis,:]
        D2l = -self.invSigma - X.T.dot(R).dot(X)
        self.mean -= np.linalg.inv(D2l).dot(Dl.T).flatten()
        self.invSigma = -D2l
        self.nu += y.size
        self.brier_sum += np.sum((y - yhat) ** 2.)
        y_char = np.char.mod('%d', y.astype(int))
        yhat_char = np.char.mod('%d', np.round(yhat).astype(int))
        res = np.atleast_1d(np.core.defchararray.add(y_char, yhat_char))
        self.preds.update(res)
        self.true_vals.append(y)
        self.binary_preds.append(np.round(yhat).astype(int))
        self.raw_preds.append(yhat)

#    def update(self, y, X):
#        """Bayesian update"""
        
#        yhat = expit(X.dot(self.mean))
#        # 1 Newton step
#        Dl = X.T.dot(y - yhat)
#        R = np.diag(yhat * (1. - yhat))
#        D2l = -self.invSigma - X.T.dot(R).dot(X)
#        self.mean -= np.linalg.inv(D2l).dot(Dl.T).flatten()
#        self.invSigma = -D2l
#        self.nu += y.size
#        self.brier_sum += np.sum((y - yhat) ** 2.)
#        y_char = np.char.mod('%d', y.astype(int))
#        yhat_char = np.char.mod('%d', np.round(yhat).astype(int))
#        res = np.core.defchararray.add(y_char, yhat_char)
#        self.preds.update(res)
#        self.true_vals.append(y)
#        self.binary_preds.append(np.round(yhat).astype(int))
#        self.raw_preds.append(yhat)

    @property
    def brier_score(self):
        """Calculation of Brier score"""
        return self.brier_sum / self.nu
    
    def log(self):
        """Logging"""
        self.mean_log.append(self.mean.copy())
        self.Sigma_log.append(self.Sigma.copy())
        self.brier_score_log.append(self.brier_score)
   
    def predictive_logpdf(self, Y, X):
        """Predictive log pdf"""
        result = []
        try:
            for y, x in zip(Y, X):
                result.append(self._predictive_logpdf(y, x))
        except TypeError:
            result.append(self._predictive_logpdf(Y, X))
        return np.array(result)
        
    def _predictive_logpdf(self, y, x):
        p = expit(x.dot(self.mean))
        log_bernoulli = y * np.log(p) + (1.-y) * np.log(1.-p)
        log_normal = mvn.logpdf(self.mean, self.mean, self.Sigma)
        yhat = p
        det_arg = self.invSigma + yhat*(1.-yhat) * np.outer(x, x)
        log_laplacian = .5 * self.mean.size * np.log(2.* np.pi)
        log_laplacian -= .5 * np.log(np.linalg.det(det_arg))
        return log_laplacian + log_normal + log_bernoulli
    
    @property
    def xi(self):
        """Hyperparameter xi"""
        xi1 = self.invSigma.dot(self.mean)
        xi2 = -.5 * self.invSigma
        return np.vstack((xi1, xi2))
        
    @xi.setter
    def xi(self, xi):
        self.invSigma = - 2. * xi[1:]
        self.mean = xi[0].dot(np.linalg.inv(self.invSigma))
    
    @property
    def Sigma(self):
        """covariance matrix"""
        return np.linalg.inv(self.invSigma)
    
#%% Test
# Doesn't work after code modification
    
if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    ndat = 500
    theta = np.array([-1.6, .03])
    np.random.seed(12345)
    x = np.random.randint(low=18, high=60, size=ndat)
    X = np.c_[np.ones(ndat), x]
    p_x = expit(np.dot(X, theta))
    y = np.random.binomial(n=1, p=p_x)
    
    mean = np.zeros(2)
    Sigma = np.eye(2) * 100.
    prior = LogReg(mean=mean, Sigma=Sigma)
    
    for yt, xt in zip(y, X):
        prior.update(yt, xt)
        prior.log()
        print(prior.mean)

    mean_log = np.array(prior.mean_log)
        
    plt.figure(1)
    plt.plot(mean_log)
    plt.axhline(theta[0])
    plt.axhline(theta[1])