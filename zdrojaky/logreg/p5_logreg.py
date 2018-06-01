import numpy as np
import matplotlib.pylab as plt
from pandas_ml import ConfusionMatrix
from logreg import LogReg

np.random.seed(1234)
ndat = 1000

fn = 'Skin_NonSkin.txt'
data = np.loadtxt(fn)
data[:,-1] -= 1
np.random.shuffle(data)

data = np.c_[np.ones(data.shape[0]), data]   # adding offset term


#%% Data
X = data[:ndat, :-1]
y = data[:ndat, -1]

#%% Prior
mean = np.zeros(X.shape[-1])
Sigma = np.eye(X.shape[-1]) * 100.
prior = LogReg(mean=mean, Sigma=Sigma)

#%% Estimation
for xt, yt in zip(X, y):
    prior.update(yt, xt)
    prior.log()
    
#%% Confusion matrix
CM = ConfusionMatrix(prior.true_vals, prior.binary_preds)
CM.print_stats()

#%% Plots
beta_log = np.array(prior.mean_log)

plt.figure(figsize=(8, 4))
plt.plot(prior.brier_score_log)
plt.xlabel('t')
plt.ylabel('Brier score')
plt.savefig('/tmp/l5-brier.png', bbox_inches='tight')