import sys
sys.path.insert(0, '../zdrojaky/')

import numpy as np
import matplotlib.pylab as plt
from nig import NiG

np.random.seed(111)

# Data
ndat = 80
v0 = 4
a = .5
noise_var = 900
y = np.empty(ndat)


beta = np.array([v0, a])
for t in range(ndat):
    xt = np.array([t+1, .5*(t+1)**2])
    print('Xt: ', xt)
    y[t] = beta.dot(xt)

noise = np.random.normal(scale=np.sqrt(noise_var), size=y.shape)
y_noisy = y + noise

#%% Test odhad MLE
X = np.c_[np.arange(1, ndat+1), .5*np.arange(1, ndat+1)**2]
np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y_noisy))


#%% Model
xi0 = np.diag([.01, .1, .1])
nu0 = 5.
regmodel = NiG(xi0, nu0)
yt_pred = np.empty(ndat)
for st, t in zip(y_noisy, np.arange(ndat)):
    xt = np.array([t+1, .5*(t+1)**2])
    yt_pred[t] = regmodel.beta_hat.dot(xt)
    yt = y_noisy[t]
    regmodel.update(yt, xt)
    regmodel.log()
    print('Estimate: ', regmodel.beta_hat)



steps = [3, 10, 20, 30, 50, ndat]
plt.figure(1, figsize=(14, 5))
for i in range(len(steps)):
    plt.subplot(2, 3, i+1)
    plt.plot(yt_pred[:steps[i]], '.')
    plt.plot(y_noisy[:steps[i]-1])
    plt.xlabel('t')
    plt.ylabel(r'$y_t$')
plt.show()

