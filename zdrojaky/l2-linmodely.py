import numpy as np
import matplotlib.pylab as plt


ndat = 20
X0 = np.c_[np.ones(ndat), np.arange(ndat)]
beta0 = np.array([3, 5])

X1 = np.c_[np.ones(ndat), np.arange(ndat), np.arange(ndat)**2]
beta1 = np.array([.4, .3, .8])

X2 = np.c_[np.arange(ndat), np.arange(ndat)**2, np.arange(ndat)**3, np.arange(ndat)**4]
beta2 = np.array([.2, .6, .5, -.03])

plt.figure(figsize=(14, 3))
plt.subplot(1, 3, 1)
plt.plot(X0.dot(beta0))
plt.plot(X0.dot(beta0) + 10*np.random.normal(size=ndat), 'r+', mew=3)
plt.title(r'$y_t = \beta_0 + \beta_1 x_t + \varepsilon_t$')
plt.xlabel(r'$x_t$')
plt.ylabel(r'$y_t$')

plt.subplot(1, 3, 2)
plt.plot(X1.dot(beta1))
plt.plot(X1.dot(beta1) + 50*np.random.normal(size=ndat), 'r+', mew=3)
plt.title(r'$y_t = \beta_0 + \beta_1 x_t + \beta_2 x_t^2 + \varepsilon_t$')
plt.xlabel(r'$x_t$')
plt.ylabel(r'$y_t$')

plt.subplot(1, 3, 3)
plt.plot(X2.dot(beta2))
plt.plot(X2.dot(beta2) + 50*np.random.normal(size=ndat), 'r+', mew=3)
plt.title(r'$y_t = \beta_1 x_t + \beta_2 x_t^2 + \beta_3 x_t^3 + \beta_4 x_t^4 + \varepsilon_t$')
plt.xlabel(r'$x_t$')
plt.ylabel(r'$y_t$')

plt.savefig('/tmp/l2-linmodely.jpg', bbox_inches='tight')