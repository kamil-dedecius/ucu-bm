import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from scipy.stats import multivariate_normal as mvn, invgamma

m = np.array([0, 0])
cov = 100 * np.eye(2)
a = .1
b = 10

x = np.linspace(-15, 15, 100)
y = np.linspace(-15, 15, 100)
xv, yv = plt.meshgrid(x, y)

z = mvn.pdf(np.c_[xv.flatten(), yv.flatten()], mean=m, cov=cov).reshape(xv.shape)

plt.figure(1, figsize=(11, 4))
plt.subplot(121)
plt.contourf(xv, yv, z, 20, cmap=cm.coolwarm)
plt.title(r'$\pi(\beta|\sigma^2) = \mathcal{N}\left([0, 0]^\intercal, 100 \cdot I_{[2\times 2]}\right)$')

plt.subplot(122)
plt.plot(np.linspace(0.001, 100, 100),
         invgamma.pdf(np.linspace(0.001, 100, 100), a=a, scale=b))
plt.title(r'$\pi(\sigma^2) = i\mathcal{G}(.1, 10)$')

plt.savefig('/tmp/l2-apriorno-nig.jpg')