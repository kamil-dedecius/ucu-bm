import numpy as np
from scipy.special import logit, expit
import matplotlib.pylab as plt


x_expit = np.linspace(-6, 6, 100)
x_logit = np.linspace(0, 1, 1000)
y_expit = expit(x_expit)
y_logit = logit(x_logit)

plt.figure(1, figsize=(15, 4))
plt.subplot(122)
plt.yticks(np.arange(11)/10)
plt.title('Logistická (sigmoid) funkce')
plt.xlabel(r'$z$')
plt.ylabel(r'$\sigma(z)$')
plt.vlines(0, 0, 1, color='black', lw=1)
plt.hlines(.5, -6, 6, color='black', lw=1)
plt.ylim((0, 1))
plt.xlim(-6, 6)
plt.plot(x_expit, y_expit, lw=3)


plt.subplot(121)
plt.plot(x_logit, y_logit, lw=3)
plt.xticks(np.arange(11)/10)
plt.title('Logit')
plt.xlabel('p')
plt.ylabel('logit(p)')
plt.hlines(0, 0, 1, color='black', lw=1)
plt.vlines(.5, -6, 6, color='black', lw=1)
plt.xlim((0, 1))
plt.ylim(-6, 6)

plt.savefig('/tmp/logit_expit.png', bbox_inches='tight')