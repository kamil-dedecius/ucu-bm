import numpy as np
from scipy.stats import binom, beta
import matplotlib.pylab as plt
np.random.seed(4444444)

n = 8
pi = 0.7
nexperiments = 9

heads = binom.rvs(n=n, p=pi, size=nexperiments)
tails = n - heads

a, b = (1, 1)
a_post = np.cumsum(heads) + a
b_post = np.cumsum(tails) + b

a_hist = np.r_[a, a_post]
b_hist = np.r_[b, b_post]
E_hist = a_hist / (a_hist + b_hist)
var_hist = a_hist * b_hist / ((a_hist + b_hist)**2 * (a_hist + b_hist + 1))

f, ax = plt.subplots(2, 5, sharey=True, figsize=(12, 6))
ax = ax.flatten()
x = np.linspace(1e-5, 1, 100)
for i in range(nexperiments + 1):
    ax[i].plot(x, beta.pdf(x, a=a_hist[i], b=b_hist[i]))
    ax[i].vlines(E_hist[i], 0, beta.pdf(E_hist[i], a=a_hist[i], b=b_hist[i]), color='red')
    ax[i].set_title('(a,b)=({0},{1})\n EX={2:.3f}\n varX={3:.3f}'.format(
            a_hist[i], b_hist[i], E_hist[i], var_hist[i]))
plt.tight_layout()

plt.savefig('l1-beta-mince.jpg')