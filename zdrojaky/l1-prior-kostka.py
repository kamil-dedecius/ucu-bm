import numpy as np
import matplotlib.pylab as plt

fair = np.ones(6)
fair /= fair.sum()

unfair = np.ones(6)
unfair[-1] = 5
unfair /= unfair.sum()

rand = np.random.dirichlet([1, 1, 1, 1, 1, 1])

x = range(1, 7)

plt.figure(figsize=(14, 3))
plt.subplot(1, 3, 1)
plt.stem(x, fair)
plt.ylim((0, 1))
plt.title('(A)')
plt.ylabel('P(X)')
plt.xlabel('X')
plt.subplot(1, 3, 2)
plt.stem(x, unfair)
plt.title('(B)')
plt.xlabel('X')
plt.ylim((0, 1))
plt.subplot(1, 3, 3)
plt.stem(x, rand)
plt.ylim((0, 1))
plt.xlabel('X')
plt.title('(C)')
plt.savefig('l1-prior-kostka.jpg', layout='tight')