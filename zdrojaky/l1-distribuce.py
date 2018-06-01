import numpy as np
import scipy.stats as ss
import matplotlib.pylab as plt

x = np.linspace(-10, 10, 100)

mean = 0
var1 = 10
var2 = 3
N_vals1 = ss.norm.pdf(x=x, loc=mean, scale=np.sqrt(var1))
N_vals2 = ss.norm.pdf(x=x, loc=mean, scale=np.sqrt(var2))

Cauchy_vals = ss.cauchy.pdf(x)
Gamma_vals1 = ss.gamma.pdf(x[x>0], a=1, scale=1)
Gamma_vals2 = ss.gamma.pdf(x[x>0], a=3, scale=1)


plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)
plt.plot(x, N_vals1, label='N(0, 10)')
plt.plot(x, N_vals2, label='N(0, 3)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.subplot(1,3,2)
plt.plot(x, Cauchy_vals, label='Cauchy(0, 1)')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')

plt.subplot(1,3,3)
plt.plot(x[x>0], Gamma_vals1, label='Gamma(1,1)')
plt.plot(x[x>0], Gamma_vals2, label='Gamma(1,1)')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('l1-distribuce.jpg', bbox_inches='tight')

