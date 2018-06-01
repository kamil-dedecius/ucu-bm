import sys
sys.path.insert(0, '../zdrojaky/')

import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
from nig import NiG
from scipy.stats import norm

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
xi0 = np.diag([10000, .1, .1])
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

#%% Animation
nanim = 30
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(( 0, nanim))
ax.set_ylim((np.min([y_noisy[:nanim].min(), yt_pred[:nanim].min()]),
             np.max([y_noisy[:nanim].max(), yt_pred[:nanim].max()])))

line, = ax.plot([], [], lw=2)
preds = ax.plot([], [])

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    x = np.arange(ndat)[:i+1]
    y = y_noisy[:i]
    y_pred = yt_pred[:i+1]
    line.set_data(x[:i], y)
    ax.plot(x, y_pred, 'g+', markersize=10, mew=3)
    ax.plot(x[-1], y_pred[-1], 'r+', markersize=10, mew=3)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nanim, interval=1, blit=True)

anim.save('/tmp/l2-regrese-anim.gif', writer='imagemagick', fps=.5)

#%%
steps = [3, 10, 20, 30, 50, ndat]
plt.figure(2, figsize=(14, 5))
for i in range(len(steps)):
    plt.subplot(2, 3, i+1)
    plt.plot(yt_pred[:steps[i]], 'g+', mew=1.5)
    plt.plot(y_noisy[:steps[i]-1], 'b')
    plt.xlabel('t')
    plt.ylabel(r'$y_t$')
plt.savefig('/tmp/l2-predikce.jpg', bbox_inches='tight')
plt.show()

#%%
std_sigma2_log = np.sqrt(np.array(regmodel.var_sigma2_log))
plt.figure(3, figsize=(14, 3))
plt.plot(regmodel.Esigma2_log, 'b', label=r'$\hat\sigma^2$')
plt.fill_between(np.arange(ndat),
                 np.array(regmodel.Esigma2_log) + 3. * std_sigma2_log,
                 np.array(regmodel.Esigma2_log) - 3. * std_sigma2_log,
                 color='whitesmoke'
                 )
#plt.plot(E_var_log - 3 * std_sigma2_log, 'g')
#plt.plot(E_var_log + 3 * std_sigma2_log, 'g')
plt.hlines(noise_var, 0, ndat, 'r', label=r'$\sigma^2$')
plt.xlabel('t')
plt.legend()
plt.xlim(xmin=0)
plt.ylim(-2500, 2500)
plt.savefig('/tmp/l2-regrese-Esigma2.jpg')
plt.show()

#%%
lim = 40
plt.figure(4, figsize=(14, 4))
#plt.plot(y[:lim]-np.sqrt(noise_var), color='darkorange')
#plt.plot(y[:lim]+np.sqrt(noise_var), color='darkorange')
#plt.plot(y[:lim]-2*np.sqrt(noise_var), color='orange')
#plt.plot(y[:lim]+2*np.sqrt(noise_var), color='orange')
plt.plot(y[:lim]-3*np.sqrt(noise_var), '--', color='darkorange', label=r'$\pm 3\sigma$ pás')
plt.plot(y[:lim]+3*np.sqrt(noise_var), '--', color='darkorange')
plt.plot(y[:lim], '--', lw=2, color='blue', label='teoretický průběh')
plt.plot(y_noisy[:lim], 'r+', mew=3, lw=2, label='pozorování (se šumem)')

plt.fill_between(
        np.arange(lim),
        y[:lim]+3*np.sqrt(noise_var),
        y[:lim]-3*np.sqrt(noise_var),
        color='whitesmoke'
        )

plt.fill_between(
         -100*norm.pdf(np.linspace(-160, 160, 100), loc=y[0], scale=np.sqrt(noise_var)),
         np.linspace(-160, 160, 100),
         color='blue'
        )
plt.legend()
plt.savefig('/tmp/l2-sigmapas.jpg', bbox_inches='tight')
plt.show()

#%%
Ebeta_log = np.array(regmodel.Ebeta_log)
std_beta_log = np.array(regmodel.var_beta_log)

plt.figure(5, figsize=(15, 6))
plt.subplot(211)
plt.plot(Ebeta_log[:, 0], label=r'$\hat\beta_1 = \hat v_0$')
plt.hlines(beta[0], 0, ndat, 'r', label=r'$\beta_1 = v_0$')
plt.fill_between(np.arange(ndat),
                 Ebeta_log[:, 0] + 3 * std_beta_log[:, 0],
                 Ebeta_log[:, 0] - 3 * std_beta_log[:, 0],
                 color='whitesmoke'
                 )
plt.xlim(0)
plt.ylim(-40, 40)

plt.subplot(212)
plt.plot(Ebeta_log[:, 1], label=r'$\hat\beta_2 = \hat a$')
plt.hlines(beta[1], 0, ndat, 'r', label=r'$\beta_2 = a$')
plt.fill_between(np.arange(ndat),
                 Ebeta_log[:, 1] + 3 * std_beta_log[:, 1],
                 Ebeta_log[:, 1] - 3 * std_beta_log[:, 1],
                 color='whitesmoke'
                 )
plt.ylim(-40, 40)
plt.xlim(0)
plt.xlabel('t')

plt.figure(6, figsize=(15, 6))
plt.subplot(211)
plt.plot(Ebeta_log[:, 0], label=r'$\hat\beta_1 = \hat v_0$')
plt.hlines(beta[0], 0, ndat, 'r', label=r'$\beta_1 = v_0$')
plt.fill_between(np.arange(ndat),
                 Ebeta_log[:, 0] + 3 * std_beta_log[:, 0],
                 Ebeta_log[:, 0] - 3 * std_beta_log[:, 0],
                 color='whitesmoke'
                 )
plt.ylim(0, 8)
plt.xlim(0)
plt.legend()
plt.savefig('/tmp/l2-regrese-Ebeta.jpg', bbox_inches='tight')

plt.subplot(212)
plt.plot(Ebeta_log[:, 1], label=r'$\hat\beta_2 = \hat a$')
plt.hlines(beta[1], 0, ndat, 'r', label=r'$\beta_2 = a$')
plt.fill_between(np.arange(ndat),
                 Ebeta_log[:, 1] + 3 * std_beta_log[:, 1],
                 Ebeta_log[:, 1] - 3 * std_beta_log[:, 1],
                 color='whitesmoke'
                 )
plt.ylim(-.0, 1)
plt.xlim(0)
plt.xlabel('t')
plt.savefig('/tmp/l2-regrese-Ebeta-detail.jpg', bbox_inches='tight')