import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cm

m = np.array([0, 0])
covs = [np.eye(2),
        np.array([[.5, -1.], [-1., 3]]),
        np.array([[2, 1], [1, 2]])]
#%%
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
xv, yv = plt.meshgrid(x, y)

zs = []
for c in covs:
    zs.append(mvn.pdf(np.c_[xv.flatten(), yv.flatten()], 
                           mean=m, cov=c).reshape(xv.shape))



#%%
#ls = LightSource(270, 45)

fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection='3d'),
                        figsize=(11,3))
for ax, z, c in zip(axs, zs, covs):
#    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(xv, yv, z, rstride=1, cstride=1, cmap=cm.coolwarm,#facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    ax.set_zlim(0, .2)
    ax.set_title(np.array_str(c))
fig.tight_layout()
plt.savefig('l1-2dgauss-surface.jpg', bbox_inches='tight')

#%%
fig, axs = plt.subplots(1, 3, figsize=(11,3))
for ax, z, c in zip(axs, zs, covs):
    ax.contour(xv, yv, z, 20, cmap=cm.coolwarm)
#plt.tight_layout()
plt.savefig('l1-2dgauss-contour.jpg', bbox_inches='tight')