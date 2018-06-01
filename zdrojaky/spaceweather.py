import sys
sys.path.insert(0, '../zdrojaky/')

import numpy as np
import matplotlib.pylab as plt
from nig import NiG

#file = 'ftp://ftp.swpc.noaa.gov/pub/lists/particle/20180224_Gs_part_5m.txt'
datafile = '20180224_Gs_part_5m.txt'

data = np.genfromtxt(datafile, skip_header=26)
e2 = data[:,13]
ndat = e2.size

xi0 = np.diag([1000, .1, .1])
nu0 = 5.
regmodel = NiG(xi0, nu0)

forg_factor = .95

yt_pred = np.zeros(ndat)


for t in range(3, ndat):
    xt = np.array([1, e2[t-1]])
    yt = e2[t]
    yt_pred[t] = regmodel.Ebeta.dot(xt)
    
    # zapomínání
    regmodel.xi *= forg_factor
    regmodel.nu *= forg_factor
    
    # update
    regmodel.update(yt, xt)
    regmodel.log()

#%%
Ebeta_log = np.array(regmodel.Ebeta_log)

plt.figure(1, figsize=(15, 5))
plt.plot(e2)
plt.plot(yt_pred, '+')

plt.figure(2, figsize=(15, 5))
plt.subplot(3, 1, 1)
plt.plot(Ebeta_log[:,0])
plt.subplot(3, 1, 2)
plt.plot(Ebeta_log[:,1])
#plt.subplot(3, 1, 3)
#plt.plot(Ebeta_log[:,2])
#%%
errors = e2[3:] - yt_pred[3:]
plt.figure(3)
plt.hist(errors, bins=100)
print('RMSE: ', np.sqrt(np.mean((e2[3:] - yt_pred[3:])**2)))

#%%
plt.figure(4)
plt.boxplot(errors, showfliers=False)