import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd


f = '../data/GW1AM220131111A_100IC0NP.dat'
fp = open(f,'rb')
ary = np.fromfile(fp, '<h', -1)
fp.close()

ic0 = np.zeros(900**2)
ic0[ary<0] = np.nan
ic0[ary>=0] = ary[ary>=0]
"""
ic0 = ic0.reshape((900,900))
ic0 = ic0[-1::-1]
"""
ic0_ = ic0.reshape((900,900))[-1::-1].T[-1::-1].T




"""
    [North Pole]
                                 [Low reso.]    [High reso.]
    Center of upper left  pixel: N35.64 E45.0   N35.61 E45.0
    Center of upper right pixel: N35.64 W45.0   N35.61 W45.0
    Center of lower right pixel: N35.64 W135.0  N35.61 W135.0
    Center of lower left  pixel: N35.64 E135.0  N35.61 E135.0
"""

coverage = np.array([[45.0,35.64],[-45.0,35.64],[135.0,35.64],[-135.0,35.64]])

#############################################################################

m = Basemap(lon_0=180,boundinglat=40,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(6,6))

#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
m.drawcoastlines(color = '0.15')


lons = coverage[:,0]
lats = coverage[:,1]
x,y = m(lons,lats)
x = np.linspace(min(x), max(x), 900)[::-1]
y = np.linspace(min(y), max(y), 900)
xx,yy = np.meshgrid(x, y)
#print (xx.ravel().shape)
#grids = np.vstack([xx.ravel(), yy.ravel()]).T
#print (grids.shape)
grids = np.vstack([xx.ravel(), yy.ravel()]).T[-1::-1]
x_ = grids[:,0]
y_ = grids[:,1]




from matplotlib import colors as c
cMap = c.ListedColormap(['g','b','w'])
#m.plot(grids[:,0][:1000], grids[:,1][:1000],'bo', markersize=0.1)
#m.plot(xx.ravel()[-1::-1][:1000],yy.ravel()[-1::-1][:1000],'bo', markersize=0.1)
#m.plot(x_[:1000],y_[:1000],'bo', markersize=0.1)

#m.pcolor(grids[:,0].reshape((900,900)), grids[:,1].reshape((900,900)), ic0_, cmap=cMap)
#m.pcolor(grids[:,0], grids[:,1], ic04girds.reshape((900*900,1)), cmap=cMap)
m.pcolormesh(xx, yy, ic0_, cmap=cMap)
#m.pcolormesh(x_[:1000], y_[:1000], ic0[:1000], cmap=cMap)


m.colorbar(location='bottom', format='%.1f')

plt.show()


data = data = np.c_[x_, y_, ic0]
#print (data)

np.savetxt('ic0_exp.csv',data,delimiter=',')


