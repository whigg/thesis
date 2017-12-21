import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd

"""
f = '../visual/data/GW1AM220120702D_IC0NP.dat'
fp = open(f,'rb')
ary = np.fromfile(fp, '<h', -1)
fp.close()

ic0 = np.zeros(900**2)
ic0[ary<0] = np.nan
ic0[ary>=0] = ary[ary>=0]

ic0 = ic0.reshape((900,900),order='F')



fname = 'latlon_low_NP'
f = "<i"
dt = np.dtype([("lon",f), ("lat",f)])
fd = open(fname,"rb")
result = np.fromfile(fd, dtype=dt, count=900*900)
fd.close()

result = np.array([np.array(item) for item in result.tolist()])
"""


"""
    [North Pole]
                                 [Low reso.]    [High reso.]
    Center of upper left  pixel: N35.64 E45.0   N35.61 E45.0
    Center of upper right pixel: N35.64 W45.0   N35.61 W45.0
    Center of lower right pixel: N35.64 W135.0  N35.61 W135.0
    Center of lower left  pixel: N35.64 E135.0  N35.61 E135.0
"""

coverage = np.array([[45.0,35.64],[-45.0,35.64],[-135.0,35.64],[135.0,35.64]])

#############################################################################
m = Basemap(lon_0=180,boundinglat=40,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(10,10))

#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
m.drawcoastlines(color = '0.15')

lons = coverage[:,0]
lats = coverage[:,1]
x,y = m(lons,lats)
x = np.linspace(min(x), max(x), 900)
y = np.linspace(min(y), max(y), 900)
x,y = np.meshgrid(x, y)

m.plot(x,y,'bo', markersize=0.01)
#m.pcolor(x, y, ic0, cmap=plt.cm.jet)
plt.show()

