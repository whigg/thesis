#風のデータの可視化

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df0 = pd.read_csv('ecm030101.csv', header=None)
wind = np.array(df0, dtype='float32')
u = wind[:,0]
v = wind[:,1]
speed = np.sqrt(u*u + v*v)
u1 = np.reshape(u, (145,145), order = 'F')
v1 = np.reshape(v, (145,145), order = 'F')
speed1 = np.sqrt(u1*u1 + v1*v1)

df1 = pd.read_csv('latlon.csv', header=None)
latlon = np.array(df1, dtype='float32')
lon = latlon[:,2]
lat = latlon[:,3]
lons = np.reshape(latlon[:,2], (145,145), order = 'F')
lats = np.reshape(latlon[:,3], (145,145), order = 'F')
###################################################################
m = Basemap(lon_0=180,boundinglat=50,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(8,8))

g = np.arange(0,145,1)
h = np.arange(0,145,1)
points = np.meshgrid(g, h)

#グリッドの描画
"""
lons, lats = m(lat,lon,inverse=False)
m.plot(lons,lats,'bo', markersize=0.3)
"""

x, y = m(lat, lon)
x = np.reshape(x, (145,145), order='F')
y = np.reshape(y, (145,145), order='F')
print (x.shape)

m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
m.drawcoastlines(color = '0.15')

#風の描画
m.quiver(x[points], y[points], 
    u1[points], v1[points], speed1[points])


plt.show()