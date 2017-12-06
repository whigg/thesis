#氷の速度データの可視化

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

idx0 = np.zeros(145*145)

df0 = pd.read_csv('030101.csv', header=None)
w_true = df0[df0<999.].dropna()
a = range(145*145)
idx_t = np.array(w_true.index)
idx_f = np.sort(list(set(a)-set(idx_t)))

"""
a[idx_t] = 1
a[idx_f] = np.nan
"""

wind = np.array(df0, dtype='float32')
u = wind[:,0]
v = wind[:,1]

u_t = idx0
u_t[idx_t] = u[idx_t]
u_t[idx_f] = np.nan
v_t = idx0
v_t[idx_t] = v[idx_t]
v_t[idx_f] = np.nan

u_true = np.reshape(u_t, (145,145), order = 'F')
v_true = np.reshape(v_t, (145,145), order = 'F')
speed_true = np.sqrt(u_true*u_true + v_true*v_true)
"""
u1 = np.reshape(u, (145,145), order = 'F')
v1 = np.reshape(v, (145,145), order = 'F')

speed1 = np.sqrt(u1*u1 + v1*v1)
"""

df1 = pd.read_csv('latlon.csv', header=None)
latlon = np.array(df1, dtype='float32')
#a = latlon[:,0]
#b = latlon[:,1]

lat = latlon[:,2]
lon = latlon[:,3]
lats = np.reshape(latlon[:,2], (145,145), order = 'F')
lons = np.reshape(latlon[:,3], (145,145), order = 'F')
###################################################################
m = Basemap(lon_0=180,boundinglat=50,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(8,8))

g = np.arange(0,145,1)
h = np.arange(0,145,1)
points = np.meshgrid(g, h)

#グリッドの描画
"""
lons, lats = m(lon,lat,inverse=False)
m.plot(lons,lats,'bo', markersize=0.3)
"""

x, y = m(lon, lat)
x = np.reshape(x, (145,145), order='F')
y = np.reshape(y, (145,145), order='F')
print (x.shape)

m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
m.drawcoastlines(color = '0.15')

#風の描画
m.quiver(x[points], y[points], 
    u_true[points], v_true[points], speed_true[points])


plt.show()

