#mask land
#NxN(N=145)のうち、海のところのグリッドをTrueにして、インデックスを返す

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 145 #grid size

df1 = pd.read_csv('data/latlon.csv', header=None)
latlon = np.array(df1, dtype='float32')

lat = latlon[:,2]
lon = latlon[:,3]

###################################################################
m = Basemap(lon_0=180,boundinglat=40,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(8,8))

x, y = m(lon, lat)
ocean_grid = np.array([not(m.is_land(x[i],y[i])) for i in range(N*N)])

print ("Writing to csv...")
np.savetxt('ocean_grid_145.csv', ocean_grid, delimiter=',')

"""
#グリッドの描画
m.plot(x[ocean_grid],y[ocean_grid],'bo', markersize=0.3)
m.drawcoastlines(color = '0.15')

plt.show()
"""
