import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd

coverage = np.array([[45.0,35.64],[-45.0,35.64],[135.0,35.64],[-135.0,35.64]])

m = Basemap(lon_0=180,boundinglat=40,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(7,7))
m.drawcoastlines(color = '0.15')

#########################################################################
#900x900グリッドの読み込み
lons = coverage[:,0]
lats = coverage[:,1]
x,y = m(lons,lats)
x = np.linspace(min(x), max(x), 900)[::-1]
y = np.linspace(min(y), max(y), 900)
xx,yy = np.meshgrid(x, y)

grids = np.vstack([xx.ravel(), yy.ravel()]).T[-1::-1]
x_900 = grids[:,0]
y_900 = grids[:,1]

#########################################################################
#145x145グリッドの読み込み
df1 = pd.read_csv('../data/latlon.csv', header=None)
latlon = np.array(df1, dtype='float32')
lon = latlon[:,3]
lat = latlon[:,2]

x_145,y_145 = m(lon,lat)

#########################################################################
#グリッド調整
initial_x = np.argmin(np.absolute(x_900[:900]-x_145[0]))
#initial_x = 20
initial_y = np.argmin(np.absolute(y_900.reshape((900,900))[:,0]-y_145[0]))
#initial_y = 20

a = np.arange(20,20+6*145,6)
x,y = np.meshgrid(a,a)
grids = np.vstack([y.ravel(), x.ravel()]).T
grid_900 = np.reshape(range(900*900),(900,900))
#print (grid_900[grids.tolist()])
grid_idx = [grid_900[grids[i][0], grids[i][1]] for i in range(145*145)]
#np.savetxt("grid900to145.csv", grid_idx, delimiter=',')
print ("Saved grid index.")
#########################################################################
#プロット

m.plot(x_900[grid_idx][:200],y_900[grid_idx][:200],'bo', markersize=0.05)
m.plot(x_145,y_145,'ro', markersize=0.1)

plt.show()
