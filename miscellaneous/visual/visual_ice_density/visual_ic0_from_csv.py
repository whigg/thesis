"""
氷のデータの可視化（csvからの読み込み）
csvのlon, latはポーラーステレオ座標なので注意
reshapeするときはorder='F'では「ない」ので，注意
plotは、3色でやると変になるのでplt.cm.jetでやっている
"""

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df0 = pd.read_csv('./IC0_csv/220131111A.csv', header=None)
ic0 = np.array(df0, dtype='float32')
ice_grid = np.reshape(ic0, (900,900))


"""
#実験
df0 = pd.read_csv('ic0_exp.csv', header=None)
ic0 = np.array(df0, dtype='float32')[:,2]
ice_grid = np.reshape(ic0, (900,900))
"""

df1 = pd.read_csv('./IC0_csv/latlon_info.csv', header=None)
latlon = np.array(df1, dtype='float32')
x_lon = latlon[:,0]
y_lat = latlon[:,1]
###################################################################
m = Basemap(lon_0=180,boundinglat=40,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(6,6))

#グリッドの描画
#m.plot(x_lon,y_lat,'bo', markersize=0.3)

xx = np.reshape(x_lon, (900,900))
yy = np.reshape(y_lat, (900,900))

m.drawcoastlines(color = '0.15')


from matplotlib import colors as c
cMap = c.ListedColormap(['g','b','w'])

#m.pcolormesh(xx, yy, ice_grid, cmap=cMap)
m.pcolormesh(xx, yy, ice_grid, cmap=plt.cm.jet)
m.colorbar(location='bottom', format='%.1f')


plt.show()

