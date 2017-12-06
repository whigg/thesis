#風と氷の速度の相関の可視化

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##################################################################
#データの読み込み、整形

#地衡風データの処理
df_wind = pd.read_csv('../data/ecm030101.csv', header=None)
wind = np.array(df_wind, dtype='float32')
w_u = wind[:,0]
w_v = wind[:,1]
w_speed = np.sqrt(w_u*w_u + w_v*w_v)
w_u1 = np.reshape(w_u, (145,145), order = 'F')
w_v1 = np.reshape(w_v, (145,145), order = 'F')
#w_speed1 = np.sqrt(w_u1*w_u1 + w_v1*w_v1)
w_speed1 = w_u1*w_u1 + w_v1*w_v1

#海氷速度データの処理
idx0 = np.zeros(145*145)

df_ice_wind = pd.read_csv('../data/030101.csv', header=None)
w_true = df_ice_wind[df_ice_wind<999.].dropna()
idx_all = range(145*145)
idx_t = np.array(w_true.index)
idx_f = np.sort(list(set(idx_all)-set(idx_t)))

wind = np.array(df_ice_wind, dtype='float32')
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
#speed_true = np.sqrt(u_true*u_true + v_true*v_true)
speed_true = u_true*u_true + v_true*v_true

#緯度、経度情報の処理
df_latlon = pd.read_csv('../data/latlon.csv', header=None)
latlon = np.array(df_latlon, dtype='float32')

lat = latlon[:,2]
lon = latlon[:,3]
lats = np.reshape(lat, (145,145), order = 'F')
lons = np.reshape(lon, (145,145), order = 'F')

###################################################################
#相関を求める
def get_theta(u,v):
	#u,v: Matrix(145*145)
	return np.arctan2(v,u)

def get_wpc(ice_v,wind_v):
	#A,B: Matrix(145*145)
	return (ice_v/wind_v)

def get_diff_theta(i_t,w_t):
	#t1,t2: theta Matrix(145*145)
	return i_t-w_t

#角度
wind_theta = get_theta(w_v1,w_u1)
ice_theta = get_theta(v_true,u_true)
diff_theta = get_diff_theta(ice_theta,wind_theta)

#風力係数
wpc = np.sqrt(get_wpc(speed_true,w_speed1))/100


###################################################################
from matplotlib.colors import LinearSegmentedColormap

def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

#unique_value = set(iris.target)
#print unique_value
# --> [0, 1, 2]

cm = generate_cmap(['#87CEEB', '#2E8B57', '#F4A460'])
"""
fig = plt.figure(figsize=(13,7))
im = plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target, linewidths=0, alpha=.8, cmap=cm)
fig.colorbar(im)
plt.show()
"""

###################################################################
m = Basemap(lon_0=180,boundinglat=50,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(10,10))

g = np.arange(0,145,1)
h = np.arange(0,145,1)
points = np.meshgrid(g, h)

#グリッドの描画
"""
lons, lats = m(lat,lon,inverse=False)
m.plot(lons,lats,'bo', markersize=0.3)
"""

x, y = m(lon, lat)
x1 = np.reshape(x, (145,145), order='F')
y1 = np.reshape(y, (145,145), order='F')

#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
m.drawcoastlines(color = '0.15')

seismic = 'seismic'

#風力係数
m.pcolormesh(x1[points], y1[points], wpc[points], cmap='tab20')
#角度
#m.pcolormesh(x1[points], y1[points], np.absolute(diff_theta)[points], cmap=plt.cm.jet)
#m.pcolormesh(x1[points], y1[points], diff_theta[points], cmap=plt.cm.jet)

#data_interp, x, y = map.transform_scalar(diff_theta.reshape((145*145,1), order='F'), lon, lat, 30, 30, returnxy=True, masked=True)
#m.pcolormesh(x, y, data_interp, cmap='Paired')

#hexbinで可視化
#m.hexbin(x, y, C=diff_theta.reshape((145*145,1), order='F'), reduce_C_function = max, gridsize=100, cmap="seismic")

m.colorbar(location='bottom', format='%.1f')

#コンター図
#m.contour(x1[points], y1[points], diff_theta)

#風の可視化(quiver)
#m.quiver(x1[points], y1[points], w_u1[points], w_v1[points], w_speed1[points])
#m.quiver(x1[points], y1[points], u_true[points], v_true[points], speed_true[points])


plt.show()

