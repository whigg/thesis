from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
色々な関数群（テンプレートとして使う？）
データの読み込み、整形
時系列データ用なのでreshapeはしない/reshapeはreshape_m関数を使う
風の大きさは2乗したものを返す
"""
##################################################################
def load_csv(fname):
    return pd.read_csv(fname, header=None)

def reshape_m(v, order='F'):
    return np.reshape(v, (145,145), order = 'F')

##################################################################
#地衡風データの処理
def read_wind(fname):
    fname = 'ecm030101.csv'
    df_wind = load_csv()
    wind = np.array(df_wind, dtype='float32')
    w_u = wind[:,0]
    w_v = wind[:,1]
    w_speed = w_u*w_u + w_v*w_v

    return w_u, w_v, w_speed

#海氷速度データの処理
def read_ice_v(fname):
    fname = '030101.csv'
    idx0 = np.zeros(145*145)

    df_ice_wind = load_csv(fname)
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
    speed_t = u_t*u_t+v_t*v_t

    return u_t, v_t, speed_t

#緯度、経度情報の処理
#145*145のグリッドのlatlon
def get_latlon():
    df_latlon = load_csv('latlon.csv')
    latlon = np.array(df_latlon, dtype='float32')
    lat = latlon[:,2]
    lon = latlon[:,3]
    #lats = reshape_m(lat)
    #lons = reshape_m(lon)
    return lat, lon

###################################################################
#2変数の関係を求める関数群

def get_theta(u,v):
	return np.arctan(v/u)

def get_corr(A,B):
	return (0)

def get_diff_theta(i_t,w_t,ab=False):
	#t1,t2: theta Matrix
    if ab==False: return i_t-w_t
	else: return np.absolute(i_t-w_t)

"""
wind_theta = get_theta(w_v1,w_u1)
ice_theta = get_theta(v_true,u_true)
diff_theta = get_diff_theta(ice_theta,wind_theta)
"""

###################################################################
#以下、本質的ではないオプション

#カラーマップの設定
#https://qiita.com/kenmatsu4/items/fe8a2f1c34c8d5676df8
from matplotlib.colors import LinearSegmentedColormap

def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)
#cm = generate_cmap(['#87CEEB', '#2E8B57', '#F4A460'])

#basemapでの描画
#建設中
def drawmap(lat,lon,draw_grid=False):
    m = Basemap(lon_0=180,boundinglat=50,resolution='l',projection='npstere')
    fig=plt.figure(figsize=(10,10))

    g = np.arange(0,145,1)
    h = np.arange(0,145,1)
    points = np.meshgrid(g, h)

    #グリッドの描画
    if draw_grid==True:
        lons, lats = m(lat,lon,inverse=False)
        m.plot(lons,lats,'bo', markersize=0.3)

    x, y = m(lon, lat)
    x1 = np.reshape(x, (145,145), order='F')
    y1 = np.reshape(y, (145,145), order='F')

    #m.drawmapboundary(fill_color='aqua')
    #m.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
    m.drawcoastlines(color = '0.15')

    m.pcolormesh(x1[points], y1[points], diff_theta[points], cmap='seismic')
    #data_interp, x, y = map.transform_scalar(diff_theta.reshape((145*145,1), order='F'), lon, lat, 30, 30, returnxy=True, masked=True)
    #m.pcolormesh(x, y, data_interp, cmap='Paired')

    #m.hexbin(x, y, C=diff_theta.reshape((145*145,1), order='F'), reduce_C_function = max, gridsize=100, cmap="seismic")
    m.colorbar(location='bottom', format='%.1f')

    #m.contour(x1[points], y1[points], diff_theta)

    m.quiver(x1[points], y1[points], w_u1[points], w_v1[points], w_speed1[points])
    #m.quiver(x1[points], y1[points], u_true[points], v_true[points], speed_true[points])

    plt.show()

