#時系列データの可視化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_ts_csv(fname):
    return np.array(pd.read_csv(fname, header=None))

def reshape_m(v, order='F'):
    return np.reshape(v, (145,145), order = 'F')

#緯度、経度情報の処理
#145*145のグリッドのlatlon
def get_latlon():
    df_latlon = load_csv('../data/latlon.csv')
    latlon = np.array(df_latlon, dtype='float32')
    lat = latlon[:,2]
    lon = latlon[:,3]
    return lat, lon

###################################################################
#2変数の関係を求める関数群

def get_theta(u,v):
	return np.arctan2(v,u)

def get_corr(A,B):
	return (0)

"""
def get_diff_theta(i_t,w_t,ab=False):
	#t1,t2: theta Matrix
    if ab==False:
        return i_t-w_t
	else:
        return np.absolute(i_t-w_t)
"""
"""
wind_theta = get_theta(w_v1,w_u1)
ice_theta = get_theta(v_true,u_true)
diff_theta = get_diff_theta(ice_theta,wind_theta)
"""

def plot_data(x,data,color='b',marker="."):
    plt.plot(x,data,color=color, marker=marker)

###################################################################
#メインの処理

#とりあえずts_w_uの2003を読み込み
ts_w_u_2003 = load_ts_csv("./ts_w_u/ts_w_u_2003.csv")
#print (ts_w_u_2003.shape)
ocean_grid = load_ts_csv("../data/ocean_grid_145.csv")
#print (ocean_grid.ravel())


idx = ocean_grid.ravel().tolist()
idx.insert(0,1)
idx = np.array(idx)
idx = np.where(idx==True)[0]
#print (type(idx))

"""
d1, d2 = datetime(2003, 1, 1), datetime(2003,6, 30)
L = (d2-d1).days+1
dt = d1
days_all = []
for i in range(L):
    days_all.append(int(str(dt)[:10].replace('-', '')))
    dt = dt + timedelta(days=1)
"""


tmp = ts_w_u_2003[:,idx]
#print (tmp.shape)
x = range(len(tmp[:,0]))
y = np.mean(tmp[:,1:], axis=1)
z = np.std(tmp[:,1:], axis=1)

plt.figure(figsize=(12,8))

plt.plot(x, y, color='k')
plt.plot(x, y+z, color='b', linestyle='--')
plt.plot(x, y-z, color='b', linestyle='--')

plt.plot(x,tmp[:,2],color='g')


"""
plt.plot(x,tmp[:,3],color='b')
plt.plot(x,tmp[:,4],color='k')
plt.plot(x,tmp[:,5],color='c')
"""

#x軸の共通化
#角度のデータ

"""
gen_ts_dataで取り出し方に問題がないか確認
このコードで取り出したやつと、既に取り出したやつで同じになるか確認
取り出したやつの確認(ice_wind)

風は場所によってバラツキが大きい
    同じ動きをしているものをクラスタリングできないか？
    kmeansでk=10くらいでやってみて、10個のグラフとグリッドを出力してみる
"""


plt.show()
















