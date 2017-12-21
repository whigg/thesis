import numpy as np
import pandas as pd

import method_map as m_map
import method_sns_graph as m_sns
import method_date as m_d


#iMac用
latlon145_file_name = "../data/" + "latlon.csv"
latlon900_file_name = "../data/IC0_csv/" + "latlon_info.csv"
grid900to145_file_name = "../data/" + "grid900to145.csv"
ocean_grid_file = "../data/ocean_grid_145.csv"
ocean_grid_145 = pd.read_csv(ocean_grid_file, header=None)
ocean_idx = np.array(ocean_grid_145[ocean_grid_145==1].dropna().index)

wind_file_name = "../data/wind_data/" + "ecm080621.csv"
ice_file_name = "../data/ice_wind_data/" + "140301.csv"
ic0_file_name = "../data/IC0_csv/" + "220131111A.csv"



"""
#macbook pro用
latlon145_file_name = "../data/" + "latlon.csv"
latlon900_file_name = "../data/" + "latlon_info.csv"
grid900to145_file_name = "../data/" + "grid900to145.csv"
ocean_grid_file = "../data/ocean_grid_145.csv"
ocean_grid_145 = pd.read_csv(ocean_grid_file, header=None)
ocean_idx = np.array(ocean_grid_145[ocean_grid_145==1].dropna().index)

wind_file_name = "../data/" + "ecm080621.csv"
ice_file_name = "../data/" + "140301.csv"
ic0_file_name = "../data/" + "220131111A.csv"
"""




g = np.arange(0,145,1)
h = np.arange(0,145,1)
points = np.meshgrid(g, h)


#method_mapを使った可視化
#m_map.visual_wind(wind_file_name, latlon145_file_name, points, show=True)
#m_map.visual_ice_wind(ice_file_name, latlon145_file_name, points, show=True)
#m_map.visual_ic0_900(ic0_file_name, latlon900_file_name, show=True)
#m_map.visual_ic0_145(ic0_file_name, latlon145_file_name, grid900to145_file_name, show=True)
#m_map.visual_2winds(wind_file_name, ice_file_name, latlon145_file_name, points, v_ratio=True, show=True)




#method_dateを使った可視化
filetype = ["wind", "ice"]
start = 20030101
end = 20030410

data = m_d.read_ts_file(filetype, start, end, date_col=True)
m_d.visualize(data)



#method_sns_graphを使った可視化
#a = m_sns.visual_w_i_1day(wind_file_name, ice_file_name, ocean_idx, show=True)


#風力比のヒストグラムを表示
"""
wi_ratio = (a[:,6]/a[:,3])/100
plt.hist(wi_ratio, bins=400, range=(0, 0.25))
plt.show()
"""

#3D描画
"""
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.plot(a[:,1], a[:,2], a[:,3], "o", color="#00cccc", ms=4, mew=0.5)
plt.show()
"""


"""
#指定したインデックスだけ地図上にプロットしたいとき
#例えば、↓
idx = np.array(a[:,0].tolist(), dtype='int64')
idx = idx[wi_ratio>=0.2]

m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
fig=plt.figure(figsize=(8,8))

#グリッドの描画
df_latlon = pd.read_csv(latlon145_file_name, header=None)
latlon = np.array(df_latlon, dtype='float32')
lat = latlon[:,2]
lon = latlon[:,3]
lons, lats = m(lon[idx], lat[idx], inverse=False)
m.plot(lons,lats,'bo', markersize=2)

m.drawcoastlines(color = '0.15')

plt.show()
"""





#カスタマイズ










