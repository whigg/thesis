"""
main
"""
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import glob

import calc_data
import visualize


#iMac用
latlon145_file_name = "../data/" + "latlon_ex.csv"
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
latlon145_file_name = "../data/" + "latlon_ex.csv"
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
m_map.visual_wind(wind_file_name, latlon145_file_name, points, show=True)
#m_map.visual_ice_wind(ice_file_name, latlon145_file_name, points, show=True)
#m_map.visual_ic0_900(ic0_file_name, latlon900_file_name, show=True)
#m_map.visual_ic0_145(ic0_file_name, latlon145_file_name, grid900to145_file_name, show=True)
#m_map.visual_2winds(wind_file_name, ice_file_name, latlon145_file_name, points, v_ratio=True, show=True)




#method_dateを使った可視化
"""
filetype = ["wind", "ice"]
start = 20030101
end = 20030410

data = m_d.read_ts_file(filetype, start, end, date_col=True)
m_d.visualize(data)
"""


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



latlon145_file_name = "../data/" + "latlon.csv"

g = np.arange(0,145,1)
h = np.arange(0,145,1)
points = np.meshgrid(g, h)



"""
spanとyearからmonthごとにとってくる
つまり、ある指定の年の月の時系列
"""
span = 30
year = np.arange(2003,2013)
file_path = sorted(glob.glob("../getA/" + str(span) + '/*/*.csv'))
N = len(year)
for i, yy in enumerate(year):
	print ("{}/{}: {}".format(i+1, N, yy))
	#data = np.arange(0, 145*145)
	data = np.zeros(145*145)
	for filename in file_path:
		str_year = str(yy)[2:]
		if filename.find(str_year) != -1:
			print ("\tspan: {}, year: {}, month: {}".format(span, yy, filename[11]))
			data_vec = read_coeffs(filename)
			data = np.c_[data, data_vec]

	data = pd.DataFrame(data)
	data = data.drop(data.columns[[0]], axis=1)
	month_list = (np.arange(6)+1).tolist()
	column_name = month_list
	data.columns = column_name
	print (data.head())

	save_name = str(yy) + "_" + str(span) + ".png"
	visual_coeffs(data, latlon145_file_name, points, show=False, save_name=save_name)


"""
spanとmonthからyearごとにとってくる
つまり、ある指定の月の年の時系列
"""
"""
span = 30
month = 1
file_path = sorted(glob.glob("../getA/" + str(span) + '/*/*.dat'))

data = np.zeros(145*145)
for filename in file_path:
	str_month = str(month)
	if filename[11] == str_month:
		print ("span: {}, month: {}, year: {}".format(span, month, str(20)+filename[25:27]))
		data_vec = read_coeffs(filename)
		data = np.c_[data, data_vec]

data = pd.DataFrame(data)
data = data.drop(data.columns[[0]], axis=1)
year_list = np.arange(2003, 2014).tolist()
data.columns = year_list
print (data.head())
"""








































