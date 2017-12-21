from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import glob

def read_coeffs(fname):
	df_coeffs = pd.read_csv(fname, sep=',', dtype='float32')
	df_coeffs.columns = ["index", "angle", "ocean_u", "ocean_v", "A", "coef", "data_num", "mean_ocean_u", "mean_ocean_v", "mean_ice_u", "mean_ice_v"]
	"""
	偏角（度）
	海流u
	海流v
	F
	相関係数
	回帰に使ったデータ数
	平均海流u
	平均海流v
	海氷流速u
	海氷流速v
	"""
	df_coeffs = df_coeffs.drop("index", axis=1)
	ary_coeffs = np.array(df_coeffs)
	angle = ary_coeffs[:, 0]
	ocean_u = ary_coeffs[:, 1]
	ocean_v = ary_coeffs[:, 2]
	A = ary_coeffs[:, 3]
	coef = ary_coeffs[:, 4]
	data_num = ary_coeffs[:, 5]
	mean_ocean_u = ary_coeffs[:, 6]
	mean_ocean_v = ary_coeffs[:, 7]
	mean_ice_u = ary_coeffs[:, 8]
	mean_ice_v = ary_coeffs[:, 9]
	

	#return df_coeffs
	return A


def visual_coeffs(data, latlon145_file_name, points, save_name, show=True):
	"""
	風力係数、偏角、相関係数などの可視化
	"""
	#print (data.columns)
	A_1 = data.loc[:, 1] #month: 1
	import method_map as m_map
	lon, lat = m_map.read_lonlat(latlon145_file_name)

	m = Basemap(lon_0=180,boundinglat=50,
	            resolution='l',projection='npstere')
	fig=plt.figure(figsize=(8,8))

	#グリッドの描画
	"""
	lons, lats = m(lon,lat,inverse=False)
	m.plot(lons,lats,'bo', markersize=0.3)
	"""
	x, y = m(lon, lat)
	x = np.reshape(x, (145,145), order='F')
	y = np.reshape(y, (145,145), order='F')

	m.drawcoastlines(color = '0.15')

	A_1[A_1==999.] = np.nan
	print (A_1.head())
	A_1 = np.array(A_1)
	A_1 = np.reshape(A_1, (145,145), order='F')
	A_1 = np.ma.masked_invalid(A_1)

	m.pcolormesh(x[points], y[points], A_1[points], cmap=plt.cm.jet)
	m.colorbar(location='bottom', format='%.2f')
	
	if show==True:
		plt.show()

	fig.savefig(save_name, dpi=1000)
	plt.clf()
	fig.clf()




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
	#print (data.head())

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





#visual_coeffs(data, latlon145_file_name, points, show=True)
















