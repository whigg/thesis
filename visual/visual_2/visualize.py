
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import calc_data



#####################################################################
#Basemapによる地図投影

def visual_ic0_900(ic0_file_name, latlon900_file_name):
	#氷のIC0データの可視化（csvからの読み込み）
	df0 = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(df0, dtype='float32')
	ic0 = np.ma.masked_invalid(ic0)
	ice_grid = np.reshape(ic0, (900,900))
	
	df1 = pd.read_csv(latlon900_file_name, header=None)
	latlon = np.array(df1, dtype='float32')
	x_lon = latlon[:,0]
	y_lat = latlon[:,1]
	
	m = Basemap(lon_0=180,boundinglat=40, resolution='l',projection='npstere')
	fig=plt.figure(figsize=(6,6))
	
	#グリッドの描画
	#m.plot(x_lon,y_lat,'bo', markersize=0.3)
	xx = np.reshape(x_lon, (900,900))
	yy = np.reshape(y_lat, (900,900))
	
	m.drawcoastlines(color = '0.15')
	m.pcolormesh(xx, yy, ice_grid, cmap=plt.cm.jet)
	m.colorbar(location='bottom')



def visual_2winds(wind_file_name, ice_file_name, latlon145_file_name, points, 
	v_ratio=False, v_difftheta=False, threshold=0.05, show=True):
	"""
	風と氷の速度の相関の可視化
	"""
	from matplotlib.colors import LinearSegmentedColormap
	def generate_cmap(colors):
		#自分で定義したカラーマップを返す
		values = range(len(colors))
	
	vmax = np.ceil(np.max(values))
	color_list = []
	for v, c in zip(values, colors):
		color_list.append( ( v/ vmax, c) )
	return LinearSegmentedColormap.from_list('custom_cmap', color_list)
	cm = generate_cmap(['#87CEEB', '#2E8B57', '#F4A460'])
	
	w_u1, w_v1, w_speed1 = calc_data.get_1day_w_data(wind_file_name)
	u_true, v_true, speed_true = calc_data.get_1day_ice_data(ice_file_name)
	lon, lat = calc_data.get_lonlat(latlon145_file_name)
	
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
	#風力比
	wpc = get_wpc(speed_true,w_speed1)/100
	wpc[wpc>threshold] = np.nan
	wpc_high = get_wpc(speed_true,w_speed1)/100
	wpc_high[wpc_high<=threshold] = np.nan
	
	m = Basemap(lon_0=180,boundinglat=50, resolution='l',projection='npstere')
	fig=plt.figure(figsize=(6,6))
	
	x, y = m(lon, lat)
	x1 = np.reshape(x, (145,145), order='F')
	y1 = np.reshape(y, (145,145), order='F')
	
	m.drawcoastlines(color = '0.15')
	if v_ratio == True:
		wpc = np.reshape(wpc, (145,145), order='F')
		wpc = np.ma.masked_invalid(wpc)
		wpc_high = np.reshape(wpc_high, (145,145), order='F')
		wpc_high = np.ma.masked_invalid(wpc_high)
		
		#m.pcolormesh(x1[points], y1[points], wpc_high[points], cmap='tab10')
		#m.plot(x1[points], y1[points], 'bo', markersize=2)
		m.pcolormesh(x1[points], y1[points], wpc[points], cmap=plt.cm.jet)
		m.colorbar(location='bottom')

	elif v_difftheta == True:
		diff_theta = np.reshape(diff_theta, (145,145), order='F')
		#m.pcolormesh(x1[points], y1[points], np.absolute(diff_theta)[points], cmap=plt.cm.jet)
		m.pcolormesh(x1[points], y1[points], diff_theta[points], cmap=plt.cm.jet)
		m.colorbar(location='bottom')

	#hexbinで可視化
	#seismic = 'seismic'
	#m.hexbin(x, y, C=diff_theta.reshape((145*145,1), order='F'), reduce_C_function = max, gridsize=100, cmap="seismic")
	#m.colorbar(location='bottom', format='%.1f')

	#コンター図
	#m.contour(x1[points], y1[points], diff_theta)
	#風の可視化(quiver)
	#m.quiver(x1[points], y1[points], w_u1[points], w_v1[points], w_speed1[points])
	#m.quiver(x1[points], y1[points], u_true[points], v_true[points], speed_true[points])

	if show==True:
		plt.show()


def visual_coeffs(data, mode, submode, latlon145_file_name, points, save_name, show=True):
	"""
	風力係数、偏角、相関係数などの可視化
	data: regression data
	"""
	lon, lat = calc_data.get_lonlat(latlon145_file_name, array=True)
	lonlat_data = [lon, lat]
	#print (data.columns)

	if mode == 0:
		if submode == 0:
			data = data.loc[:, ["data_idx", "w_u", "w_v", "w_speed", "Label", "Name"]]
			data_type = "type_wind"
		elif submode == 1:
			data = data.loc[:, ["data_idx", "iw_u", "iw_v", "iw_speed", "Label", "Name"]]
			data_type = "type_wind"
		elif submode == 2:
			data = data.loc[:, ["data_idx", "mean_ocean_u", "mean_ocean_v", "Label", "Name"]]
			data_type = "type_wind"
			mean_ocean_u = np.array(data.iloc[:,1])
			mean_ocean_v = np.array(data.iloc[:,2])
			mean_ocean_speed = np.sqrt(mean_ocean_u*mean_ocean_u + mean_ocean_v*mean_ocean_v)
		elif submode == 3:
			data = data.loc[:, ["data_idx", "real_iw_u", "real_iw_v", "real_iw_speed", "Label", "Name"]]
			data_type = "type_wind"
		elif submode == 4:
			data = data.loc[:, ["data_idx", "A_by_day", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == 5:
			data = data.loc[:, ["data_idx", "theta_by_day", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == 6:
			data = data.loc[:, ["data_idx", "ic0_145", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == 7:
			data = data.loc[:, ["data_idx", "A", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == 8:
			data = data.loc[:, ["data_idx", "angle", "Label", "Name"]]
			data_type = "type_non_wind"
		elif submode == 99:
			#以降，ここにcustom計算のdataを追加していく
			data = data
			data_type = "type_other"

		visual_all_by_map(data, data_type, lonlat_data, points)

	elif mode == 1:
		ic0_file_name, latlon900_file_name = data[0], data[1]
		visual_ic0_900(ic0_file_name, latlon900_file_name)
	
	if show == True:
		plt.show()
	if save_name is not None:
		fig.savefig(save_name, dpi=1200)
	
	#plt.clf()
	#fig.clf()



def visual_all_by_map(data, data_type, lonlat_data, points):
	#quiver, pcolormesh全てできるようにする
	lon, lat = lonlat_data[0], lonlat_data[1]
	m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
	fig=plt.figure(figsize=(7.5, 7.5))

	x, y = m(lon, lat)
	x1 = np.reshape(x, (145,145), order='F')
	y1 = np.reshape(y, (145,145), order='F')
	
	m.drawcoastlines(color = '0.15')

	if data_type == "type_wind":
		#data column: ["data_idx", "w_u", "w_v", "w_speed", "Label", "Name"]
		data[data.data_idx==0.] = np.nan

		vector_u = np.array(data.iloc[:,1])
		vector_v = np.array(data.iloc[:,2])
		vector_speed = np.array(data.iloc[:,3])
		
		vector_u = np.ma.masked_invalid(vector_u)
		vector_v = np.ma.masked_invalid(vector_v)
		vector_speed = np.ma.masked_invalid(vector_speed)
		
		vector_u1 = np.reshape(vector_u, (145,145), order='F')
		vector_v1 = np.reshape(vector_v, (145,145), order='F')
		vector_speed1 = np.reshape(vector_speed, (145,145), order='F')
		
		#風の描画
		m.quiver(x1[points], y1[points], vector_u1[points], vector_v1[points], vector_speed1[points])

	elif data_type == "type_non_wind":
		#data column: ["data_idx", "ic0_145", "Label", "Name"]
		data[data.data_idx==0.] = np.nan

		plot_data = np.array(data.iloc[:,1])
		#ここに微調整を書く
		#plot_data[plot_data>=0.05] = np.nan
		#plot_data = np.absolute(plot_data)
		#plot_data[(plot_data>50) | (plot_data<-50)] = np.nan
		#print(set(plot_data.tolist()))

		plot_data = np.ma.masked_invalid(plot_data)
		
		plot_data1 = np.reshape(plot_data, (145,145), order='F')
		
		m.pcolormesh(x1, y1, plot_data1, cmap=plt.cm.jet)
		m.colorbar(location='bottom')

	else:
		plot_data = np.array(data.iloc[:,1])
		plot_data = np.ma.masked_invalid(plot_data)
		m.scatter(x, y, marker='o', color = "b", s=1.2, alpha=0.9)





#############################################################################################
#時系列ではないプロット

def visual_1day_all_2d(data, mode, save_name, show=True):
	"""
	ある日の地衡風-海流速度と海氷流速の可視化
	calc_dataのget_wind_ic0_regression_data関数から接続
	data: calc_data.get_wind_ic0_regression_data(...)
	"""

	#modeの処理
	plot_type, plot_code = mode[0], mode[1]
	data = data.dropna()
	#print (data.head(3))
	
	#プロット
	sns.set_style("darkgrid")
	if plot_type == "scatter":
		if plot_code == 0:
			#sns.jointplot(x="w_speed", y="iw_speed", data=data)
			#sns.jointplot(x="w_speed", y="iw_speed", data=data, kind="reg")
			#sns.jointplot(x="w_speed", y="iw_speed", data=data, kind="kde")
			#sns.jointplot(x="w_speed", y="iw_speed", data=data, kind="hex")
			sns.regplot(x="w_speed", y="iw_speed", data=data)
			sns.regplot(x="w_speed", y="real_iw_speed", data=data)
		elif plot_code == 1:
			#sns.jointplot(x=data.ic0_145[data.ic0_145>=80], y=data.A_by_day[data.ic0_145>=80], kind="reg")
			#sns.jointplot(x=data.ic0_145[data.A_by_day<0.05], y=data.A_by_day[data.A_by_day<0.05], kind="reg")
			sns.jointplot(x=data.ic0_145[data.ic0_145>=80], y=data.A[data.ic0_145>=80], kind="reg")
			#sns.jointplot(x=data.ic0_145[data.ic0_145<=80], y=data.angle[data.ic0_145<=80], kind="reg")
			#sns.jointplot(x="ic0_145", y="A", data=data, kind="reg")
			#sns.jointplot(x="ic0_145", y="A_by_day", data=data, kind="reg")
			#sns.regplot(x="A_by_day", y="ic0_145", data=data)
			#sns.regplot(x="theta_by_day", y="ic0_145", data=data)
	elif plot_type == "hist":
		if plot_code == 0:
			sns.distplot(data["A_by_day"])
		elif plot_code == 1:
			sns.distplot(data["theta_by_day"])
		elif plot_code == 2:
			sns.distplot(data["ic0_145"])
		elif plot_code == 99:
			#カスタム処理
			sns.distplot(data["A_by_day"])

	elif plot_type == "custom":
		if plot_code == 0:
			sns.distplot(data["A_by_day"])
	#dataがクラスタでラベルづけされていた場合
	#TODO: 引数も変更


	if show==True:
		plt.show()
	if save_name is not None:
		fig.savefig(save_name, dpi=1200)

	#plt.clf()
	#fig.clf()



#############################################################################################
#時系列プロット

def visual_ts(data):
	"""
	DataFrame型のdataをプロットする
	
	[参考]
	http://sinhrks.hatenablog.com/entry/2015/11/15/222543
	"""
	
	#普通の時系列プロット
		
	tmp = pd.to_datetime(data["date"])
	data["date"] = data.index
	data.index = tmp
	data = data.rename(columns={'date': 'idx'})
	
	#data[["wind", "ice"]].plot(figsize=(16,4), alpha=0.5)
	"""
	ax = data.wind.plot(figsize=(16,4), ylim=(0, 30), color="blue" )
	ax2 = ax.twinx()
	data.ice.plot( ax=ax2, ylim=(0, 0.8), color="red" )
	"""
	
	
	plt.show()
	
	#時間軸が違う場合(share axis)




#############################################################################################	










