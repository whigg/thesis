#データを読み込み、main_v.pyに返す関数群
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, date, timezone, timedelta

import basic_file as b_f

latlon145_file_name = b_f.latlon145_file_name
latlon900_file_name = b_f.latlon900_file_name
grid900to145_file_name = b_f.grid900to145_file_name
ocean_grid_file = b_f.ocean_grid_file
ocean_grid_145 = b_f.ocean_grid_145
ocean_idx = b_f.ocean_idx

########################################################################################

def get_lonlat(latlon145_file_name, array=False):
	#緯度、経度情報の読み込み
	#df_latlon = pd.read_csv(latlon145_file_name, header=None)
	#latlon_exはheaderありなので注意
	df_latlon = pd.read_csv(latlon145_file_name)
	if array == False:
		return df_latlon.Lon, df_latlon.Lat
	else:
		lon = np.array(df_latlon.Lon)
		lat = np.array(df_latlon.Lat)
		return lon, lat

def get_lonlat_labels(latlon145_file_name):
	df_latlon = pd.read_csv(latlon145_file_name)
	data = df_latlon.loc[:,["Label", "Name"]]
	return data

def get_1day_w_data(wind_file_name):
	df_wind = pd.read_csv(wind_file_name, header=None)
	wind = np.array(df_wind, dtype='float32')
	w_u = wind[:,0]
	w_v = wind[:,1]
	w_speed = np.sqrt(w_u*w_u + w_v*w_v)
	return pd.DataFrame({"w_u": w_u, "w_v": w_v, "w_speed": w_speed})

def get_1day_ice_data(ice_file_name):
	df_ice_wind = pd.read_csv(ice_file_name, header=None)
	w_true = df_ice_wind[df_ice_wind<999.].dropna()
	idx_all = range(145*145)
	idx_t = np.array(w_true.index)
	idx_f = np.sort(list(set(idx_all)-set(idx_t)))

	wind = np.array(df_ice_wind, dtype='float32')
	u = wind[:,0]/100
	v = wind[:,1]/100
	iw_u = np.zeros(145*145)
	iw_u[idx_t] = u[idx_t]
	iw_u[idx_f] = np.nan
	iw_v = np.zeros(145*145)
	iw_v[idx_t] = v[idx_t]
	iw_v[idx_f] = np.nan
	iw_speed = np.sqrt(iw_u*iw_u+iw_v*iw_v)
	iw_idx_t = idx_t
	return pd.DataFrame({"iw_u": iw_u, "iw_v": iw_v, "iw_speed": iw_speed}), iw_idx_t

def get_1day_ic0_data(ic0_file_name, grid900to145_file_name):
	#145x145グリッドのものを想定
	grid_data = pd.read_csv(grid900to145_file_name, header=None)
	grid145 = np.array(grid_data, dtype='int64').ravel()

	ic0_data = pd.read_csv(ic0_file_name, header=None)
	ic0 = np.array(ic0_data, dtype='float32').ravel()
	ic0_145 = ic0[grid145]
	ic0_idx_t = ~np.isnan(ic0_145)
	return pd.DataFrame({"ic0_145": ic0_145}), ic0_idx_t

def get_1month_coeff_data(coeff_file_name):
	"""
	風力係数、偏角データの読み込み
	偏角（度）, 海流u, 海流v, F, 相関係数, 回帰に使ったデータ数, 海氷流速u, 海氷流速v, 地衡風u, 地衡風v
	"""
	df_coeffs = pd.read_csv(coeff_file_name, sep=',', dtype='float32')
	df_coeffs.columns = ["index", "angle", "mean_ocean_u", "mean_ocean_v", "A", "coef", "data_num", "mean_ice_u", "mean_ice_v", "mean_w_u", "mean_w_v"]
	df_coeffs = df_coeffs.drop("index", axis=1)
	df_coeffs[df_coeffs==999.] = np.nan
	return df_coeffs

def cvt_date(dt):
	# "2013-01-01" -> "20130101"
	return str(dt)[:10].replace('-', '')

def get_non_nan_idx(mat, ocean_idx, strict=True):
	"""
	mat: 複数のidx_tを結合したもの
	strict: matを考慮するかどうか
	"""
	if strict==True:
		data_idx = set(mat) & set(ocean_idx)
		data_idx = np.sort(np.array(list(data_idx)))
	else:
		data_idx = np.sort(ocean_idx)

	return data_idx

################################################################################################
#自分でデータを計算する場所

"""
風力係数などのデータ一式の取得
"""
def get_w_regression_data(wind_file_name, ice_file_name, coeff_file_name):
	"""
	1日ごとの地衡風と流氷速度と、30日分の平均海流を取ってきて、Aとthetaを計算
	存在するインデックスのみ。nanは削除されている
	columns_all = [
		"data_idx", 
		"Lon","Lat","Label", "Name",
		"w_u","w_v","w_speed",
		"w_u10","w_v10","w_speed10",
		"iw_u","iw_v","iw_speed",
		"A_by_day","theta_by_day",
		"ic0_145",
		"t2m",
		"angle", "mean_ocean_u", "mean_ocean_v", "A", "coef", "data_num", "mean_ice_u", "mean_ice_v", "mean_w_u", "mean_w_v"
		]
	"""
	w_data = get_1day_w_data(wind_file_name)
	w_u, w_v, w_speed = np.array(w_data["w_u"]), np.array(w_data["w_v"]), np.array(w_data["w_speed"])
	iw_data, iw_idx_t = get_1day_ice_data(ice_file_name)
	iw_u, iw_v, iw_speed = np.array(iw_data["iw_u"]), np.array(iw_data["iw_v"]), np.array(iw_data["iw_speed"])
	df_coeffs = get_1month_coeff_data(coeff_file_name)
	mean_ocean_u = np.array(df_coeffs["mean_ocean_u"])
	mean_ocean_v = np.array(df_coeffs["mean_ocean_v"])

	real_iw_u = iw_u - mean_ocean_u
	real_iw_v = iw_v - mean_ocean_v
	real_iw_speed = np.sqrt(real_iw_u*real_iw_u + real_iw_v*real_iw_v)
	A_by_day = real_iw_speed / w_speed
	theta_by_day = (np.arctan2(real_iw_v, real_iw_u) - np.arctan2(w_v, w_u))*180/np.pi
	idx1 = np.where(theta_by_day>=180)[0]
	theta_by_day[idx1] = theta_by_day[idx1]-360
	idx2 = np.where(theta_by_day<=-180)[0]
	theta_by_day[idx2] = theta_by_day[idx2]+360

	mat = iw_idx_t.tolist()
	data_idx = get_non_nan_idx(mat, ocean_idx, strict=True)
	data_index_145 = np.zeros(145*145)
	#データがある場所が1
	if len(data_idx != 0):
		data_index_145[data_idx] = 1

	data = pd.DataFrame({"data_idx": data_index_145, "A_by_day": A_by_day, "theta_by_day": theta_by_day})
	return data


def get_masked_region_data(data, region):
	data = data
	if region is not None:
		region_all = list(data.Name.values.flatten())
		region_nan = list(set(region_all)-set(region))
		data_columns = set(data.columns)
		nan_columns = list(data_columns - set(["data_idx", "Label", "Name", "Lon", "Lat"]))
		data.loc[data.Name==area, nan_columns] = np.nan
	return data
