
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from datetime import datetime, date, timezone, timedelta
import os.path
import os
import seaborn as sns
import matplotlib.gridspec as gridspec
import random
import itertools
import matplotlib.dates as mdates

import pandas.plotting._converter as pandacnv
pandacnv.register()

import calc_data
import visualize


latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

latlon_ex = calc_data.get_lonlat_data()
no_sensor_data = pd.read_csv("../data/non_sensor_index.csv", header=None)
no_sensor_data.columns = ["non"]

start_list = []
n = 20000000
y_list = [3,4,5,6,7,8,9,10,13,14,15,16]
for i in y_list:
	m = n + i*10000
	for j in range(12):
		start_list.append(m + (j+1)*100 + 1)
M = len(start_list)
#start_list_plus_1month = start_list + [20170901]


cm_angle_2 = visualize.generate_cmap([
	"blue", 
	"gainsboro", 
	"red"
	])


###############################################################################################################

def plot_ic0_std():
	dirs = "../result_h_1day_delay/std_map/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for month in month_list:
		print(month)
		file_list = sorted(glob.glob("../data/csv_Helmert_30_1day_delay/Helmert_30_1day_delay_*" + month + ".csv"))
		accumulate_data = []
		for file in file_list:
			data = pd.read_csv(file)
			data.loc[data["A"].isnull(), "ic0_30"] = np.nan
			data = np.array(data["ic0_30"])
			#print(np.sum(np.isnan(data)))
			accumulate_data.append(data)
		accumulate_data = np.array(accumulate_data)
		#print(accumulate_data.shape)
		ic0_std = np.nanstd(accumulate_data, axis=0)
		ic0_count = np.nansum(~np.isnan(accumulate_data), axis=0)
		ic0_std = np.where(ic0_count>5, ic0_std, np.nan)
		#print(ocean_grid_145.shape)
		#print(ic0_std.shape)
		#ic0_std = np.where(np.array(ocean_grid_145).ravel()==1, ic0_std, np.nan)

		#print(np.sum(np.isnan(ic0_std)))
		save_name = dirs + "ic0_std_" + month + ".png"
		visualize.plot_map_once(ic0_std, data_type="type_non_wind", show=False, 
				save_name=save_name, vmax=None, vmin=None, cmap=plt.cm.jet)


###############################################################################################################

def test_ts_all_regression():
	# https://org-technology.com/posts/detrend.html
	dates1 = pd.date_range("2003", "2006", freq='MS')[:-1]
	y = [1,2,2,3,4,3,5,3,6,6,4,2,
	2,1,3,4,3,5,6,6,7,5,3,2,
	3,2,4,4,6,5,8,9,5,4,3,3]

	from scipy import signal
	yd = signal.detrend(y)
	plt.figure(figsize=(6,4))
	plt.plot(t, y, label="Original Data")
	plt.plot(t, y-yd, "--r", label="Trend")
	plt.plot(t, yd, "c", label="Detrended Data")
	plt.axhline(0, color="k", linestyle="--", label="Mean of Detrended Data")
	plt.axis("tight")
	plt.legend(loc=0)
	plt.show()



def ts_all_regression():




















###############################################################################################################
		
plot_ic0_std_1day_delay()






























