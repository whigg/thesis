
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

import calc_data
import visualize
from main_v import mkdir, main_data

latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

latlon_ex = calc_data.get_lonlat_data()
basic_region = ["bearing_sea", "chukchi_sea", "beaufort_sea", "canada_islands", "hudson_bay", "buffin_bay", "labrador_sea", "greenland_sea", 
	"norwegian_sea", "barents_sea", "kara_sea", "laptev_sea", "east_siberian_sea", "north_polar"]

#基本的な変数の定義
start_list = []
n = 20000000
y_list = [3,4,5,6,7,8,9,10,13,14,15,16]
for i in y_list:
	m = n + i*10000
	for j in range(12):
		start_list.append(m + (j+1)*100 + 1)
start_ex_list = [20170101, 20170201, 20170301, 20170401, 20170501, 20170601,20170701,20170801]
start_list = np.sort(np.array(list(set(start_list)|set(start_ex_list)))).tolist()
M = len(start_list)



###################################################################################################################


def plot_hist():
	mkdir("../result_h/search_anomaly/hist/")

	start_list.pop()
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M-1))

		data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_" + str(start)[:-2] + ".csv"
		data = pd.read_csv(data_ex_dir)

		x = data.R2[((data.area_label==16)) & ((data.A<0.003)|(data.ic0_30<97))].dropna().values
		y = data.R2[((data.area_label==16)) & ((data.A>=0.003)&(data.ic0_30>=97))].dropna().values
		sns.distplot(x, color="skyblue")
		sns.distplot(y, color="red")
		"""
		fig, ax = plt.subplots()
		for a in [x, y]:
			sns.distplot(a, ax=ax, kde=True)
		ax.set_xlim([0, 0.03])
		"""

		save_name = "../result_h/search_anomaly/hist/" + "A_ic0_" + str(start)[:-2] + ".png"
		plt.savefig(save_name, dpi=600)
		plt.close()

		print("\n")


if __name__ == '__main__':
	plot_hist()






































