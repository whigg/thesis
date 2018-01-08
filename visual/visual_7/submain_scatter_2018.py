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
from main_v import get_date_ax, mkdir, main_data

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
#main_vと同じ
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
start_list_plus_1month = start_list + [20170901]


#threshold_R2 = 0.4**2
#threshold_R2 = 0.4


###############################################################################################################

def scatter_ic0_by_csv():
	csv_kind_list = ["csv_30_30", "csv_30_90", "csv_30_30_w_iw"]
	plot_y_list = ["A", "theta"]
	area_list = ["area_" + str(i) for i in range(17)]
	for csv_item in csv_kind_list:
		for plot_item in plot_y_list:
			for area_item in area_list:
				dirs = "../result_h/scatter/ic0_by_csv/" + csv_item + "/" + plot_item + "/" + area_item + "/"
				mkdir(dirs)

	kw_y_list = [["A", "A_30_90", "A_30_w_iw"],
		["theta", "theta_30_90", "theta_w_iw"]]

	sns.set_style("darkgrid")
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		helmert_30_30_fname = "../data/csv_Helmert_30/Helmert_30_" + str(start)[:6] + ".csv"
		data_30 = pd.read_csv(helmert_30_30_fname)
		helmert_30_90_fname = "../data/csv_Helmert_ocean_90/Helmert_30_90_" + str(start)[:6] + ".csv"
		data_90 = pd.read_csv(helmert_30_90_fname)
		helmert_30_30_w_iw_fname = "../data/csv_Helmert_both_30_w_iw/Helmert_both_30_w_iw_" + str(start)[:6] + ".csv"
		data_30_w_iw = pd.read_csv(helmert_30_30_w_iw_fname)

		for j, kw in enumerate(kw_list):
			for k, area_item in enumerate(area_list):
				save_name_30_30 = "../result_h/scatter/ic0_by_csv/" + csv_kind_list[0] + "/" + plot_y_list[j] + "/" + area_item + "/" + "lmplot_coastal_2_" + str(start)[:6] + ".png"
				sns.lmplot(x="ic0_30", y=kw[0], col="coastal_region_2", data=data_30[data_30.area_label==k], size=3, fit_reg=True)
				plt.savefig(save_name_30_30, dpi=900)
				plt.close()

				save_name_30_90 = "../result_h/scatter/ic0_by_csv/" + csv_kind_list[1] + "/" + plot_y_list[j] + "/" + area_item + "/" + "lmplot_coastal_2_" + str(start)[:6] + ".png"
				sns.lmplot(x="ic0_30", y=kw[1], col="coastal_region_2", data=data_90[data_90.area_label==k], size=3, fit_reg=True)
				plt.savefig(save_name_30_90, dpi=900)
				plt.close()

				save_name_30_30_w_iw = "../result_h/scatter/ic0_by_csv/" + csv_kind_list[2] + "/" + plot_y_list[j] + "/" + area_item + "/" + "lmplot_coastal_2_" + str(start)[:6] + ".png"
				sns.lmplot(x="ic0_30", y=kw[2], col="coastal_region_2", data=data_30_w_iw[data_30_w_iw.area_label==k], size=3, fit_reg=True)
				plt.savefig(save_name_30_30_w_iw, dpi=900)
				plt.close()





if __name__ == '__main__':
	scatter_ic0_by_csv()











