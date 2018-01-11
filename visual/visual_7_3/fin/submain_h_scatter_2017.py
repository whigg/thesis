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
threshold_R2 = 0.4


###############################################################################################################

#散布図：Aとic0 北極のみ
#散布図：Aとic0 相関が低いものは除く 北極のみ
#散布図：angleとic0 北極のみ
#散布図：angleとic0 相関が低いものは除く 北極のみ
def H_scatter_ic0_np(mode):
	if mode == "mean":
		dirs_A_30_and_ic0_np = "../result_h/scatter/scatter_A_30_and_ic0_np/"
		mkdir(dirs_A_30_and_ic0_np)
		dirs_A_30_and_ic0_h_np = "../result_h/scatter/scatter_A_30_and_ic0_h_np/"
		mkdir(dirs_A_30_and_ic0_h_np)
		dirs_angle_30_and_ic0_np = "../result_h/scatter/scatter_angle_30_and_ic0_np/"
		mkdir(dirs_angle_30_and_ic0_np)
		dirs_angle_30_and_ic0_h_np = "../result_h/scatter/scatter_angle_30_and_ic0_h_np/"
		mkdir(dirs_angle_30_and_ic0_h_np)
	elif mode == "median":
		dirs_A_30_and_ic0_median_np = "../result_h/scatter/scatter_A_30_and_ic0_median_np/"
		mkdir(dirs_A_30_and_ic0_median_np)
		dirs_A_30_and_ic0_median_h_np = "../result_h/scatter/scatter_A_30_and_ic0_median_h_np/"
		mkdir(dirs_A_30_and_ic0_median_h_np)
		dirs_angle_30_and_ic0_median_np = "../result_h/scatter/scatter_angle_30_and_ic0_median_np/"
		mkdir(dirs_angle_30_and_ic0_median_np)
		dirs_angle_30_and_ic0_median_h_np = "../result_h/scatter/scatter_angle_30_and_ic0_median_h_np/"
		mkdir(dirs_angle_30_and_ic0_median_h_np)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A_original = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		_, _, _, data_ic0_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["ic0_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_ic0_30)

		if mode == "mean":
			data_ave = np.nanmean(data_array, axis=0)
			data_ave = pd.DataFrame(data_ave)
			data_ave.columns = ["ic0_30"]

			#dirs_A_30_and_ic0_np
			data_A = data_A_original["A"]
			data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
			data = data[data.Name=="north_polar"]

			save_name = dirs_A_30_and_ic0_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data,
				mode=["scatter", ["ic0_30", "A"]],
				save_name=save_name,
				show=False
				)

			#dirs_A_30_and_ic0_h_np
			"""
			data_A_1 = data_A_original.loc[:, ["A", "R2"]]
			data_A_1.A[data_A_1.R2<threshold_R2] = np.nan
			data_1 = pd.concat([latlon_ex, data_A_1, data_ave], axis=1)
			data_1 = data_1[data_1.Name=="north_polar"]

			save_name = dirs_A_30_and_ic0_h_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_1,
				mode=["scatter", ["ic0_30", "A"]],
				save_name=save_name,
				show=False
				)
			"""

			#dirs_angle_30_and_ic0_np
			data_angle = data_A_original["theta"]
			data_2 = pd.concat([latlon_ex, data_angle, data_ave], axis=1)
			data_2 = data_2[data_2.Name=="north_polar"]

			save_name = dirs_angle_30_and_ic0_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_2,
				mode=["scatter", ["ic0_30", "theta"]],
				save_name=save_name,
				show=False
				)

			#dirs_angle_30_and_ic0_h_np
			"""
			data_angle_1 = data_A_original.loc[:, ["theta", "R2"]]
			data_angle_1.theta[data_angle_1.R2<threshold_R2] = np.nan
			data_3 = pd.concat([latlon_ex, data_angle_1, data_ave], axis=1)
			data_3 = data_3[data_3.Name=="north_polar"]

			save_name = dirs_angle_30_and_ic0_h_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_3,
				mode=["scatter", ["ic0_30", "theta"]],
				save_name=save_name,
				show=False
				)
			"""

		elif mode == "median":
			data_ave = np.nanmedian(data_array, axis=0)
			data_ave = pd.DataFrame(data_ave)
			data_ave.columns = ["ic0_30"]

			#dirs_A_30_and_ic0_np
			data_A = data_A_original["A"]
			data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
			data = data[data.Name=="north_polar"]

			save_name = dirs_A_30_and_ic0_median_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data,
				mode=["scatter", ["ic0_30", "A"]],
				save_name=save_name,
				show=False
				)

			#dirs_A_30_and_ic0_h_np
			"""
			data_A_1 = data_A_original.loc[:, ["A", "R2"]]
			data_A_1.A[data_A_1.R2<threshold_R2] = np.nan
			data_1 = pd.concat([latlon_ex, data_A_1, data_ave], axis=1)
			data_1 = data_1[data_1.Name=="north_polar"]

			save_name = dirs_A_30_and_ic0_median_h_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_1,
				mode=["scatter", ["ic0_30", "A"]],
				save_name=save_name,
				show=False
				)
			"""

			#dirs_angle_30_and_ic0_np
			data_angle = data_A_original["theta"]
			data_2 = pd.concat([latlon_ex, data_angle, data_ave], axis=1)
			data_2 = data_2[data_2.Name=="north_polar"]
			
			save_name = dirs_angle_30_and_ic0_median_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_2,
				mode=["scatter", ["ic0_30", "theta"]],
				save_name=save_name,
				show=False
				)

			#dirs_angle_30_and_ic0_h_np
			"""
			data_angle_1 = data_A_original.loc[:, ["theta", "R2"]]
			data_angle_1.theta[data_angle_1.R2<threshold_R2] = np.nan
			data_3 = pd.concat([latlon_ex, data_angle_1, data_ave], axis=1)
			data_3 = data_3[data_3.Name=="north_polar"]

			save_name = dirs_angle_30_and_ic0_median_h_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_3,
				mode=["scatter", ["ic0_30", "theta"]],
				save_name=save_name,
				show=False
				)
			"""

		print("\n")






#散布図：Aとsit 全海域
#散布図：Aとsit 北極のみ
#散布図：Aとsit 相関が低いものは除く 北極のみ
#散布図：angleとsit 全海域
#散布図：angleとsit 北極海のみ
#散布図：angleとsit 相関が低いものは除く 北極のみ
def H_scatter_sit_np(worker):
	if worker == 0:
		dirs_A_30_and_sit_all = "../result_h/scatter/scatter_A_30_and_sit_all/"
		mkdir(dirs_A_30_and_sit_all)
		dirs_A_30_and_sit_np = "../result_h/scatter/scatter_A_30_and_sit_np/"
		mkdir(dirs_A_30_and_sit_np)
		dirs_A_30_and_sit_h_np = "../result_h/scatter/scatter_A_30_and_sit_h_np/"
		mkdir(dirs_A_30_and_sit_h_np)
	elif worker == 1:
		dirs_angle_30_and_sit_all = "../result_h/scatter/scatter_angle_30_and_sit_all/"
		mkdir(dirs_angle_30_and_sit_all)
		dirs_angle_30_and_sit_np = "../result_h/scatter/scatter_angle_30_and_sit_np/"
		mkdir(dirs_angle_30_and_sit_np)
		dirs_angle_30_and_sit_h_np = "../result_h/scatter/scatter_angle_30_and_sit_h_np/"
		mkdir(dirs_angle_30_and_sit_h_np)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A_original = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		_, _, _, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_sit_30)
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["sit_30"]

		if worker == 0:
			#dirs_A_30_and_sit_all
			data_A = data_A_original["A"]
			data = pd.concat([latlon_ex, data_A, data_ave], axis=1)

			save_name = dirs_A_30_and_sit_all + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data,
				mode=["scatter", ["sit_30", "A"]],
				save_name=save_name,
				show=False
				)

			#dirs_A_30_and_sit_np
			data_A_1 = data_A_original.loc[:, ["A", "R2"]]
			data_1 = pd.concat([latlon_ex, data_A_1, data_ave], axis=1)
			data_1 = data_1[data_1.Name=="north_polar"]

			save_name = dirs_A_30_and_sit_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_1,
				mode=["scatter", ["sit_30", "A"]],
				save_name=save_name,
				show=False
				)

			#dirs_A_30_and_sit_h_np
			"""
			data_A_2 = data_A_original.loc[:, ["A", "R2"]]
			data_A_2.A[data_A_2.R2<threshold_R2] = np.nan
			data_1_2 = pd.concat([latlon_ex, data_A_2, data_ave], axis=1)
			data_1_2 = data_1_2[data_1_2.Name=="north_polar"]

			save_name = dirs_A_30_and_sit_h_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_1_2,
				mode=["scatter", ["sit_30", "A"]],
				save_name=save_name,
				show=False
				)
			"""

		elif worker == 1:
			#dirs_angle_30_and_sit_all
			data_angle = data_A_original["theta"]
			data_2 = pd.concat([latlon_ex, data_angle, data_ave], axis=1)

			save_name = dirs_angle_30_and_sit_all + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_2,
				mode=["scatter", ["sit_30", "theta"]],
				save_name=save_name,
				show=False
				)

			#dirs_angle_30_and_sit_np
			data_angle_1 = data_A_original.loc[:, ["theta", "R2"]]
			data_3 = pd.concat([latlon_ex, data_angle_1, data_ave], axis=1)
			data_3 = data_3[data_3.Name=="north_polar"]

			save_name = dirs_angle_30_and_sit_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_3,
				mode=["scatter", ["sit_30", "theta"]],
				save_name=save_name,
				show=False
				)

			#dirs_angle_30_and_sit_h_np
			"""
			data_angle_1 = data_A_original.loc[:, ["theta", "R2"]]
			data_angle_1.theta[data_angle_1.R2<threshold_R2] = np.nan
			data_3 = pd.concat([latlon_ex, data_angle_1, data_ave], axis=1)
			data_3 = data_3[data_3.Name=="north_polar"]

			save_name = dirs_angle_30_and_sit_h_np + str(start)[:6] + ".png"
			visualize.visual_non_line(
				data_3,
				mode=["scatter", ["sit_30", "theta"]],
				save_name=save_name,
				show=False
				)
			"""

		print("\n")




#散布図：[A, theta, e2]と[ic0, sit]
#基本的には上のscatter関数群と同じ
#ic0とsitをかけたものも出力できる(この部分は新たに書く)
#1画像に6グラフ
#列：npかどうか
#行：R2の閾値(3段階)
def H_scatter_6win_2area(worker):
	dirs_basic = "../result_h/scatter/"
	if worker == 0:
		dirs_A_ic0 = dirs_basic + "A_ic0/"
		mkdir(dirs_A_ic0)
		dirs_theta_ic0 = dirs_basic + "theta_ic0/"
		mkdir(dirs_theta_ic0)
		dirs_e2_ic0 = dirs_basic + "e2_ic0/"
		mkdir(dirs_e2_ic0)
	elif worker == 1:
		dirs_A_sit = dirs_basic + "A_sit/"
		mkdir(dirs_A_sit)
		dirs_theta_sit = dirs_basic + "theta_sit/"
		mkdir(dirs_theta_sit)
		dirs_e2_sit = dirs_basic + "e2_sit/"
		mkdir(dirs_e2_sit)

	sns.set_style("darkgrid")
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_hermert = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)

		_, _, _, data_ic0_sit = main_data(
			start, end, 
			span=30, 
			get_columns=["ic0_145", "sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_ic0_sit)
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["ic0_30", "sit_30"]
		data_cross = data_array[:, :, 0] * data_array[:, :, 1]
		data_cross_ave = np.nanmean(data_cross, axis=0) / 100
		data_ave["cross_ic0_sit"] = data_cross_ave

		data_tmp = data_hermert.loc[:, ["A", "theta", "R2", "epsilon2"]]
		data_basic = pd.concat([latlon_ex, data_tmp, data_ave], axis=1)

		rank_np = np.zeros(145**2)
		rank_np[data_basic[data_basic.Name=="north_polar"].index] = 1
		rank_R2 = np.ones(145**2)
		rank_R2[data_basic[data_basic.R2<=(0.5)**2].index] = 0
		rank_R2[data_basic[data_basic.R2>(0.6)**2].index] = 2
		data_rank = pd.DataFrame({"rank_np": rank_np, "rank_R2": rank_R2})

		data = pd.concat([data_basic, data_rank], axis=1)

		if worker == 0:
			save_name = dirs_A_ic0 + str(start)[:6] + ".png"
			sns.lmplot(x="ic0_30", y="A", row="rank_R2", col="rank_np", data=data, size=3, fit_reg=True)
			plt.savefig(save_name, dpi=900)
			plt.close()

			save_name = dirs_theta_ic0 + str(start)[:6] + ".png"
			sns.lmplot(x="ic0_30", y="theta", row="rank_R2", col="rank_np", data=data, size=3)
			plt.savefig(save_name, dpi=900)
			plt.close()

			save_name = dirs_e2_ic0 + str(start)[:6] + ".png"
			sns.lmplot(x="ic0_30", y="epsilon2", row="rank_R2", col="rank_np", data=data, size=3)
			plt.savefig(save_name, dpi=900)
			plt.close()

		elif worker == 1:
			save_name = dirs_A_sit + str(start)[:6] + ".png"
			sns.lmplot(x="sit_30", y="A", row="rank_R2", col="rank_np", data=data, size=3)
			plt.savefig(save_name, dpi=900)
			plt.close()

			save_name = dirs_theta_sit + str(start)[:6] + ".png"
			sns.lmplot(x="sit_30", y="theta", row="rank_R2", col="rank_np", data=data, size=3)
			plt.savefig(save_name, dpi=900)
			plt.close()

			save_name = dirs_e2_sit + str(start)[:6] + ".png"
			sns.lmplot(x="sit_30", y="epsilon2", row="rank_R2", col="rank_np", data=data, size=3)
			plt.savefig(save_name, dpi=900)
			plt.close()

		print("\n")

