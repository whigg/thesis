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
import main_v
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
#start_list_plus_1month = start_list + [20170901]


cm_angle = visualize.generate_cmap([
	[0, 0, 96/255], 
	[0/255, 0/255, 255/255], 
	[108/255, 108/255, 108/255], 
	[255/255, 0/255, 0/255], 
	[96/255, 0, 0]
	])

cm_angle_1 = visualize.generate_cmap([
	[0, 0, 204/255], 
	[0/255, 204/255, 0/255], 
	[160/255, 160/255, 160/255], 
	[255/255, 255/255, 51/255], 
	[204/255, 0, 0]
	])

cm_angle_2 = visualize.generate_cmap([
	"blue", 
	"Lime", 
	"grey", 
	"yellow", 
	"red"
	])

#threshold_R2 = 0.4**2
#threshold_R2 = 0.4


###############################################################################################################

#30_30, 90_90でどれだけ海流が取れているかのマップ
def ocean_30_vs_90():
	dirs = "../result_h/mean_vector/ocean_currents_30_vs_90/"
	mkdir(dirs)

	start_list.pop()
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		helmert_30_30_fname = "../data/csv_Helmert_30/Helmert_30_" + str(start)[:6] + ".csv"
		data_30 = pd.read_csv(helmert_30_30_fname)
		data_30_vec = [np.array(data_30["ocean_u"]), np.array(data_30["ocean_v"])]
		helmert_90_90_fname = "../data/csv_Helmert_both_90/Helmert_both_90_" + str(start)[:6] + ".csv"
		data_90 = pd.read_csv(helmert_90_90_fname)
		data_90_vec = [np.array(data_90["ocean_u_90"]), np.array(data_90["ocean_v_90"])]
		helmert_30_30_w_iw_fname = "../data/csv_Helmert_both_30_w_iw/Helmert_both_30_w_iw_" + str(start)[:6] + ".csv"
		data_30_w_iw = pd.read_csv(helmert_30_30_w_iw_fname)
		data_30_w_iw_vec = [np.array(data_30_w_iw["ocean_u_w_iw"]), np.array(data_30_w_iw["ocean_v_w_iw"])]

		fig, axes = plt.subplots(1,3)
		fig.figsize=(6, 9)
		title_list = ["30", "90", "30_w_iw"]
		vector_list = [data_30_vec, data_90_vec, data_30_w_iw_vec]
		for j, title in enumerate(title_list):
			ax = axes[j]
			ax.set_title(title)
			m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
			m.ax = ax
			m.drawcoastlines(color = '0.15')
			m.fillcontinents(color='#555555')
			lon = np.array(latlon_ex.Lon)
			lat = np.array(latlon_ex.Lat)
			x, y = m(lon, lat)
			x1 = np.reshape(x, (145,145), order='F')
			y1 = np.reshape(y, (145,145), order='F')
			dx1 = (x1[1,0]-x1[0,0])/2
			dy1 = (y1[0,1]-y1[0,0])/2

			x2 = np.linspace(x1[0,0], x1[144,0], 145)
			y2 = np.linspace(y1[0,0], y1[0,144], 145)
			xx, yy = np.meshgrid(x2, y2)
			xx, yy = xx.T, yy.T

			vector_u = np.ma.masked_invalid(vector_list[j][0])
			vector_v = np.ma.masked_invalid(vector_list[j][1])
			vector_speed = np.sqrt(vector_u*vector_u + vector_v*vector_v)

			data_non_wind = vector_speed
			data_non_wind = np.ma.masked_invalid(data_non_wind)
			data1 = np.reshape(data_non_wind, (145,145), order='F')

			xx = np.hstack([xx, xx[:,0].reshape(145,1)])
			xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
			yy = np.vstack([yy, yy[0,:]])
			yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

			m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=0.2, vmin=0)
			#m.colorbar(location='bottom')
			m.quiver(x, y, vector_u, vector_v, color="k")
		save_name = dirs + str(start)[:6] + ".png"
		print(save_name)
		plt.savefig(save_name, dpi=750)
		plt.close()
		print("\n")



#A,theta,R2,e2が30_30,30_90,30_w_iwでどう変化したかのマップ
def compare_3csv():
	dirs = "../result_h/compare_3csv/"
	dirs_A = "../result_h/compare_3csv/A/"
	dirs_theta = "../result_h/compare_3csv/theta/"
	dirs_R2 = "../result_h/compare_3csv/R2/"
	dirs_e2 = "../result_h/compare_3csv/e2/"
	mkdir(dirs_A)
	mkdir(dirs_theta)
	mkdir(dirs_R2)
	mkdir(dirs_e2)

	kw_list = [["A", "A_30_90", "A_30_w_iw"],
		["theta", "theta_30_90", "theta_w_iw"],
		["R2", "R2_30_90", "R2_w_iw"],
		["epsilon2", "epsilon2_30_90", "epsilon2_w_iw"]]
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		helmert_30_30_fname = "../data/csv_Helmert_30/Helmert_30_" + str(start)[:6] + ".csv"
		data_30 = pd.read_csv(helmert_30_30_fname)
		helmert_30_90_fname = "../data/csv_Helmert_ocean_90/Helmert_30_90_" + str(start)[:6] + ".csv"
		data_90 = pd.read_csv(helmert_30_90_fname)
		helmert_30_30_w_iw_fname = "../data/csv_Helmert_both_30_w_iw/Helmert_both_30_w_iw_" + str(start)[:6] + ".csv"
		data_30_w_iw = pd.read_csv(helmert_30_30_w_iw_fname)

		save_name_list = ["A", "theta", "R2", "e2"]
		for j in range(4):
			data_30_plot = np.array(data_30[kw_list[j][0]])
			data_90_plot = np.array(data_90[kw_list[j][1]])
			data_30_w_iw_plot = np.array(data_30_w_iw[kw_list[j][2]])

			data_list = [data_30_plot, data_90_plot, data_30_w_iw_plot]
			v_lim_list = [[0.025,0], [180,-180], [1,0], [None,0]]
			fig, axes = plt.subplots(1,3)
			for k in range(3):
				ax = axes[k]
				ax.set_title(kw_list[j][k])
				m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
				m.ax = ax
				m.drawcoastlines(color = '0.15')
				m.fillcontinents(color='#555555')
				lon = np.array(latlon_ex.Lon)
				lat = np.array(latlon_ex.Lat)
				x, y = m(lon, lat)
				x1 = np.reshape(x, (145,145), order='F')
				y1 = np.reshape(y, (145,145), order='F')
				dx1 = (x1[1,0]-x1[0,0])/2
				dy1 = (y1[0,1]-y1[0,0])/2

				x2 = np.linspace(x1[0,0], x1[144,0], 145)
				y2 = np.linspace(y1[0,0], y1[0,144], 145)
				xx, yy = np.meshgrid(x2, y2)
				xx, yy = xx.T, yy.T
				xx = np.hstack([xx, xx[:,0].reshape(145,1)])
				xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
				yy = np.vstack([yy, yy[0,:]])
				yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])

				data = np.ma.masked_invalid(data_list[k])
				data1 = np.reshape(data, (145,145), order='F')

				im = m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=v_lim_list[j][0], vmin=v_lim_list[j][1])
				fig.colorbar(im, ax=ax)
				#m.colorbar(location='bottom')
			save_name = dirs + save_name_list[j] + "/" + str(start)[:6] + ".png"
			print(save_name)
			plt.savefig(save_name, dpi=750)
			plt.close()
			#print("\n")






if __name__ == '__main__':

	#ocean_30_vs_90()
	compare_3csv()





