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

import pandas.plotting._converter as pandacnv
pandacnv.register()

import calc_data


latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

latlon_ex = calc_data.get_lonlat_data()

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



###############################################################################################################

def ts_by_month():
	dirs = "../result_h/ts_by_month/"
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	corr_all = []
	for year in y_list:

		data_A_month, data_theta_month, data_R2_month, data_e2_month = [], [], [], []
		for month in month_list:
			file_list = "../data/csv_Helmert_both_30/Helmert_both_30_20" + year + month + ".csv"
			df = pd.read_csv(file_list)
			data = df.groupby("area_label")[["A", "theta", "R2", "epsilon2"]].describe()

			for item in ["A", "theta", "R2", "epsilon2"]:
				data.loc[:, (item, "1sigma_pos")] = data.loc[:, (item, "mean")] + data.loc[:, (item, "std")]
				data.loc[:, (item, "1sigma_neg")] = data.loc[:, (item, "mean")] - data.loc[:, (item, "std")]
				data.loc[:, (item, "2sigma_pos")] = data.loc[:, (item, "mean")] + 2*data.loc[:, (item, "std")]
				data.loc[:, (item, "2sigma_neg")] = data.loc[:, (item, "mean")] - 2*data.loc[:, (item, "std")]

			data_A = data.loc[:, ("A", ["mean", "1sigma_pos", "1sigma_neg", "2sigma_pos", "2sigma_neg", "count"])].values
			data_theta = data.loc[:, ("theta", ["mean", "1sigma_pos", "1sigma_neg", "2sigma_pos", "2sigma_neg", "count"])].values
			data_R2 = data.loc[:, ("R2", ["mean", "1sigma_pos", "1sigma_neg", "2sigma_pos", "2sigma_neg", "count"])].values
			data_e2 = data.loc[:, ("epsilon2", ["mean", "1sigma_pos", "1sigma_neg", "2sigma_pos", "2sigma_neg", "count"])].values
			data_A_month.append(data_A)
			data_theta_month.append(data_theta)
			data_R2_month.append(data_R2)
			data_e2_month.append(data_e2)

		data_A_month = np.array(data_A_month)
		data_theta_month = np.array(data_theta_month)
		data_R2_month = np.array(data_R2_month)
		data_e2_month = np.array(data_e2_month)

		for i in range(18):
			plt.figure(figsize=(9, 6))
			gs = gridspec.GridSpec(3,2)
			dates = pd.date_range("2001", periods=12, freq='MS')

			plt.subplot(gs[0, 0])
			plt.plot(dates, data_A_month[:,i,0], '-', color="k")
			plt.fill_between(dates, data_A_month[:,i,1], data_A_month[:,i,2],
				facecolor='green', alpha=0.3, interpolate=True)
			plt.ylim([0, 0.025])
			plt.ylabel('A')
			plt.lot(gs[0, 0]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))

			plt.subplot(gs[1, 0])
			plt.plot(dates, data_theta_month[:,i,0], '-', color="k")
			plt.fill_between(dates, data_theta_month[:,i,1], data_theta_month[:,i,2],
				facecolor='lightskyblue', alpha=0.3, interpolate=True)
			plt.ylim([-40, 40])
			plt.yticks([-40, -30, -20, -10, 0, 10, 20, 30, 40])
			plt.ylabel(r'$\theta$')
			plt.subplot(gs[1, 0]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))

			plt.subplot(gs[0, 1])
			plt.plot(dates, data_R2_month[:,i,0], '-', color="k")
			plt.fill_between(dates, data_R2_month[:,i,1], data_R2_month[:,i,2],
				facecolor='coral', alpha=0.3, interpolate=True)
			plt.ylim([0, 1])
			plt.yticks([0, .2, .4, .6, .8, 1])
			plt.ylabel(r'$R^{2}$')
			plt.subplot(gs[0, 1]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))

			plt.subplot(gs[1, 1])
			plt.plot(dates, data_e2_month[:,i,0], '-', color="k")
			plt.fill_between(dates, data_e2_month[:,i,1], data_e2_month[:,i,2],
				facecolor='silver', alpha=0.3, interpolate=True)
			plt.ylim([0, 1.5])
			plt.yticks([0, .5, 1, 1.5])
			plt.ylabel(r'$e^{2}$')
			plt.subplot(gs[1, 1]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))

			plt.subplot(gs[2, :])
			y = data_A_month[:,i,5]
			plt.plot(dates, y, '-', color="k")
			y_lim_min = max(y.min-5,0)
			y_lim_max = y.max+5
			plt.ylim([y_lim_min, y_lim_max])
			plt.yticks(y_lim_min, y_lim_max, int(y_lim_max-y_lim_min+1))
			plt.ylabel("number of data")
			plt.subplot(gs[2, :]).get_xaxis().set_major_formatter(mdates.DateFormatter('%m'))
			plt.grid(True)

			plt.tight_layout()

			save_name = dirs + "all_20" + year + month + ".png"
			plt.savefig(save_name, dpi=400)
			plt.close()


		"""
		for i in range(18):
			
			fig, axes = plt.subplots(2, 2)
			dates = pd.date_range("2001", periods=12, freq='MS')

			axes[0,0].plot(dates, data_A_month[:,i,0], '-', color="k")
			axes[0,0].fill_between(dates, data_A_month[:,i,1], data_A_month[:,i,2],
				facecolor='green', alpha=0.3, interpolate=True)
			axes[0,0].set_ylim([0, 0.025])
			axes[0,0].set_ylabel('A')
			axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

			axes[1,0].plot(dates, data_theta_month[:,i,0], '-', color="k")
			axes[1,0].fill_between(dates, data_theta_month[:,i,1], data_theta_month[:,i,2],
				facecolor='lightskyblue', alpha=0.3, interpolate=True)
			axes[1,0].set_ylim([-40, 40])
			axes[1,0].set_yticks([-40, -30, -20, -10, 0, 10, 20, 30, 40])
			axes[1,0].set_ylabel(r'$\theta$')
			axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

			axes[0,1].plot(dates, data_R2_month[:,i,0], '-', color="k")
			axes[0,1].fill_between(dates, data_R2_month[:,i,1], data_R2_month[:,i,2],
				facecolor='coral', alpha=0.3, interpolate=True)
			axes[0,1].set_ylim([0, 1])
			axes[0,1].set_yticks([0, .2, .4, .6, .8, 1])
			axes[0,1].set_ylabel(r'$R^{2}$')
			axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

			axes[1,1].plot(dates, data_e2_month[:,i,0], '-', color="k")
			axes[1,1].fill_between(dates, data_e2_month[:,i,1], data_e2_month[:,i,2],
				facecolor='silver', alpha=0.3, interpolate=True)
			axes[1,1].set_ylim([0, 1.5])
			axes[1,1].set_yticks([0, .5, 1, 1.5])
			axes[1,1].set_ylabel(r'$e^{2}$')
			axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

			#plt.setp(axes[0,0].get_xticklabels(), visible=False)
			#plt.setp(axes[0,1].get_xticklabels(), visible=False)
			plt.tight_layout()

			save_name = dirs + "all_20" + year + month + ".png"
			plt.savefig(save_name, dpi=400)
			plt.close()
		"""













if __name__ == '__main__':
	ts_by_month()



"""
TODO
・countの処理

"""




