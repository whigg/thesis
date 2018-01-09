import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import seaborn as sns

import pandas.plotting._converter as pandacnv
pandacnv.register()

import calc_data
from main_v import mkdir

latlon145_file_name = calc_data.latlon145_file_name
latlon900_file_name = calc_data.latlon900_file_name
grid900to145_file_name = calc_data.grid900to145_file_name
ocean_grid_file = calc_data.ocean_grid_file
ocean_grid_145 = calc_data.ocean_grid_145
ocean_idx = calc_data.ocean_idx

latlon_ex = calc_data.get_lonlat_data()
basic_region = ["bearing_sea", "chukchi_sea", "beaufort_sea", "canada_islands", "hudson_bay", "buffin_bay", "labrador_sea", "greenland_sea", 
	"norwegian_sea", "barents_sea", "kara_sea", "laptev_sea", "east_siberian_sea", "north_polar"]



#Aの時系列変化
#月ごと
def H_ts_A_month(m_plot=True, y_plot=True):
	dirs_1g = "../result_h/H_ts_month_1g/"
	mkdir(dirs_1g)
	dirs_2g = "../result_h/H_ts_month_2g/"
	mkdir(dirs_2g)
	dirs_3g = "../result_h/H_ts_month_3g/"
	mkdir(dirs_3g)
	dirs_year = "../result_h/H_ts_year/"
	mkdir(dirs_year)

	def plot_param_1g(data_m_1g, save_name):
		fig, axes = plt.subplots(3, 1)
		dates = pd.date_range("2001", periods=12, freq='MS')

		plot_data_1g = data_m_1g["A"].loc[:, ["mean", "std", "50%"]]
		plot_data_1g["2sigma_pos"] = plot_data_1g['mean']+1*np.sqrt(plot_data_1g['std'])
		plot_data_1g["2sigma_neg"] = plot_data_1g['mean']-1*np.sqrt(plot_data_1g['std'])
		#plot_data_1g["Month"] = dates
		axes[0].plot(dates, plot_data_1g['mean'], '-', color="k")
		#d = plot_data_1g['Month'].values
		axes[0].fill_between(dates, plot_data_1g['2sigma_pos'], plot_data_1g['2sigma_neg'],
			facecolor='green', alpha=0.3, interpolate=True)
		#axes[0].set_ylim([0, 0.025])
		axes[0].set_ylabel('A')
		#axes[0].set_title('A, ' + str_year)
		axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

		plot_data_1g_theta = data_m_1g["theta"].loc[:, ["mean", "std", "50%"]]
		plot_data_1g_theta["2sigma_pos"] = plot_data_1g_theta['mean']+2*np.sqrt(plot_data_1g_theta['std'])
		plot_data_1g_theta["2sigma_neg"] = plot_data_1g_theta['mean']-2*np.sqrt(plot_data_1g_theta['std'])
		axes[1].plot(dates, plot_data_1g_theta['mean'], '-', color="k")
		axes[1].fill_between(dates, plot_data_1g_theta['2sigma_pos'], plot_data_1g_theta['2sigma_neg'],
			facecolor='lightskyblue', alpha=0.3, interpolate=True)
		#axes[1].set_ylim([-180, 180])
		axes[1].set_ylim([-60, 60])
		#axes[1].set_yticks([-180, -120, -60, 0, 60, 120, 180])
		axes[1].set_ylabel('theta')
		#axes[1].set_title('theta, ' + str_year)
		axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

		plot_data_1g_e2 = data_m_1g["epsilon2"].loc[:, ["mean", "std", "50%"]]
		plot_data_1g_e2["2sigma_pos"] = plot_data_1g_e2['mean']+2*np.sqrt(plot_data_1g_e2['std'])
		plot_data_1g_e2["2sigma_neg"] = plot_data_1g_e2['mean']-2*np.sqrt(plot_data_1g_e2['std'])
		axes[2].plot(dates, plot_data_1g_e2['mean'], '-', color="k")
		axes[2].fill_between(dates, plot_data_1g_e2['2sigma_pos'], plot_data_1g_e2['2sigma_neg'],
			facecolor='silver', alpha=0.3, interpolate=True)
		#axes[2].set_ylim([-180, 180])
		#axes[2].set_yticks([-180, -120, -60, 0, 60, 120, 180])
		axes[2].set_ylabel('e2')
		#axes[2].set_title('e2, ' + str_year)
		axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

		#plt.setp(axes[0].get_xticklabels(), visible=False)
		#plt.setp(axes[1].get_xticklabels(), visible=False)
		plt.tight_layout()

		plt.savefig(save_name, dpi=900)
		plt.close()

	def plot_param_2g(data_m_2g, save_name):
		plot_param_list = ["A", "theta", "epsilon2"]
		fig, axes = plt.subplots(3, 2)
		dates = pd.date_range("2001", periods=12, freq='MS')
		axes[0, 0].set_title("Polar Region")
		axes[0, 1].set_title("Coastal Region")
		for i, item in enumerate(plot_param_list):
			for j, is_np in enumerate([1,0]):
				plot_data_2g_np_pos = data_m_2g.loc[(is_np), (item, ["mean", "std", "50%"])]
				plot_data_2g_np_pos["2sigma_pos"] = plot_data_2g_np_pos[(item, 'mean')]+2*np.sqrt(plot_data_2g_np_pos[(item, 'std')])
				plot_data_2g_np_pos["2sigma_neg"] = plot_data_2g_np_pos[(item, 'mean')]-2*np.sqrt(plot_data_2g_np_pos[(item, 'std')])
				#plot_data_2g_np_pos["Month"] = np.arange(1, 13, 1)
				ax = axes[i, j]
				#plt.subplot(321+j+i*2)
				ax.plot_date(dates, plot_data_2g_np_pos[(item, 'mean')], '-', color="k")
				#d = plot_data_2g_np_pos['Month'].values
				ax.fill_between(dates, plot_data_2g_np_pos['2sigma_pos'], plot_data_2g_np_pos['2sigma_neg'],
					facecolor='lightskyblue', alpha=0.3, interpolate=True)
				if i == 0:
					ax.set_ylim([0, 0.025])
					#ax.set_yticks([0, 0.025])
				elif i == 1:
					ax.set_ylim([-180, 180])
					ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
				elif i == 2:
					ax.set_ylim([0, 1.5])
					#ax.set_yticks([0, 0.025])
				ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
			
		axes[0, 0].set_ylabel('A')
		axes[1, 0].set_ylabel('theta')
		axes[2, 0].set_ylabel('e2')
		axes[2, 0].set_xlabel("Month")
		axes[2, 1].set_xlabel("Month")
		"""
		plt.setp(axes[0, 0].get_xticklabels(), visible=False)
		plt.setp(axes[1, 0].get_xticklabels(), visible=False)
		plt.setp(axes[0, 1].get_xticklabels(), visible=False)
		plt.setp(axes[1, 1].get_xticklabels(), visible=False)
		plt.setp(axes[0, 1].get_yticklabels(), visible=False)
		plt.setp(axes[1, 1].get_yticklabels(), visible=False)
		plt.setp(axes[2, 1].get_yticklabels(), visible=False)
		"""
		plt.tight_layout()

		plt.savefig(save_name, dpi=900)
		plt.close()

	def plot_param_3g(data_m_3g, plot_param_item, save_name):
		fig, axes = plt.subplots(3, 2)
		dates = pd.date_range("2001", periods=12, freq='MS')
		axes[0, 0].set_title("Polar Region")
		axes[0, 1].set_title("Coastal Region")
		for i, is_np in enumerate([1, 0]):
			plot_data_3g_np_pos = data_m_3g.loc[(is_np), (plot_param_item, ["mean", "std", "50%"])]
			plot_data_3g_np_pos["2sigma_pos"] = plot_data_3g_np_pos[(plot_param_item, "mean")] + 2*np.sqrt(plot_data_3g_np_pos[(plot_param_item, "std")])
			plot_data_3g_np_pos["2sigma_neg"] = plot_data_3g_np_pos[(plot_param_item, "mean")] - 2*np.sqrt(plot_data_3g_np_pos[(plot_param_item, "std")])
			for j, subplot_idx in enumerate([2,1,0]):
				ax = axes[j, i]
				ax.plot(dates, plot_data_3g_np_pos.loc[(subplot_idx), (plot_param_item, "mean")], '-', color="k")
				ax.fill_between(dates, plot_data_3g_np_pos.loc[(subplot_idx), ("2sigma_pos")], plot_data_3g_np_pos.loc[(subplot_idx), ("2sigma_neg")],
					facecolor='lightskyblue', alpha=0.3, interpolate=True)
		if plot_param_item == "A":
			for ax in axes:
				ax.set_ylim([0, 0.025])
		elif plot_param_item == "theta":
			for ax in axes:
				ax.set_ylim([-180, 180])
				ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
		elif plot_param_item == "e2":
			for ax in axes:
				ax.set_ylim([0, 1.5])
		axes[0, 0].set_ylabel('R2 High')
		axes[1, 0].set_ylabel('R2 Middle')
		axes[2, 0].set_ylabel('R2 Low')
		axes[2, 0].set_xlabel("Month")
		axes[2, 1].set_xlabel("Month")
		"""
		plt.setp(axes[0, 0].get_xticklabels(), visible=False)
		plt.setp(axes[1, 0].get_xticklabels(), visible=False)
		plt.setp(axes[0, 1].get_xticklabels(), visible=False)
		plt.setp(axes[1, 1].get_xticklabels(), visible=False)
		plt.setp(axes[0, 1].get_yticklabels(), visible=False)
		plt.setp(axes[1, 1].get_yticklabels(), visible=False)
		plt.setp(axes[2, 1].get_yticklabels(), visible=False)
		"""
		plt.tight_layout()

		plt.savefig(save_name, dpi=900)
		plt.close()

	def plot_param_1g_through_years(data_1g_dic, save_name):
		fig, axes = plt.subplots(3, 1)
		dates1 = pd.date_range("2003", "2010", freq='MS').append(pd.date_range("2013", "2017", freq='MS'))
		dates = dates1[:-1]

		data_A = pd.DataFrame([])
		data_theta = pd.DataFrame([])
		data_e2 = pd.DataFrame([])
		for y in y_list:
			data_A = pd.concat([data_A, data_1g_dic[y]["1g_A"]])
			data_theta = pd.concat([data_theta, data_1g_dic[y]["1g_theta"]])
			data_e2 = pd.concat([data_e2, data_1g_dic[y]["1g_e2"]])

		axes[0].plot(dates, data_A['mean'], '-', color="k")
		axes[0].fill_between(
			dates, data_A['mean']+2*np.sqrt(data_A['std']), data_A['mean']-2*np.sqrt(data_A['std']),
			facecolor='green', alpha=0.3, interpolate=True
			)
		axes[0].set_ylim([0, 0.025])
		axes[0].set_ylabel('A')
		axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		axes[1].plot(dates, data_theta['mean'], '-', color="k")
		axes[1].fill_between(
			dates, data_theta['mean']+2*np.sqrt(data_theta['std']), data_theta['mean']-2*np.sqrt(data_theta['std']),
			facecolor='green', alpha=0.3, interpolate=True
			)
		axes[1].set_ylim([-180, 180])
		axes[1].set_ylabel('theta')
		axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		axes[2].plot(dates, data_e2['mean'], '-', color="k")
		axes[2].fill_between(
			dates, data_e2['mean']+2*np.sqrt(data_e2['std']), data_e2['mean']-2*np.sqrt(data_e2['std']),
			facecolor='green', alpha=0.3, interpolate=True
			)
		#axes[2].set_ylim([0, 0.025])
		axes[2].set_ylabel('e2')
		axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		plt.tight_layout()
		plt.savefig(save_name, dpi=900)
		plt.close()

	def plot_param_2g_through_years(data_2g_dic, save_name):
		fig, axes = plt.subplots(3, 2)
		dates1 = pd.date_range("2003", "2010", freq='MS').append(pd.date_range("2013", "2017", freq='MS'))
		dates = dates1[:-1]

		data_A_np = pd.DataFrame([])
		data_A_coastal = pd.DataFrame([])
		data_theta_np = pd.DataFrame([])
		data_theta_coastal = pd.DataFrame([])
		data_e2_np = pd.DataFrame([])
		data_e2_coastal = pd.DataFrame([])
		for y in y_list:
			data_A_np = pd.concat([data_A_np, data_2g_dic[y]["2g_A_polar"]])
			data_A_coastal = pd.concat([data_A_coastal, data_2g_dic[y]["2g_A_coastal"]])
			data_theta_np = pd.concat([data_theta_np, data_2g_dic[y]["2g_theta_polar"]])
			data_theta_coastal = pd.concat([data_theta_coastal, data_2g_dic[y]["2g_theta_coastal"]])
			data_e2_np = pd.concat([data_e2_np, data_2g_dic[y]["2g_e2_polar"]])
			data_e2_coastal = pd.concat([data_e2_coastal, data_2g_dic[y]["2g_e2_coastal"]])

		axes[0, 0].plot(dates, data_A_np['mean'], '-', color="k")
		axes[0, 0].fill_between(
			dates, data_A_np['mean']+2*np.sqrt(data_A_np['std']), data_A_np['mean']-2*np.sqrt(data_A_np['std']),
			facecolor='lightskyblue', alpha=0.3, interpolate=True
			)
		axes[0, 0].set_ylim([0, 0.025])
		axes[0, 0].set_ylabel('A')
		axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		axes[0, 1].plot(dates, data_A_coastal['mean'], '-', color="k")
		axes[0, 1].fill_between(
			dates, data_A_coastal['mean']+2*np.sqrt(data_A_coastal['std']), data_A_coastal['mean']-2*np.sqrt(data_A_coastal['std']),
			facecolor='lightskyblue', alpha=0.3, interpolate=True
			)
		axes[0, 1].set_ylim([0, 0.025])
		axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		axes[1, 0].plot(dates, data_theta_np['mean'], '-', color="k")
		axes[1, 0].fill_between(
			dates, data_theta_np['mean']+2*np.sqrt(data_theta_np['std']), data_theta_np['mean']-2*np.sqrt(data_theta_np['std']),
			facecolor='lightskyblue', alpha=0.3, interpolate=True
			)
		axes[1, 0].set_ylim([-180, 180])
		axes[1, 0].set_ylabel('theta')
		axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		axes[1, 1].plot(dates, data_theta_coastal['mean'], '-', color="k")
		axes[1, 1].fill_between(
			dates, data_theta_coastal['mean']+2*np.sqrt(data_theta_coastal['std']), data_theta_coastal['mean']-2*np.sqrt(data_theta_coastal['std']),
			facecolor='lightskyblue', alpha=0.3, interpolate=True
			)
		axes[1, 1].set_ylim([-180, 180])
		axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		axes[2, 0].plot(dates, data_e2_np['mean'], '-', color="k")
		axes[2, 0].fill_between(
			dates, data_e2_np['mean']+2*np.sqrt(data_e2_np['std']), data_e2_np['mean']-2*np.sqrt(data_e2_np['std']),
			facecolor='lightskyblue', alpha=0.3, interpolate=True
			)
		#axes[2, 0].set_ylim([0, 0.025])
		axes[2, 0].set_ylabel('e2')
		axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		axes[2, 1].plot(dates, data_e2_coastal['mean'], '-', color="k")
		axes[2, 1].fill_between(
			dates, data_e2_coastal['mean']+2*np.sqrt(data_e2_coastal['std']), data_e2_coastal['mean']-2*np.sqrt(data_e2_coastal['std']),
			facecolor='lightskyblue', alpha=0.3, interpolate=True
			)
		#axes[2, 1].set_ylim([0, 0.025])
		axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y%m'))

		#axes[2, 0].set_xlabel("Year")
		#axes[2, 1].set_xlabel("Year")
		plt.tight_layout()
		plt.savefig(save_name, dpi=900)
		plt.close()

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	data_1g_dic = {}
	data_2g_dic = {}
	data_3g_dic = {}
	for y in y_list:
		data_m = pd.DataFrame([])
		for m in month_list:
			yymm = "20" + y + m
			hermert_file_name = "../data/csv_Helmert_30/Helmert_30_" + yymm + ".csv"
			data = calc_data.get_1month_hermert_data(hermert_file_name)
			data = pd.concat([latlon_ex["Name"], data], axis=1)
			#data = data.drop(["Lat", "Lon", "Label", "idx1", "idx2"], axis=1)
			rank_np = np.zeros(145**2)
			rank_np[data[data.Name=="north_polar"].index] = 1
			data["rank_np"] = rank_np
			rank_R2 = np.ones(145**2)
			rank_R2[data[data.R2<=(1/3)**2].index] = 0
			rank_R2[data[data.R2>(2/3)**2].index] = 2
			data["rank_R2"] = rank_R2
			data = data.dropna()
			#https://code.i-harness.com/ja/q/1c29878
			print(data.isnull().sum().sum())
			data["yymm"] = [pd.to_datetime(yymm, format="%Y%m")] * len(data)
			data_m = pd.concat([data_m, data])

		#月ごとに全てのエリアの平均などを取得
		data_m_1g = data_m.groupby("yymm")[["A", "theta", "epsilon2"]].describe().sort_index(level='yymm')
		#月ごとにrank_npで分類したものを取得
		data_m_2g = data_m.groupby(["rank_np", "yymm"])[["A", "theta", "epsilon2"]].describe().sort_index(level='yymm')
		#月ごとにrank_npとrank_R2で分類したものを取得
		data_m_3g = data_m.groupby(["rank_np", "rank_R2", "yymm"])[["A", "theta", "epsilon2"]].describe().sort_index(level='yymm')

		if m_plot == True:
			#1gのプロット
			save_name_1g = dirs_1g + "1g_" + y + ".png"
			plot_param_1g(data_m_1g, save_name_1g)
			#2gのプロット
			save_name_2g = dirs_2g + "2g_" + y + ".png"
			plot_param_2g(data_m_2g, save_name_2g)
			#3gのプロット
			plot_param_list = ["A", "theta", "epsilon2"]
			for item in plot_param_list:
				save_name_3g = dirs_3g + item + "/" + "3g_" + y + ".png"
				plot_param_3g(data_m_3g, item, save_name_3g)

		"""
		sns.tsplot(data=data_m_A, ci="sd")
		plt.plot(np.nanmean(data_m_A, axis=0))
		data_m_theta = data_m[:,:,1].T
		sns.tsplot(data=data_m_theta, ci="sd")
		data_m_e2 = data_m[:,:,2].T
		sns.tsplot(data=data_m_e2, ci="sd")
		"""

		tmp_1g = {
			"1g_A": data_m_1g["A"].loc[:, ["mean", "std", "50%"]],
			"1g_theta": data_m_1g["theta"].loc[:, ["mean", "std", "50%"]],
			"1g_e2": data_m_1g["epsilon2"].loc[:, ["mean", "std", "50%"]]
			}
		data_1g_dic[y] = tmp_1g
		tmp_2g = {
			"2g_A_polar": data_m_2g.loc[(1), ("A", ["mean", "std", "50%"])],
			"2g_A_coastal": data_m_2g.loc[(0), ("A", ["mean", "std", "50%"])],
			"2g_theta_polar": data_m_2g.loc[(1), ("theta", ["mean", "std", "50%"])],
			"2g_theta_coastal": data_m_2g.loc[(0), ("theta", ["mean", "std", "50%"])],
			"2g_e2_polar": data_m_2g.loc[(1), ("epsilon2", ["mean", "std", "50%"])],
			"2g_e2_coastal": data_m_2g.loc[(0), ("epsilon2", ["mean", "std", "50%"])]
			}
		data_2g_dic[y] = tmp_2g
		tmp_3g = {
			"3g_A_polar": data_m_3g.loc[(1), ("A", ["mean", "std", "50%"])],
			"3g_A_coastal": data_m_3g.loc[(0), ("A", ["mean", "std", "50%"])],
			"3g_theta_polar": data_m_3g.loc[(1), ("theta", ["mean", "std", "50%"])],
			"3g_theta_coastal": data_m_3g.loc[(0), ("theta", ["mean", "std", "50%"])],
			"3g_e2_polar": data_m_3g.loc[(1), ("epsilon2", ["mean", "std", "50%"])],
			"3g_e2_coastal": data_m_3g.loc[(0), ("epsilon2", ["mean", "std", "50%"])]
			}
		data_3g_dic[y] = tmp_3g

	if y_plot == True:
		#1g
		save_name_1g_year = dirs_year + "1g.png"
		plot_param_1g_through_years(data_1g_dic, save_name_1g_year)
		#2g
		save_name_2g_year = dirs_year + "2g.png"
		plot_param_2g_through_years(data_2g_dic, save_name_2g_year)

	return data_1g_dic, data_2g_dic, data_3g_dic
















"""
#Aの時系列変化
#年ごと
def H_ts_A_year():
	def y_plot_param_1g(save_name):
		plot_param_list = ["A", "theta", "epsilon2"]
		for i, item in enumerate(plot_param_list):
			plot_data_1g = data_m_1g[item].loc[:, ["mean", "std", "50%"]]
			plot_data_1g["2sigma_pos"] = plot_data_1g['mean']+2*np.sqrt(plot_data_1g['std'])
			plot_data_1g["2sigma_neg"] = plot_data_1g['mean']-2*np.sqrt(plot_data_1g['std'])
			plot_data_1g["Year"] = np.array([2003,2004,2005,2006,2007,2008,2009,2010,2013,2014,2015,2016])
			plt.subplot(311+i)
			plt.plot_date(plot_data_1g['Year'], plot_data_1g['mean'], '-')
			plt.plot_date(plot_data_1g['Year'], plot_data_1g["2sigma_pos"], '-')
			plt.plot_date(plot_data_1g['Year'], plot_data_1g["2sigma_neg"], '-')
			d = plot_data_1g['Year'].values
			plt.fill_between(d, plot_data_1g['2sigma_pos'], plot_data_1g['2sigma_neg'],
				facecolor='green', alpha=0.2, interpolate=True)
		plt.savefig(save_name, dpi=900)
		plt.close()

	def y_plot_param_2g(save_name):
		plot_param_list = ["A", "theta", "epsilon2"]
		for i, item in enumerate(plot_param_list):
			for j, is_np in enumerate([1,0]):
				plot_data_2g_np_pos = data_m_2g.loc[(is_np), (item, ["mean", "std", "50%"])]
				plot_data_2g_np_pos["2sigma_pos"] = plot_data_2g_np_pos['mean']+2*np.sqrt(plot_data_2g_np_pos['std'])
				plot_data_2g_np_pos["2sigma_neg"] = plot_data_2g_np_pos['mean']-2*np.sqrt(plot_data_2g_np_pos['std'])
				plot_data_2g_np_pos["Year"] = np.array([2003,2004,2005,2006,2007,2008,2009,2010,2013,2014,2015,2016])
				plt.subplot(321+j+i*2)
				plt.plot_date(plot_data_2g_np_pos['Year'], plot_data_2g_np_pos['mean'], '-')
				plt.plot_date(plot_data_2g_np_pos['Year'], plot_data_2g_np_pos["2sigma_pos"], '-')
				plt.plot_date(plot_data_2g_np_pos['Year'], plot_data_2g_np_pos["2sigma_neg"], '-')
				d = plot_data_2g_np_pos['Year'].values
				plt.fill_between(d, plot_data_2g_np_pos['2sigma_pos'], plot_data_2g_np_pos['2sigma_neg'],
					facecolor='green', alpha=0.2, interpolate=True)
		
		plt.savefig(save_name, dpi=900)
		plt.close()

	def y_plot_param_3g(plot_param_item, save_name):
		for i, is_np in enumerate([1, 0]):
			plot_data_3g_np_pos = data_m_2g.loc[(is_np), (plot_param_item, ["mean", "std", "50%"])]
			plot_data_3g_np_pos["2sigma_pos"] = plot_data_3g_np_pos[(plot_param_item, "mean")] + 2*np.sqrt(plot_data_3g_np_pos[(plot_param_item, "std")])
			plot_data_3g_np_neg["2sigma_neg"] = plot_data_3g_np_neg[(plot_param_item, "mean")] - 2*np.sqrt(plot_data_3g_np_neg[(plot_param_item, "std")])
			#plot_data_3g_np_neg["Month"] = np.arange(1, 13, 1)
			year_array = np.array([2003,2004,2005,2006,2007,2008,2009,2010,2013,2014,2015,2016])
			for j, subplot_idx in enumerate([0,1,2]):
				plt.subplot(321+i+subplot_idx*2)
				plt.plot_date(year_array, plot_data_3g_np_pos.loc[(j), (plot_param_item, "mean")], '-')
				plt.plot_date(year_array, plot_data_3g_np_pos.loc[(j), ("2sigma_pos")], '-')
				plt.plot_date(year_array, plot_data_3g_np_pos.loc[(j), ("2sigma_neg")], '-')
				plt.fill_between(year_array, plot_data_3g_np_pos.loc[(j), ("2sigma_pos")], plot_data_3g_np_pos.loc[(j), ("2sigma_neg")],
					facecolor='green', alpha=0.2, interpolate=True)
		plt.savefig(save_name, dpi=900)
		plt.close()

	y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
	month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

	for m in month_list:
		data_m = pd.DataFrame([])
		for y in y_list:
			yymm = "20" + y + m
			hermert_file_name = "../data/csv_Helmert_30/Helmert_30_" + yymm + ".csv"
			data = calc_data.get_1month_hermert_data(hermert_file_name)
			data = pd.concat([latlon_ex["Name"], data], axis=1)
			#data = data.drop(["Lat", "Lon", "Label", "idx1", "idx2"], axis=1)
			rank_np = np.zeros(145**2)
			rank_np[data[data.Name=="north_polar"].index] = 1
			data["rank_np"] = rank_np
			rank_R2 = np.ones(145**2)
			rank_R2[data[data.R2<=(1/3)**2].index] = 0
			rank_R2[data[data.R2>(2/3)**2].index] = 2
			data["rank_R2"] = rank_R2
			data = data.dropna()
			#https://code.i-harness.com/ja/q/1c29878
			print(data.isnull().sum().sum())
			data["Year"] = [int("20"+y)] * len(data)
			data_m = pd.concat([data_m, data])

		#年ごとに全てのエリアの平均などを取得
		data_m_1g = data_m.groupby("Year")[["A", "theta", "epsilon2"]].describe().sort_index(level='Year')
		#年ごとにrank_npで分類したものを取得
		data_m_2g = data_m.groupby(["rank_np", "Year"])[["A", "theta", "epsilon2"]].describe().sort_index(level='Year')
		#年ごとにrank_npとrank_R2で分類したものを取得
		data_m_3g = data_m.groupby(["rank_np", "rank_R2", "Year"])[["A", "theta", "epsilon2"]].describe().sort_index(level='Year')

		#1gのプロット
		y_plot_param_1g(save_name)

		#2gのプロット
		y_plot_param_2g(save_name)

		#3gのプロット
		plot_param_list = ["A", "theta", "epsilon2"]
		for item in plot_param_list:
			y_plot_param_3g(item, save_name)
"""







