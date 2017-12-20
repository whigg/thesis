
"""
マップを月ごとに出力して保存するコード
	A
	angle
	相関係数
"""
start_list = []
M = len(start_list)
plot_kw = "A"

for i, start in enumerate(start_list):
	print("**************  {}/{}  ***************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["coeff"], 
		region=None, 
		accumulate=False
		)

	data = data[plot_kw]
	save_name = "A_30_" + str(start)[:6] + ".png"
	main_plot(
		data,
		what_type="map",
		mode=[0, plot_kw],
		save=True,
		save_name=save_name,
		show=False
		)
	print("\n")


###############################################################################################################

"""
A_by_dayのマップを月ごとに出力して保存するコード
	A_by_dayの平均
	theta_by_dayの平均
"""
plot_kw = "A_by_day"
start_list = []
M = len(start_list)

for i, start in enumerate(start_list):
	print("***************  {}/{}  ***************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["ex_1"], 
		region=None, 
		accumulate=True
		)

	data_array = np.array(data)
	data_array = np.ma.masked_invalid(data_array)
	data_count_nan = np.sum(data_array.recordmask, axis=0)
	data_ave = np.sum(data_array, axis=0) / (len(date_ax) - data_count_nan)
	#A_by_dayなので0列目
	data_ave = pd.DataFrame(data_ave[:, 0])
	data_ave.columns = plot_kw
	#閾値を設ける場合
	#threshold = 0.05
	#data_ave.loc[data_ave.plot_kw>=threshold, :] = np.nan

	save_name = "A_by_30_" + str(start)[:6] + ".png"
	main_plot(
		data_ave,
		what_type="map",
		mode=[0, plot_kw],
		save=True,
		save_name=save_name,
		show=False
		)
	print("\n")








###############################################################################################################

"""
散布図　これも月ごとに描く
	相関係数とA_by_day
	相関係数とA
	IC0とA
偏角も同様
"""
start_list = []
M = len(start_list)

for i, start in enumerate(start_list):
	print("***************  {}/{}  ***************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["ic0_145"], 
		region=None, 
		accumulate=True
		)

	data_array = np.array(data)
	data_array = np.ma.masked_invalid(data_array)
	data_count_nan = np.sum(data_array.recordmask, axis=0)
	data_ave = np.sum(data_array, axis=0) / (len(date_ax) - data_count_nan)
	#ic0_145なので列指定はなし
	data_ave = pd.DataFrame(data_ave)
	#閾値を設ける場合
	#threshold = 0.05
	#data_ave.loc[data_ave>=threshold, :] = np.nan

	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["coeff"], 
		region=None, 
		accumulate=False
		)
	data_y = data["A"]
	plot_data = pd.concat([data_ave, data_y])

	save_name = "scatter_ic0_A_" + str(start)[:6] + ".png"
	main_plot(
		data_ave,
		what_type="nonline",
		mode=["scatter", ["ic0_145", "A"]],
		save=True,
		save_name=save_name,
		show=False
		)
	print("\n")













"""
#指定したインデックスだけ地図上にプロットしたいとき
#例えば、↓
idx = np.array(a[:,0].tolist(), dtype='int64')
idx = idx[wi_ratio>=0.2]

m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
fig=plt.figure(figsize=(8,8))

#グリッドの描画
df_latlon = pd.read_csv(latlon145_file_name, header=None)
latlon = np.array(df_latlon, dtype='float32')
lat = latlon[:,2]
lon = latlon[:,3]
lons, lats = m(lon[idx], lat[idx], inverse=False)
m.plot(lons,lats,'bo', markersize=2)

m.drawcoastlines(color = '0.15')

plt.show()
"""




