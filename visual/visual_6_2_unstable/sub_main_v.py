
"""
簡単なテスト関数
Aの描画
angleの描画
svatter A, angle, ic0, sit
"""

#木村さんのmean_w_uと３０日平均のw_uがあっているかの確認
def test_0():
	start_list = [20030101]
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_mean_w_u = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		data_mean_w_u = data_mean_w_u.loc[:,["A", "mean_w_u"]]


		date_ax, _, skipping_date_str, data_w_u = main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_w_u)
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		date_ax_len = len(date_ax)
		data_ave_sum = np.sum(data_array, axis=0)
		data_ave_sum[date_ax_len-data_count_nan<=20] = np.nan
		data_ave = data_ave_sum / (date_ax_len-data_count_nan)

		#data_ave = np.sum(data_array, axis=0) / (len(date_ax))
		#w_uは1列目
		data_ave = data_ave[:,1]
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["w_u_30"]

		data = pd.concat([data_mean_w_u, data_ave], axis=1)
		data.to_csv("test_w_u.csv", index=False)

		print("\n")

#test_0()


#IC0データの可視化実験
def test_1():
	filename = "../data/csv_ic0/IC0_csv/220171010D.csv"
	"""
	data = calc_data.get_1day_ic0_data(filename)
	visualize.plot_map_once(
		data,
		data_type="type_non_wind",
		save_name=None,
		show=True,
		vmax=None,
		vmin=None,
		cmap="jet")
	"""
	visualize.plot_900(
		filename,
		save_name=None,
		show=True)



#SITデータの可視化実験
#メルトポンドの挙動を兼ねて
def test_2():
	filename = "../data/csv_sit/SIT_20020813.csv"
	
	data = calc_data.get_1day_sit_data(filename)
	data[data>=10001] = np.nan
	visualize.plot_map_once(
		data,
		data_type="type_non_wind",
		save_name=None,
		show=True,
		vmax=None,
		vmin=None,
		cmap="jet")
	"""
	visualize.plot_900(
		filename,
		save_name=None,
		show=True)
	"""


###############################################################################################################

#地衡風の書き出し。確認用
def w_0():
	wind_files = sorted(glob.glob("../data/binary_w/*.ads60"))
	lon = latlon_ex.Lon
	lat = latlon_ex.Lat
	lon = np.array(lon)
	lat = np.array(lat)

	dirs = "../result/test_wind/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for item in tqdm(wind_files):
		dt = np.dtype([("u","<f"), ("v","<f"), ("slp","<f")])
		fd = open(item, "r")
		result = np.fromfile(fd, dtype=dt, count=-1)
		data = result.tolist()
		data1 = np.array([list(data[i]) for i in range(len(data))])

		w_u = data1[:,0]
		w_v = data1[:,1]
		w_speed = np.sqrt(w_u**2+w_v**2)

		m = Basemap(lon_0=180, boundinglat=50, resolution='l', projection='npstere')
		fig = plt.figure(figsize=(7, 7))
		x, y = m(lon, lat)

		m.drawcoastlines(color = '0.15')
		m.quiver(x, y, w_u, w_v, w_speed)

		save_name = dirs + item[17:26] + ".png"

		plt.savefig(save_name, dpi=450)
		plt.close()


#iwの書き出し。確認用
def w_1():
	wind_files = sorted(glob.glob("../data/csv_iw/*.csv"))
	lon = latlon_ex.Lon
	lat = latlon_ex.Lat
	lon = np.array(lon)
	lat = np.array(lat)

	dirs = "../result/test_ice_wind/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for item in wind_files:
		data = calc_data.get_1day_iw_data(item)
		save_name = dirs + item[15:21] + ".png"
		print(save_name)
		visualize.plot_map_once(
			data.loc[:, ["iw_u", "iw_v"]],
			data_type="type_wind",
			show=False,
			save_name=save_name,
			vmax=None,
			vmin=None,
			cmap=None)
#w_1()

###############################################################################################################

#Aのマップを月ごとに出力して保存するコード
#相関がマイナスでもそのまま出力
def A_0():
	dirs = "../result/A/A_30_original/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		
		#data.A.loc[data.A<0] *= -1
		#data.A.loc[data.data_idx==0.] = np.nan

		save_name = dirs + str(start)[:6] + ".png"

		visualize.plot_map_once(
			data["A"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=0.025, 
			vmin=0,
			cmap="jet"
			)
		print("\n")

#A_0()

#Aのマップを月ごとに出力して保存するコード
#相関がマイナスはプラスにする
def A_1():
	dirs = "../result/A/A_30_modified/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		
		data.A.loc[data.A<0] *= -1
		#data.A.loc[data.data_idx==0.] = np.nan

		save_name = dirs + str(start)[:6] + ".png"

		visualize.plot_map_once(
			data["A"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=0.025, 
			vmin=0,
			cmap="jet"
			)
		print("\n")

#A_1()

#Aのマップを月ごとに出力して保存するコード
#相関がマイナスはプラスにする
#ocean_idxによる補正あり
def A_2():
	dirs = "../result/A/A_30_modified_ocean/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		
		data.A.loc[data.A<0] *= -1
		data.A.loc[data.data_idx==0.] = np.nan

		save_name = dirs + str(start)[:6] + ".png"

		visualize.plot_map_once(
			data["A"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=0.025, 
			vmin=0,
			cmap="jet"
			)
		print("\n")
#A_2()

#Aのマップを月ごとに出力して保存するコード
#相関の絶対値が0.4未満は茶色、それ以外は普通のjetで描きたい
def A_3():
	dirs = "../result/A/A_30_with_coef/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))
	df_latlon = pd.read_csv("../data/latlon_ex.csv")
	lon = df_latlon.Lon
	lat = df_latlon.Lat
	lon = np.array(lon)
	lat = np.array(lat)
	#start_list=[20030101]
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)

		idx_low_coef = data[data.coef**2<0.4**2].index
		data_low_coef = np.array([np.nan]*(145**2))
		data_low_coef[idx_low_coef] = 1
		data_low_coef1 = np.reshape(data_low_coef, (145,145), order="F")
		data_low_coef1 = np.ma.masked_invalid(data_low_coef1)
		data.A.loc[data.A<0] *= -1
		data.A.loc[data.coef**2<0.4**2] = np.nan
		# data.A.loc[data.data_idx==0.] = np.nan
		data = np.array(data.A)
		data1 = np.reshape(data, (145,145), order="F")
		data1 = np.ma.masked_invalid(data1)

		save_name = dirs + str(start)[:6] + ".png"

		m = Basemap(lon_0=180, boundinglat=50, resolution='i', projection='npstere')
		fig = plt.figure(figsize=(6.5, 6.5))
		x, y = m(lon, lat)
		x1 = np.reshape(x, (145,145), order="F")
		y1 = np.reshape(y, (145,145), order="F")
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

		m.drawcoastlines(color = '0.15')
		# m.plot(xx[144,0], yy[144,0], "bo")
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data1, cmap=plt.cm.jet, vmax=0.025, vmin=0)
		m.colorbar(location='bottom')
		cm_brown = visualize.generate_cmap(["saddlebrown", "saddlebrown"])
		m.pcolormesh(xx_ex-dx1, yy_ex+dy1, data_low_coef1, cmap=cm_brown, vmax=0.025, vmin=0)
		#plt.show()
		plt.savefig(save_name, dpi=900)
		plt.close()

		print("\n")
#A_3()


#A_by_yearのマップ出力
#オリジナル
def A_4():
	A_by_year_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	dirs = "../result/A/A_by_year_original/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for item in A_by_year_list:
		fname = "../data/csv_A_by_year/ssc_amsr_ads" + item + "_fin.csv"
		df_coeffs = pd.read_csv(fname, sep=',', dtype='float32')
		df_coeffs.columns = ["index", "angle", "mean_ocean_u", "mean_ocean_v", "A", "coef", "data_num", "mean_ice_u", "mean_ice_v", "mean_w_u", "mean_w_v"]
		df_coeffs = df_coeffs.drop("index", axis=1)
		df_coeffs[df_coeffs==999.] = np.nan
		
		save_name = dirs + "A_" + item + ".png"
		visualize.plot_map_once(
			df_coeffs["A"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=0.025, 
			vmin=0,
			cmap="jet"
			)
		print("\n")
#A_4()


#A_by_yearのマップ出力
#相関が負のところをプラスにする
def A_5():
	A_by_year_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	dirs = "../result/A/A_by_year_modified/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for item in A_by_year_list:
		fname = "../data/csv_A_by_year/ssc_amsr_ads" + item + "_fin.csv"
		df_coeffs = pd.read_csv(fname, sep=',', dtype='float32')
		df_coeffs.columns = ["index", "angle", "mean_ocean_u", "mean_ocean_v", "A", "coef", "data_num", "mean_ice_u", "mean_ice_v", "mean_w_u", "mean_w_v"]
		df_coeffs = df_coeffs.drop("index", axis=1)
		df_coeffs[df_coeffs==999.] = np.nan

		df_coeffs.A[df_coeffs.A<0] *= -1
		
		save_name = dirs + "A_" + item + ".png"
		visualize.plot_map_once(
			df_coeffs["A"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=0.025, 
			vmin=0,
			cmap="jet"
			)
		print("\n")
#A_5()


#A_by_day_30の出力
#A_by_dayのマップを月ごとに出力して保存するコード
def A_6():
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

	dirs = "../result/A/A_by_day_30/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1
		date_ax, date_ax_str, skipping_date_str, data = main_data(
			start, end, 
			span=30, 
			get_columns=["ex_1"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data)
		"""
		data_array = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array.recordmask, axis=0)
		#print(data_count_nan)
		#print(len(date_ax))
		#print(len(date_ax_str))
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		#A_by_dayなので0列目
		data_ave = pd.DataFrame(data_ave[:, 0])
		#print(data_ave)
		data_ave.columns = ["A_by_day"]
		# 閾値を設ける場合
		#data_ave.loc[data_ave.A_by_day>=0.05, :] = np.nan

		# data_aveにLabelとかNameをくっつける場合、以下のdataをmain_plotに渡す
		# Labelなどで絞り込む場合は、ここに操作を付け足す
		#data = pd.concat([latlon_ex, data], axis=1)

		save_name = dirs + "A_by_day_30_" + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data_ave["A_by_day"],
			data_type="type_non_wind", 
			save_name=save_name,
			show=False, 
			vmax=0.025, 
			vmin=None,
			cmap="jet"
			)
		print("\n")
#A_6()

###############################################################################################################

#angleのマップを月ごとに出力して保存するコード
#original
def angle_0():
	dirs = "../result/angle/angle_30_original/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		
		#data.angle.loc[(data.A<0) & (data.angle<0)] += 180
		#data.angle.loc[(data.A<0) & (data.angle>0)] -= 180
		#data.angle.loc[data.coef**2<0.4**2] = np.nan
		#data[data.data_idx==0.] = np.nan

		save_name = dirs + "angle_30_original_" + str(start)[:6] + ".png"

		#visualize.pyで関数を選ぶ
		visualize.plot_map_once(
			data["angle"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle
			)
		print("\n")

#angle_0()


#angleのマップを月ごとに出力して保存するコード
#modified
def angle_1():
	dirs = "../result/angle/angle_30_modified_test/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		
		data[(data.A<0) & (data.angle<0)] += 180
		data[(data.A<0) & (data.angle>0)] -= 180
		#data.angle.loc[data.coef**2<0.4**2] = np.nan
		#data[data.data_idx==0.] = np.nan

		save_name = dirs + "angle_30_modified_" + str(start)[:6] + ".png"

		#visualize.pyで関数を選ぶ
		visualize.plot_map_once(
			data["angle"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle
			)
		print("\n")
#angle_1()


#angleのマップを月ごとに出力して保存するコード
#modified, high coef
def angle_2():
	dirs = "../result/angle/angle_30_high_coef/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		date_ax, date_ax_str, skipping_date_str, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		
		data.angle[(data.A<0) & (data.angle<0)] += 180
		data.angle[(data.A<0) & (data.angle>0)] -= 180

		data.angle[data.coef**2<0.4**2] = np.nan
		#data[data.data_idx==0.] = np.nan

		save_name = dirs + "angle_30_high_coef_" + str(start)[:6] + ".png"

		#visualize.pyで関数を選ぶ
		visualize.plot_map_once(
			data["angle"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle
			)
		print("\n")
#angle_2()


#angleにwindの平均を重ねたマップ
def angle_3():
	dirs = "../result/angle/angle_30_and_wind/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		#angleデータの取得
		date_ax, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		
		data.angle.loc[(data.A<0) & (data.angle<0)] += 180
		data.angle.loc[(data.A<0) & (data.angle>0)] -= 180
		data.angle.loc[data.coef**2<0.4**2] = np.nan

		#地衡風平均の出力
		date_ax, _, _, data_w = main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_w)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)

		date_ax_len = len(date_ax)
		data_ave_sum = np.sum(data_array, axis=0)
		data_ave_sum[date_ax_len-data_count_nan<=20] = np.nan
		data_ave = data_ave_sum / (date_ax_len-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)

		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["w_speed", "w_u", "w_v"]
		print(data_ave.head(3))

		save_name = dirs + "angle_30_and_wind_" + str(start)[:6] + ".png"

		#visualize.pyで関数を選ぶ
		visualize.plot_map_multi(
			data_ave.loc[:, ["w_u", "w_v"]], 
			data["angle"],
			data_type="type_non_wind",
			save_name=save_name,
			show=False, 
			vmax=180, 
			vmin=-180,
			cmap=cm_angle_1
			)
		print("\n")
#angle_3()

###############################################################################################################

#海流の平均のマップ出力
def vec_0():
	dirs = "../result/mean_vector/mean_ocean_currents/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		_, _, _, data = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)

		save_name = dirs + "mean_ocean_currents_" + str(start)[:6] + ".png"

		#visualize.pyで関数を選ぶ
		visualize.plot_map_once(
			data.loc[:, ["mean_ocean_u", "mean_ocean_v"]],
			data_type="type_wind",
			save_name=save_name,
			show=False, 
			vmax=None, 
			vmin=None,
			cmap=None
			)
		print("\n")
#vec_0()


#地衡風のマップを月ごとに出力して保存するコード
def vec_1():
	dirs = "../result/mean_vector/mean_wind/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1
		date_ax, date_ax_str, skipping_date_str, data = main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data)
		#data_array_1 = np.ma.masked_invalid(data_array)
		#data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		#date_ax_len = len(date_ax)
		#data_ave_sum = np.sum(data_array, axis=0)
		data_ave = np.nanmean(data_array, axis=0)
		#data_ave[date_ax_len-data_count_nan<=20] = np.nan

		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["w_speed", "w_u", "w_v"]
		print(data_ave.head(3))

		save_name = dirs + "mean_wind_" + str(start)[:6] + ".png"
		visualize.plot_map_once(
			data_ave.loc[:, ["w_u", "w_v"]],
			data_type="type_wind", 
			save_name=save_name,
			show=False, 
			vmax=None, 
			vmin=None,
			cmap=None
			)
		print("\n")
#vec_1()

###############################################################################################################

#散布図：Aとic0 北極のみ
def scatter_0():
	"""
	start_list = []
	n = 20000000
	y_list = [13,14,15,16]
	for i in y_list:
		m = n + i*10000
		for j in range(12):
			start_list.append(m + (j+1)*100 + 1)
	start_ex_list = [20170101, 20170201, 20170301, 20170401, 20170501, 20170601,20170701,20170801]
	start_list = np.sort(np.array(list(set(start_list)|set(start_ex_list)))).tolist()
	M = len(start_list)
	start_list_plus_1month = start_list + [20170901]
	"""

	dirs = "../result/scatter/scatter_A_30_and_ic0_np/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		data_A = data_A.loc[:, ["data_idx", "A"]]
		data_A.A.loc[data_A.A<0] *= -1
		#data_A.A.loc[data_A.data_idx==0.] = np.nan

		date_ax, _, _, data_ic0_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["ic0_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_ic0_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["ic0_30"]

		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		data = data[data.Name=="north_polar"]
		#print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["ic0_30", "A"]],
			save_name=save_name,
			show=False
			)
		print("\n")

#scatter_0()




#散布図：Aとic0 相関が低いものは除く 北極のみ
def scatter_1():
	dirs = "../result/scatter/scatter_A_30_and_ic0_h_np/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		data_A = data_A.loc[:, ["data_idx", "A", "coef"]]

		data_A.A.loc[data_A.coef**2<0.4**2] = np.nan
		data_A.A.loc[data_A.A<0] *= -1
		#data_A.A.loc[data_A.data_idx==0.] = np.nan


		date_ax, _, skipping_date_str, data_ic0_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["ic0_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_ic0_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["ic0_30"]

		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		data = data[data.Name=="north_polar"]
		print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["ic0_30", "A"]],
			save_name=save_name,
			show=False
			)
		print("\n")

#scatter_1()





#散布図：angleとic0 相関が低いものは除く 北極のみ
def scatter_2():
	dirs = "../result/scatter/scatter_angle_30_and_ic0_h_np/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_angle = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)

		data_angle.angle.loc[(data_angle.A<0) & (data_angle.angle<0)] += 180
		data_angle.angle.loc[(data_angle.A<0) & (data_angle.angle>0)] -= 180
		data_angle.angle.loc[data_angle.coef**2<0.4**2] = np.nan


		date_ax, _, skipping_date_str, data_ic0_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["ic0_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_ic0_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["ic0_30"]

		data = pd.concat([latlon_ex, data_angle, data_ave], axis=1)
		data = data[data.Name=="north_polar"]
		#print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["ic0_30", "angle"]],
			save_name=save_name,
			show=False
			)
		print("\n")

#scatter_2()




#散布図：Aとsit 全海域
def scatter_3():
	dirs = "../result/scatter/scatter_A_30_and_sit_all/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		data_A = data_A.loc[:, ["data_idx", "A"]]
		data_A.A.loc[data_A.A<0] *= -1
		#data_A.A.loc[data_A.data_idx==0.] = np.nan

		date_ax, _, _, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_sit_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["sit_30"]

		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		#data = data[data.Name=="north_polar"]
		#print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "A"]],
			save_name=save_name,
			show=False
			)
		print("\n")

#scatter_3()



#散布図：Aとsit 北極のみ
def scatter_4():
	dirs = "../result/scatter/scatter_A_30_and_sit_np/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		data_A = data_A.loc[:, ["data_idx", "A"]]
		data_A.A.loc[data_A.A<0] *= -1
		#data_A.A.loc[data_A.data_idx==0.] = np.nan

		date_ax, _, _, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_sit_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["sit_30"]

		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		data = data[data.Name=="north_polar"]
		#print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "A"]],
			save_name=save_name,
			show=False
			)
		print("\n")

#scatter_4()




#散布図：Aとsit 相関が低いものは除く 北極のみ
def scatter_5():
	dirs = "../result/scatter/scatter_A_30_and_sit_h_np/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_A = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)
		data_A = data_A.loc[:, ["data_idx", "A", "coef"]]

		data_A.A.loc[data_A.coef**2<0.4**2] = np.nan
		data_A.A.loc[data_A.A<0] *= -1
		#data_A.A.loc[data_A.data_idx==0.] = np.nan


		date_ax, _, skipping_date_str, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_sit_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["sit_30"]

		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		data = data[data.Name=="north_polar"]
		print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "A"]],
			save_name=save_name,
			show=False
			)
		print("\n")

#scatter_5()



#散布図：angleとsit 全海域
def scatter_6():
	dirs = "../result/scatter/scatter_angle_30_and_sit_all/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_angle = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)

		data_angle.angle.loc[(data_angle.A<0) & (data_angle.angle<0)] += 180
		data_angle.angle.loc[(data_angle.A<0) & (data_angle.angle>0)] -= 180
		#data_angle.angle.loc[data_angle.coef**2<0.4**2] = np.nan

		date_ax, _, skipping_date_str, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_sit_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["sit_30"]

		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		#data = data[data.Name=="north_polar"]
		#print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "angle"]],
			save_name=save_name,
			show=False
			)
		print("\n")






#散布図：angleとsit 北極海のみ
def scatter_7():
	dirs = "../result/scatter/scatter_angle_30_and_sit_np/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_angle = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)

		data_angle.angle.loc[(data_angle.A<0) & (data_angle.angle<0)] += 180
		data_angle.angle.loc[(data_angle.A<0) & (data_angle.angle>0)] -= 180
		#data_angle.angle.loc[data_angle.coef**2<0.4**2] = np.nan

		date_ax, _, skipping_date_str, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_sit_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["sit_30"]

		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		data = data[data.Name=="north_polar"]
		#print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "angle"]],
			save_name=save_name,
			show=False
			)
		print("\n")






#散布図：angleとsit 相関が低いものは除く 北極のみ
def scatter_8():
	dirs = "../result/scatter/scatter_angle_30_and_sit_h_np/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_angle = main_data(
			start, start, 
			span=30, 
			get_columns=["coeff"], 
			region=None, 
			accumulate=False
			)

		data_angle.angle.loc[(data_angle.A<0) & (data_angle.angle<0)] += 180
		data_angle.angle.loc[(data_angle.A<0) & (data_angle.angle>0)] -= 180
		#data_angle.angle.loc[data_angle.coef**2<0.4**2] = np.nan

		date_ax, _, skipping_date_str, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)

		data_array = np.array(data_sit_30)
		"""
		data_array_1 = np.ma.masked_invalid(data_array)
		data_count_nan = np.sum(data_array_1.recordmask, axis=0)
		data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
		"""
		data_ave = np.nanmean(data_array, axis=0)
		data_ave = pd.DataFrame(data_ave)
		data_ave.columns = ["sit_30"]

		data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
		#data = data[data.Name=="north_polar"]
		#print(data.dropna().head())

		save_name = dirs + str(start)[:6] + ".png"
		#visualize.pyで関数を選ぶ
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "angle"]],
			save_name=save_name,
			show=False
			)
		print("\n")






###############################################################################################################



"""
TODO
・w_0()関数以下の部分が不完全な可能性があるので、確認・修正
	ディレクトリの操作、名前、accumulateのカラム、cmap引数の有無など
	とくにscatterはいろいろ必要

"""


def get_helmert():
	dirs = "../data/csv_Helmert_30/"
	try:
		os.makedirs(dirs)
	except:
		print('directory {} already exists'.format(dirs))

	#start_list = [20030101]
	for i, start in enumerate(start_list):
		print("*******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		#wデータの取得・整形
		date_ax_w, _, _, data_w = main_data(
			start, end, 
			span=30, 
			get_columns=["w"], 
			region=None, 
			accumulate=True
			)
		data_array_w = np.array(data_w)
		"""
		data_array_w_1 = np.ma.masked_invalid(data_array_w)
		data_w_count_nan = np.sum(data_array_w_1.recordmask, axis=0)
		date_ax_w_len = len(date_ax_w)
		#data_array_w[date_ax_w_len-data_w_count_nan<=20] = np.nan
		
		data_ave_w_sum = np.sum(data_array_w, axis=0)
		#data_ave_sum[date_ax_len-data_count_nan<=20] = np.nan
		data_ave_w = data_ave_w_sum / (date_ax_w_len-data_w_count_nan)
		"""
		data_ave_w = np.nanmean(data_array_w, axis=0)

		#iwデータの取得・整形
		date_ax_iw, _, _, data_iw = main_data(
			start, end, 
			span=30, 
			get_columns=["iw"], 
			region=None, 
			accumulate=True
			)

		data_array_iw = np.array(data_iw)
		print("\n")
		#print("data_array_iw:  {}".format(data_array_iw[0,1001:2000,:]))
		#print("data_array_w:  {}".format(data_array_w.shape))
		#print("data_array_iw:  {}".format(data_array_iw.shape))
		#print("data_array_iw[:,:,0]:  {}".format(data_array_iw[:,:,0].shape))
		#print("data_array_w[:,0,0]:  {}".format(data_array_w[:,0,0].shape))

		w_array = np.vstack((data_array_iw[:,:,1], data_array_iw[:,:,2]))
		Helmert = []
		#for i in range(1218,1220):
		for i in range(145**2):
			print("i: {}".format(i))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_array_iw[:, i, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_array_w[:, i, 0][not_nan_idx].reshape((-1,1))
			y = data_array_w[:, i, 1][not_nan_idx].reshape((-1,1))
			w = w_array[:, i][np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_array_iw[:,i,1])
			iw_v_ave = np.nanmean(data_array_iw[:,i,2])
			N_c = np.sum(not_nan_idx==True)
			if N_c <= 1:
				print("\tskipping for loop...")
				Helmert.append([np.nan, np.nan, np.nan, np.nan, np.nan, N_c, np.nan, np.nan])
				continue
			one_N = np.ones(N_c).reshape((-1,1))
			zero_N = np.zeros(N_c).reshape((-1,1))
			D_1 = np.hstack((one_N, zero_N, x, -y))
			D_2 = np.hstack((zero_N, one_N, y, x))
			Dt = np.vstack((D_1, D_2))
			#print(N_c, Dt, np.dot(Dt.T, Dt))
			D_inv = np.linalg.inv(np.dot(Dt.T, Dt))
			gamma = np.dot(D_inv, np.dot(Dt.T, w))
			A = np.sqrt(gamma[2]**2 + gamma[3]**2)
			theta = np.arctan2(gamma[3], gamma[2]) * 180/np.pi
			R_denominator = np.dot(w.T, w) - N_c*(iw_u_ave**2 + iw_v_ave**2)
			epsilon = w - np.dot(Dt, gamma)
			R_numerator = np.dot(epsilon.T, epsilon)
			R2 = 1 - (R_numerator/R_denominator)[0,0]
			print("\t{}".format([A[0], theta[0], gamma[0,0], gamma[1,0], R2, N_c]))
			if N_c <= 20:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_30_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)


#get_helmert()









