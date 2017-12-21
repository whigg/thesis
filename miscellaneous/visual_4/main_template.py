

#Aのマップを月ごとに出力して保存するコード
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

#start_list = [20130101]
for i, start in enumerate(start_list):
	print("******************  {}/{}  *******************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["coeff"], 
		region=None, 
		accumulate=False
		)
	#print(data.head())
	#data = data["A"]
	#print(len(data))

	# latlon_exで絞り込む場合，ここに処理を書く
	#data = pd.concat([latlon_ex, data], axis=1)
	data.loc[data.A<0] *= -1
	data[data.data_idx==0.] = np.nan
	save_name = "../result/A_30_visual_4/A_30_" + str(start)[:6] + ".png"

	#visualize.pyで関数を選ぶ
	visualize.plot_map_once(
		data["A"],
		data_type="type_non_wind",
		save_name=save_name,
		show=False, 
		vmax=0.025, 
		vmin=None
		)
	print("\n")



#angleのマップを月ごとに出力して保存するコード
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

#start_list = [20130101]
for i, start in enumerate(start_list):
	print("******************  {}/{}  *******************".format(i+1, M))
	date_ax, date_ax_str, skipping_date_str, data = main_data(
		start, start, 
		span=30, 
		get_columns=["coeff"], 
		region=None, 
		accumulate=False
		)
	#print(data.head())
	#data = data["A"]
	#print(len(data))

	# latlon_exで絞り込む場合，ここに処理を書く
	#data = pd.concat([latlon_ex, data], axis=1)
	#print(data.head())
	
	data.loc[((data.A<0) & (data.angle<0))] += 180
	data.loc[((data.A<0) & (data.angle>0))] -= 180
	
	data[data.data_idx==0.] = np.nan
	save_name = "../result/angle_30/angle_30_" + str(start)[:6] + ".png"

	#visualize.pyで関数を選ぶ
	visualize.plot_map_once(
		data["angle"],
		data_type="type_non_wind",
		save_name=save_name,
		show=False, 
		vmax=180, 
		vmin=-180
		)
	print("\n")



##################################################################################################################

#A_by_dayのマップを月ごとに出力して保存するコード
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

#start_list = [20030101]
plot_kw = "A_by_day"
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
	data_array_1 = np.ma.masked_invalid(data_array)
	data_count_nan = np.sum(data_array_1.recordmask, axis=0)
	#print(data_count_nan)
	#print(len(date_ax))
	#print(len(date_ax_str))
	data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
	#A_by_dayなので0列目
	data_ave = pd.DataFrame(data_ave[:, 0])
	#print(data_ave)
	data_ave.columns = [plot_kw]
	# 閾値を設ける場合
	#data_ave.loc[data_ave.plot_kw>=0.05, :] = np.nan

	# data_aveにLabelとかNameをくっつける場合、以下のdataをmain_plotに渡す
	# Labelなどで絞り込む場合は、ここに操作を付け足す
	#data = pd.concat([latlon_ex, data], axis=1)

	save_name = "../result/A_by_day_30/A_by_day_30_" + str(start)[:6] + ".png"
	visualize.plot_map_once(
		data_ave["A_by_day"],
		data_type="type_non_wind", 
		save_name=save_name,
		show=False, 
		vmax=0.025, 
		vmin=None
		)
	print("\n")


##################################################################################################################

#散布図：Aとic0
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

#start_list = start_list[:15]
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
	data_A[data_A.data_idx==0.] = np.nan


	date_ax, _, skipping_date_str, data_ic0_30 = main_data(
		start, end, 
		span=30, 
		get_columns=["ic0_145"], 
		region=None, 
		accumulate=True
		)

	data_array = np.array(data_ic0_30)/100
	data_array_1 = np.ma.masked_invalid(data_array)
	data_count_nan = np.sum(data_array_1.recordmask, axis=0)
	data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
	data_ave = pd.DataFrame(data_ave)
	data_ave.columns = ["ic0_30"]

	data = pd.concat([latlon_ex, data_A, data_ave], axis=1)
	data = data[data.Name=="north_polar"]
	print(data.dropna().head())

	save_name = "../result/scatter_A_30_and_ic0/A_30_and_ic0_" + str(start)[:6] + ".png"
	#visualize.pyで関数を選ぶ
	visualize.visual_non_line(
		data,
		mode=["scatter", ["ic0_30", "A"]],
		save_name=save_name,
		show=False
		)
	print("\n")




##################################################################################################################

#木村さんのmean_w_uと３０日平均のw_uがあっているかの確認
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

#start_list = start_list[:5]
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
	data_ave = np.sum(data_array, axis=0) / (len(date_ax)-data_count_nan)
	data_ave = data_ave[:,1]
	data_ave = pd.DataFrame(data_ave)
	data_ave.columns = ["w_u_30"]

	data = pd.concat([data_mean_w_u, data_ave], axis=1)

	data.to_csv("test_w_u.csv", index=False)

	print("\n")











