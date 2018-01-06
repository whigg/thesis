
def get_helmert():
	dirs = "../data/csv_Helmert_30/"
	mkdir(dirs)

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
		#for j in range(1218,1220):
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_array_iw[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_array_w[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_array_w[:, j, 2][not_nan_idx].reshape((-1,1))
			w = w_array[:, j][np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_array_iw[:, j, 1])
			iw_v_ave = np.nanmean(data_array_iw[:, j, 2])
			N_c = np.sum(not_nan_idx==True)
			if N_c <= 1:
				print("\tskipping for loop...")
				Helmert.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c, np.nan, np.nan])
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
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_30_" + str(start)[:6] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)


#get_helmert()




def get_hermert_by_year():
	dirs = "../data/csv_Helmert_by_year/"
	mkdir(dirs)

	start_list = []
	n = 20000000
	y_list = [3,4,5,6,7,8,9,10,13,14,15,16]
	for i in y_list:
		m = n + i*10000
		for j in range(12):
			start_list.append(m + (j+1)*100 + 1)
	start_list_plus_1month = start_list.copy()
	start_list_plus_1month.append(201701)

	start_list = np.array(start_list)
	month_list_str = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
	for k in range(12):
		print("************************  Month: {}/{}  ************************".format(k+1, 12))
		#12年分
		month_idx = np.arange(0, 144, 12) + k
		month_next_idx = np.arange(0, 144, 12) + k + 1
		year_list = start_list[month_idx]
		y_next_list = start_list[month_next_idx]

		data_w_year = np.zeros((1, 145**2, 2))
		data_iw_year = np.zeros((1, 145**2, 2))
		for i, start in enumerate(year_list):
			print("  *******************  Year: {}  *******************".format(str(start)[:6]))
			month_end = y_next_list[i]
			month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
			end = start + month_end.day - 1

			_, _, _, data_w = main_data(
				start, end, 
				span=30, 
				get_columns=["w"], 
				region=None, 
				accumulate=True
				)
			data_array_w = np.array(data_w)
			data_w_year = np.concatenate([data_w_year, data_array_w], axis=0)

			_, _, _, data_iw = main_data(
				start, end, 
				span=30, 
				get_columns=["iw"], 
				region=None, 
				accumulate=True
				)
			data_array_iw = np.array(data_iw)
			data_iw_year = np.concatenate([data_iw_year, data_array_iw], axis=0)

			print("\n")

		data_w_year = data_w_year[1:, :, :]
		data_iw_year = data_iw_year[1:, :, :]
		data_ave_w = np.nanmean(data_w_year, axis=0)
		data_ave_iw = np.nanmean(data_iw_year, axis=0)
		w_array = np.vstack((data_iw_year[:,:,1], data_iw_year[:,:,2]))

		Helmert = []
		#for j in range(1218,1220):
		for j in range(145**2):
			print("j: {}".format(j))
			#欠損データの処理
			not_nan_idx = np.sum(np.isnan(data_iw_year[:, j, :]), axis=1)==False
			#print("\tnot_nan_idx: {}".format(not_nan_idx))
			x = data_w_year[:, j, 1][not_nan_idx].reshape((-1,1))
			y = data_w_year[:, j, 2][not_nan_idx].reshape((-1,1))
			w = w_array[:, j][np.tile(not_nan_idx, 2)].reshape((-1,1))
			iw_u_ave = np.nanmean(data_iw_year[:, j, 1])
			iw_v_ave = np.nanmean(data_iw_year[:, j, 2])
			N_c = np.sum(not_nan_idx==True)
			if N_c <= 1:
				print("\tskipping for loop...")
				Helmert.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, N_c, np.nan, np.nan])
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
			if N_c <= 120:
				Helmert.append([np.nan, np.nan, gamma[0], gamma[1], np.nan, np.nan, N_c, iw_u_ave, iw_v_ave])
			else:
				Helmert.append([A, theta, gamma[0], gamma[1], R2, R_numerator, N_c, iw_u_ave, iw_v_ave])

		result = np.hstack((Helmert, data_ave_w[:,[1,2]]))
		#print(result.shape)
		data = pd.DataFrame(result)
		data.columns = ["A", "theta", "ocean_u", "ocean_v", "R2", "epsilon2", "N_c", "mean_iw_u", "mean_iw_v", "mean_w_u", "mean_w_v"]
		#print(data.head(3))
		save_name = dirs + "Helmert_by_year_" + month_list_str[k] + ".csv"
		print(save_name)
		data.to_csv(save_name, index=False)



