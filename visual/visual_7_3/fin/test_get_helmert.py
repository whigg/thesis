




start_list = [20030101]
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
	data_array_w_1 = np.ma.masked_invalid(data_array_w)
	data_w_count_nan = np.sum(data_array_w_1.recordmask, axis=0)
	date_ax_w_len = len(date_ax_w)
	data_array_w[date_ax_w_len-data_w_count_nan<=20] = np.nan
	
	data_ave_w_sum = np.sum(data_array_w, axis=0)
	#data_ave_sum[date_ax_len-data_count_nan<=20] = np.nan
	data_ave_w = data_ave_w_sum / (date_ax_w_len-data_w_count_nan)
	#data_ave = pd.DataFrame(data_ave)
	#data_ave.columns = ["w_speed", "w_u", "w_v"]
	#print(data_ave.head(3))

	#iwデータの取得・整形
	date_ax_iw, _, _, data_iw = main_data(
		start, end, 
		span=30, 
		get_columns=["iw"], 
		region=None, 
		accumulate=True
		)

	data_array_iw = np.array(data_iw)
	data_array_iw_1 = np.ma.masked_invalid(data_array_iw)
	data_iw_count_nan = np.sum(data_array_iw_1.recordmask, axis=0)
	date_ax_iw_len = len(date_ax_iw)
	data_array_iw[date_ax_iw_len-data_iw_count_nan<=20] = np.nan
	
	data_ave_iw_sum = np.sum(data_array_iw, axis=0)
	#data_ave_iw_sum[date_ax_iw_len-data_count_nan<=20] = np.nan
	data_ave_iw = data_ave_iw_sum / (date_ax_iw_len-data_iw_count_nan)
	#data_ave = pd.DataFrame(data_ave)
	#data_ave.columns = ["w_speed", "w_u", "w_v"]
	#print(data_ave.head(3))
	

	print("data_array_w:  {}".format(data_array_w.shape))
	print("data_array_iw:  {}".format(data_array_iw.shape))
	print("data_array_iw[:,:,0]:  {}".format(data_array_iw[:,:,0].shape))
	print("data_array_w[:,0,0]:  {}".format(data_array_w[:,0,0].shape))
	N = month_end.day
	print(N)
	one_N = np.ones(N)
	zero_N = np.zeros(N)

	w_array = np.vstack((data_array_iw[:,:,0], data_array_iw[:,:,1]))
	Helmert = []
	for i in range(2):
	# for i in range(145**2):
		#欠損データの処理
		not_nan_idx = np.sum(np.isnan(data_array_iw[:, i, :]), axis=1)==False
		x = data_array_w[:, i, 0][not_nan_idx]
		y = data_array_w[:, i, 1][not_nan_idx]
		w = w_array[:, i][not_nan_idx]
		N_c = len(not_nan_idx)
		one_N = np.ones(N_c)
		zero_N = np.zeros(N_c)
		print("i: {}".format(i))
		print("\tx: {}\n\ty: {}\n\tw: {}\n\tN_c: {}".format(x.shape, y.shape, w.shape, N_c))
		D_1 = np.hstack((one_N, zero_N, x, -y))
		D_2 = np.hstack((zero_N, one_N, y, x))
		Dt = np.vstack((D_1, D_2))
		print("\tDt: {}".format(i, Dt.shape))
		D_inv = np.linalg.inv(np.dot(Dt.T, Dt))
		gamma = np.dot(D_inv, np.dot(Dt.T, w))
		print("\tgamma: {}, {}".format(gamma, gamma.shape))
		A = np.sqrt(gamma[2]**2 + gamma[3]**2)
		theta = np.arctan2(gamma[3], gamme[2])
		R_denominator = np.dot(w.T, w) - N_c*np.sum(data_ave_iw[i, :]**2)
		epsilon = w - np.dot(Dt, gamma)
		R_numerator = np.dot(epsilon.T, epsilon)
		R2 = 1 - R_numerator/R_denominator
		Helmert.append([A, theta, gamma[0], gamma[1], R2, N_c])

	result = np.hstack(Helmert, data_ave_w[:,[1,2]], data_ave_iw[:,[1,2]])





































