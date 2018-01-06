
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
start_list_plus_1month = start_list + [20170901]


###############################################################################################################

def divide_area():
	ocean_array = np.array(pd.read_csv("../data/ocean_grid_145.csv", header=None), dtype="int64").ravel()
	ocean_array_index = np.where(ocean_array==1)[0]

	df1 = pd.read_csv('../data/latlon.csv', header=None)
	df1.columns = ["idx1", "idx2", "Lat", "Lon"]
	df_latlon = df1.loc[:,["Lat", "Lon"]]

	label_array = np.array([99]*(145**2))

	area_0 = df_latlon[(df_latlon.Lat>=75) & (df_latlon.Lat<=80) & (df_latlon.Lon>=-180) & (df_latlon.Lon<-122)]
	area_0_idx = np.array(list(set(area_0.index.tolist())&set(ocean_array_index.tolist())))
	label_array[area_0_idx.tolist()] = 0
	area_0 = df_latlon.iloc[area_0_idx]

	area_1 = df_latlon[(df_latlon.Lat>=65) & (df_latlon.Lat<75) & (df_latlon.Lon>=-180) & (df_latlon.Lon<-122)]
	area_1_idx = np.array(list(set(area_1.index.tolist())&set(ocean_array_index.tolist())))
	label_array[area_1_idx.tolist()] = 1
	area_1 = df_latlon.iloc[area_1_idx]

	bearing_sea = df_latlon[(df_latlon.Lat>=54.5) & ((df_latlon.Lat<65)) & (((df_latlon.Lon>=-180) & (df_latlon.Lon<=-157)) 
		| ((df_latlon.Lon>=162) & (df_latlon.Lon<=180)))]
	bearing_sea_index = np.array(list(set(bearing_sea.index.tolist())&set(ocean_array_index.tolist())))
	label_array[bearing_sea_index.tolist()] = 2
	bearing_sea = df_latlon.iloc[bearing_sea_index]

	canada_islands = df_latlon[(df_latlon.Lat>=65) & (df_latlon.Lat<=80) & (df_latlon.Lon>=-122) & (df_latlon.Lon<-75)]
	canada_islands_index = np.array(list(set(canada_islands.index.tolist())&set(ocean_array_index.tolist())))
	label_array[canada_islands_index.tolist()] = 3
	canada_islands = df_latlon.iloc[canada_islands_index]

	hudson_bay = df_latlon[(df_latlon.Lat>=54) & (df_latlon.Lat<=65) & (df_latlon.Lon>=-100) & (df_latlon.Lon<=-75)]
	hudson_bay_index = np.array(list(set(hudson_bay.index.tolist())&set(ocean_array_index.tolist())))
	label_array[hudson_bay_index.tolist()] = 4
	hudson_bay = df_latlon.iloc[hudson_bay_index]

	buffin_bay = df_latlon[(df_latlon.Lat>=65) & (df_latlon.Lat<=80) & (df_latlon.Lon>=-75) & (df_latlon.Lon<=-45)]
	buffin_bay_index = np.array(list(set(buffin_bay.index.tolist())&set(ocean_array_index.tolist())))
	label_array[buffin_bay_index.tolist()] = 5
	buffin_bay = df_latlon.iloc[buffin_bay_index]

	labrador_sea = df_latlon[(df_latlon.Lat>=57.5) & (df_latlon.Lat<=65) & (df_latlon.Lon>=-75) & (df_latlon.Lon<=-50)]
	labrador_sea_index = np.array(list(set(labrador_sea.index.tolist())&set(ocean_array_index.tolist())))
	label_array[labrador_sea_index.tolist()] = 6
	labrador_sea = df_latlon.iloc[labrador_sea_index]

	greenland_sea = df_latlon[(df_latlon.Lat>=70) & (df_latlon.Lat<=81.5) & (((df_latlon.Lon>=-30) & (df_latlon.Lon<=0)) 
		| ((df_latlon.Lon>=0) & (df_latlon.Lon<=22.5)))]
	greenland_sea_index = np.array(list(set(greenland_sea.index.tolist())&set(ocean_array_index.tolist())))
	label_array[greenland_sea_index.tolist()] = 7
	greenland_sea = df_latlon.iloc[greenland_sea_index]

	area_8 = df_latlon[(df_latlon.Lat>=75) & (df_latlon.Lat<81.5) & (df_latlon.Lon>=22.5) & (df_latlon.Lon<60)]
	area_8_idx = np.array(list(set(area_8.index.tolist())&set(ocean_array_index.tolist())))
	label_array[area_8_idx.tolist()] = 8
	area_8 = df_latlon.iloc[area_8_idx]

	barents_sea = df_latlon[(df_latlon.Lat>=63) & (df_latlon.Lat<=75) & (df_latlon.Lon>=22.5) & (df_latlon.Lon<=58)]
	barents_sea_index = np.array(list(set(barents_sea.index.tolist())&set(ocean_array_index.tolist())))
	label_array[barents_sea_index.tolist()] = 9
	barents_sea = df_latlon.iloc[barents_sea_index]

	area_10 = df_latlon[(df_latlon.Lat>=75) & (df_latlon.Lat<81.5) & (df_latlon.Lon>=60) & (df_latlon.Lon<104)]
	area_10_idx = np.array(list(set(area_10.index.tolist())&set(ocean_array_index.tolist())))
	label_array[area_10_idx.tolist()] = 10
	area_10 = df_latlon.iloc[area_10_idx]

	kara_sea = df_latlon[(df_latlon.Lat>=63) & (df_latlon.Lat<=75) & (df_latlon.Lon>=58) & (df_latlon.Lon<=90)]
	kara_sea_index = np.array(list(set(kara_sea.index.tolist())&set(ocean_array_index.tolist())))
	label_array[kara_sea_index.tolist()] = 11
	kara_sea = df_latlon.iloc[kara_sea_index]

	area_12 = df_latlon[(df_latlon.Lat>=75) & (df_latlon.Lat<80) & (df_latlon.Lon>=104) & (df_latlon.Lon<180)]
	area_12_idx = np.array(list(set(area_12.index.tolist())&set(ocean_array_index.tolist())))
	label_array[area_12_idx.tolist()] = 12
	area_12 = df_latlon.iloc[area_12_idx]

	laptev_sea = df_latlon[(df_latlon.Lat>=63) & (df_latlon.Lat<=76) & (df_latlon.Lon>=105) & (df_latlon.Lon<=142)]
	laptev_sea_index = np.array(list(set(laptev_sea.index.tolist())&set(ocean_array_index.tolist())))
	label_array[laptev_sea_index.tolist()] = 13
	laptev_sea = df_latlon.iloc[laptev_sea_index]

	east_siberian_sea = df_latlon[(df_latlon.Lat>=67) & (df_latlon.Lat<=76) & (df_latlon.Lon>=142) & (df_latlon.Lon<=180)]
	east_siberian_sea_index = np.array(list(set(east_siberian_sea.index.tolist())&set(ocean_array_index.tolist())))
	label_array[east_siberian_sea_index.tolist()] = 14
	east_siberian_sea = df_latlon.iloc[east_siberian_sea_index]

	area_15 = df_latlon[(df_latlon.Lat>=80) & (df_latlon.Lat<85) & (df_latlon.Lon>-90) & (df_latlon.Lon<=-15)]
	area_15_idx = np.array(list(set(area_15.index.tolist())&set(ocean_array_index.tolist())))
	label_array[area_15_idx.tolist()] = 15
	area_15 = df_latlon.iloc[area_15_idx]

	north_polar = df_latlon[df_latlon.Lat>=70].index
	north_polar_index = np.where(label_array==99)[0]
	north_polar_index = np.array(list(set(north_polar_index.tolist())&set(ocean_array_index.tolist())&set(north_polar.tolist())))
	label_array[north_polar_index.tolist()] = 16
	north_polar = df_latlon.iloc[north_polar_index]

	other_sea_index = np.where(label_array==99)[0]
	other_sea_index = np.array(list(set(other_sea_index.tolist())&set(ocean_array_index.tolist())))
	label_array[other_sea_index.tolist()] = 17
	other_sea = df_latlon.iloc[other_sea_index]

	other_land_index = np.where(label_array==99)[0]
	other_land = df_latlon.iloc[other_land_index]
	

	df_label = pd.DataFrame({
		'Area': label_array
		})
	data = pd.concat([df1.loc[:,["idx1", "idx2"]], df_latlon, df_label], axis=1)
	#print (data.head())
	#csvに書き出し
	#data.to_csv("latlon_ex.csv", index=False)
	#print (data[data.Name=="north_polar"])
	###################################################################
	"""
	m = Basemap(lon_0=180,boundinglat=50,
	            resolution='h',projection='npstere')
	fig = plt.figure(figsize=(8.5, 8.5))
	#colors = plt.cm.gist_ncar(np.linspace(0, 1, 20))
	#colors = plt.cm.tab20
	import random
	#idx_random = list(range(17))
	#random.shuffle(idx_random)
	#random.shuffle(colors)
	#colors = [""]
	from matplotlib.colors import LinearSegmentedColormap

	def generate_cmap(colors):
	    values = range(len(colors))
	    vmax = np.ceil(np.max(values))
	    color_list = []
	    for v, c in zip(values, colors):
	        color_list.append( ( v/ vmax, c) )
	    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

	cm = generate_cmap(['mediumblue', 'limegreen', 'orangered', 'indigo', 'white', 'salmon'])
	colors = cm(np.linspace(0, 1, 20))
	random.shuffle(colors)
	
	x1, y1 = m(np.array(data.Lon[data.Area==0]), np.array(data.Lat[data.Area==0]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[0], label=0, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==1]), np.array(data.Lat[data.Area==1]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[1], label=1, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==2]), np.array(data.Lat[data.Area==2]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[2], label=2, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==3]), np.array(data.Lat[data.Area==3]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[3], label=3, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==4]), np.array(data.Lat[data.Area==4]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[4], label=4, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==5]), np.array(data.Lat[data.Area==5]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[5], label=5, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==6]), np.array(data.Lat[data.Area==6]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[6], label=6, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==7]), np.array(data.Lat[data.Area==7]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[7], label=7, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==8]), np.array(data.Lat[data.Area==8]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[8], label=8, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==9]), np.array(data.Lat[data.Area==9]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[9], label=9, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==10]), np.array(data.Lat[data.Area==10]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[10], label=10, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==11]), np.array(data.Lat[data.Area==11]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[11], label=11, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==12]), np.array(data.Lat[data.Area==12]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[12], label=12, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==13]), np.array(data.Lat[data.Area==13]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[13], label=13, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==14]), np.array(data.Lat[data.Area==14]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[14], label=14, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==15]), np.array(data.Lat[data.Area==15]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[15], label=15, s=0.9, alpha=0.9)
	x1, y1 = m(np.array(data.Lon[data.Area==16]), np.array(data.Lat[data.Area==16]), inverse=False)
	m.scatter(x1, y1, marker='o', color = colors[16], label=16, s=0.9, alpha=0.9)

	m.drawcoastlines(color = '0.15')
	m.fillcontinents(color='#555555')
	m.drawparallels(np.arange(-80.,101.,5.), labels=[1,0,0,0])
	m.drawmeridians(np.arange(-180.,181.,15.), labels=[0,1,1,0])

	plt.show()
	"""

	return label_array



def get_coastal_region():
	"""
	海岸付近の海域はTrue(1)、それ以外はFalse
	"""
	"""
	def distance_from_coast(lon,lat,resolution='l',degree_in_km=111.12):
	    plt.ioff()
	    m = Basemap(projection='robin',lon_0=0,resolution=resolution)
	    coast = m.drawcoastlines()
	    coordinates = np.vstack(coast.get_segments())
	    lons,lats = m(coordinates[:,0],coordinates[:,1],inverse=True)
	    dists = np.sqrt((lons-lon)**2+(lats-lat)**2)
	    if np.min(dists)*degree_in_km<1:
	      return True
	    else:
	      return False
	"""
	#print(distance_from_coast(-117.2547,32.8049,resolution="i"))
	latlon145_file_name = calc_data.latlon145_file_name
	latlon900_file_name = calc_data.latlon900_file_name
	grid900to145_file_name = calc_data.grid900to145_file_name
	ocean_grid_file = calc_data.ocean_grid_file
	ocean_grid_145 = calc_data.ocean_grid_145
	ocean_idx = calc_data.ocean_idx

	latlon_ex = calc_data.get_lonlat_data()

	is_near_coastal = np.zeros(145**2)
	ocean_grid_145 = ocean_grid_145.values
	ocean_grid_145 *= -1
	ocean_grid_145 += 1
	ocean_grid_145 = np.reshape(ocean_grid_145, (145,145), order="F")
	#print(ocean_grid_145)
	ocean_x_pos_1 = np.hstack((np.ones((145,1)), ocean_grid_145))[:,:-1]
	ocean_x_pos_2 = np.hstack((np.ones((145,1)), ocean_x_pos_1))[:,:-1]
	ocean_x_pos_3 = np.hstack((np.ones((145,1)), ocean_x_pos_2))[:,:-1]
	ocean_x_neg_1 = np.hstack((ocean_grid_145, np.ones((145,1))))[:,1:]
	ocean_x_neg_2 = np.hstack((ocean_x_neg_1, np.ones((145,1))))[:,1:]
	ocean_x_neg_3 = np.hstack((ocean_x_neg_2, np.ones((145,1))))[:,1:]
	ocean_y_pos_1 = np.vstack((np.ones((1,145)), ocean_grid_145))[:-1,:]
	ocean_y_pos_2 = np.vstack((np.ones((1,145)), ocean_y_pos_1))[:-1,:]
	ocean_y_pos_3 = np.vstack((np.ones((1,145)), ocean_y_pos_2))[:-1,:]
	ocean_y_neg_1 = np.vstack((ocean_grid_145, np.ones((1,145))))[1:,:]
	ocean_y_neg_2 = np.vstack((ocean_y_neg_1, np.ones((1,145))))[1:,:]
	ocean_y_neg_3 = np.vstack((ocean_y_neg_2, np.ones((1,145))))[1:,:]
	coastal_region_1 = ocean_x_pos_1 + ocean_x_neg_1 + ocean_y_pos_1 + ocean_y_neg_1
	#print(coastal_region_1)
	coastal_region_1[ocean_grid_145==1] = -1
	coastal_region_1 = coastal_region_1>0
	#coastal_region_1 = coastal_region_1.T.ravel()
	coastal_region_2 = ocean_x_pos_1 + ocean_x_pos_2 + ocean_x_neg_1 + ocean_x_neg_2 + ocean_y_pos_1 + ocean_y_pos_2 + ocean_y_neg_1 + ocean_y_neg_2
	coastal_region_2[ocean_grid_145==1] = -1
	coastal_region_2 = coastal_region_2>0
	#coastal_region_2 = coastal_region_2.T.ravel()
	"""
	coastal_region_3 = ocean_x_pos_1 + ocean_x_pos_2 + ocean_x_pos_3 + ocean_x_neg_1 + ocean_x_neg_2 + ocean_x_neg_3 + ocean_y_pos_1 + ocean_y_pos_2 + ocean_y_pos_3 + ocean_y_neg_1 + ocean_y_neg_2 + ocean_y_neg_3
	coastal_region_3[ocean_grid_145==1] = -1
	coastal_region_3 = coastal_region_3>0
	"""
	"""
	for i in range(1,143):
		for j in range(1,143):
			if coastal_region > 0:
	"""
	"""
	m = Basemap(lon_0=180, boundinglat=50, resolution='h', projection='npstere')
	fig = plt.figure(figsize=(7.5, 7.5))
	m.drawcoastlines(color = '0.15')
	m.fillcontinents(color='#555555')
	x, y = m(np.array(latlon_ex.Lon), np.array(latlon_ex.Lat), inverse=False)
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
	#print((coastal_region_1==True).shape)
	#m.scatter(x1[coastal_region_1==True], y1[coastal_region_1==True], color="b", marker='o', s=2, alpha=0.9)
	m.pcolormesh(xx_ex-dx1, yy_ex+dy1, coastal_region_1==True)
	#m.scatter(x1[coastal_region_2==True], y1[coastal_region_2==True], color="g", marker='o', s=1.2, alpha=0.9)
	m.colorbar(location="bottom")
	plt.show()
	"""
	return coastal_region_1.T.ravel(), coastal_region_2.T.ravel()





def get_csv_ex(array_label, array_coastal_region_1, array_coastal_region_2):
	data_ex_dir = "../data/csv_Helmert_ex/"
	mkdir(data_ex_dir)
	dirs_pair = "../result_h/pairplot/"
	mkdir(dirs_pair)

	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))
		month_end = start_list_plus_1month[i+1]
		month_end = date(month_end//10000, (month_end%10000)//100, (month_end%10000)%100) - timedelta(days=1)
		end = start + month_end.day - 1

		_, _, _, data_ic0_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["ic0_145"], 
			region=None, 
			accumulate=True
			)
		data_array_ic0 = np.array(data_ic0_30)
		data_ave_ic0 = np.nanmean(data_array_ic0, axis=0).ravel()
		data_med_ic0 = np.nanmedian(data_array_ic0, axis=0).ravel()
		data_ic0_ave_med_30 = pd.DataFrame({"ic0_30": data_ave_ic0, "ic0_30_median": data_med_ic0})

		_, _, _, data_sit_30 = main_data(
			start, end, 
			span=30, 
			get_columns=["sit_145"], 
			region=None, 
			accumulate=True
			)
		data_array_sit = np.array(data_sit_30)
		data_ave_sit = np.nanmean(data_array_sit, axis=0).ravel()
		data_med_sit = np.nanmedian(data_array_sit, axis=0).ravel()
		data_sit_ave_med_30 = pd.DataFrame({"sit_30": data_ave_sit, "sit_30_median": data_med_sit})

		_, _, _, data_A_original = main_data(
			start, start, 
			span=30, 
			get_columns=["hermert"], 
			region=None, 
			accumulate=False
			)
		data_A_original["mean_ocean_speed"] = np.sqrt(data_A_original["ocean_u"]**2 + data_A_original["ocean_v"]**2)
		data_A_original["mean_iw_speed"] = np.sqrt(data_A_original["mean_iw_u"]**2 + data_A_original["mean_iw_v"]**2)
		data_A_original["mean_w_speed"] = np.sqrt(data_A_original["mean_w_u"]**2 + data_A_original["mean_w_v"]**2)
		data_A_original["w_iw_ratio"] = data_A_original["mean_iw_speed"] / data_A_original["mean_w_speed"]
		data_A_original["mean_iw_speed_no_ocean"] = np.sqrt((data_A_original["mean_iw_u"]-data_A_original["ocean_u"])**2 + (data_A_original["mean_iw_v"]-data_A_original["ocean_v"])**2)

		data = pd.concat([latlon_ex, data_A_original, data_ic0_ave_med_30, data_sit_ave_med_30], axis=1)

		#data["t_ocean"] = t_ocean
		#data["t_iw"] = t_iw
		#print(len(data))
		#print(array_coastal_region_1.shape)
		data["coastal_region_1"] = array_coastal_region_1
		data["coastal_region_2"] = array_coastal_region_2
		data["area_label"] = array_label

		#print("\n\n{}\n".format(data.head(3)))
		data_name = data_ex_dir + "Helmert_ex_" + str(start)[:-2] + ".csv"
		data.to_csv(data_name, index=False)

		data_no_nan = data_A_original.dropna()
		sns.pairplot(data_no_nan.loc[:, ["A", "R2", "epsilon2", "mean_ocean_speed"]])
		plt.savefig(dirs_pair + "ocean_" + str(start)[:-2] + ".png")
		plt.close()
		sns.pairplot(data_no_nan.loc[:, ["A", "R2", "epsilon2", "mean_iw_speed"]])
		plt.savefig(dirs_pair + "iw_" + str(start)[:-2] + ".png")
		plt.close()
		sns.pairplot(data_no_nan.loc[:, ["A", "R2", "epsilon2", "w_iw_ratio"]])
		plt.savefig(dirs_pair + "ratio_" + str(start)[:-2] + ".png")
		plt.close()
		sns.pairplot(data_no_nan.loc[:, ["A", "R2", "epsilon2", "mean_iw_speed_no_ocean"]])
		plt.savefig(dirs_pair + "iw_no_ocean_" + str(start)[:-2] + ".png")
		plt.close()
		sns.pairplot(data_no_nan.loc[:, ["mean_iw_speed", "mean_w_speed", "mean_ocean_speed", "mean_iw_speed_no_ocean"]])
		plt.savefig(dirs_pair + "speed_" + str(start)[:-2] + ".png")
		plt.close()

		print("\n")



def test_scatter():
	mkdir("../result_h/test/test_scatter/")
	for i, start in enumerate(start_list):
		print("******************  {}/{}  *******************".format(i+1, M))

		data_ex_dir = "../data/csv_Helmert_ex/Helmert_ex_" + str(start)[:-2] + ".csv"
		data = pd.read_csv(data_ex_dir)

		#data01 = data.loc[data.coastal_region_1==0, :].dropna()
		data_01 = data[((data.area_label==4)|(data.area_label==5)) & (data.coastal_region_1==False) & (data.ic0_30<99)].dropna()
		sns.jointplot(x="ic0_30", y="A", data=data_01, kind="reg")

		save_name = "../result_h/test/test_scatter/" + "A_ic0_" + str(start)[:-2] + ".png"
		plt.savefig(save_name, dpi=600)
		plt.close()
		"""
		visualize.visual_non_line(
			data,
			mode=["scatter", ["sit_30", "A"]],
			save_name=save_dir + str(start)[:6] + ".png",
			show=False
			)
		"""
		print("\n")




if __name__ == '__main__':
	"""
	・get_csv_exにcoastalを組み込む．引数の設定
	・test_scatterの完成・実行
		jointplotをベタ書きして，hueで分けるもの
		visualizeモジュールで描くもの
	"""

	#label_array = divide_area()
	#coastal_region_1, coastal_region_2 = get_coastal_region()
	#get_csv_ex(array_label=label_array, array_coastal_region_1=coastal_region_1, array_coastal_region_2=coastal_region_2)
	test_scatter()










