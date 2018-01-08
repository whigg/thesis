
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ocean_array = np.array(pd.read_csv("../../data/ocean_grid_145.csv", header=None), dtype="int64").ravel()
ocean_array_index = np.where(ocean_array==1)[0]

df1 = pd.read_csv('../../data/latlon.csv', header=None)
df1.columns = ["idx1", "idx2", "Lat", "Lon"]

df_latlon = df1.loc[:,["Lat", "Lon"]]

label_array = np.array([15]*(145*145))
name_list = ["other_land"]*(145*145)


bearing_sea = df_latlon[(df_latlon.Lat>=54.5) & ((df_latlon.Lat<65)) & (((df_latlon.Lon>=-180) & (df_latlon.Lon<=-157)) 
	| ((df_latlon.Lon>=162) & (df_latlon.Lon<=180)))]
bearing_sea_index = np.array(list(set(bearing_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[bearing_sea_index.tolist()] = 0
for idx in bearing_sea_index.tolist():
	name_list[idx] = "bearing_sea"
bearing_sea = df_latlon.iloc[bearing_sea_index]

chukchi_sea = df_latlon[(df_latlon.Lat>=65) & (df_latlon.Lat<=72) & (df_latlon.Lon>=-180) & (df_latlon.Lon<=-160)]
chukchi_sea_index = np.array(list(set(chukchi_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[chukchi_sea_index.tolist()] = 1
for idx in chukchi_sea_index.tolist():
	name_list[idx] = "chukchi_sea"
chukchi_sea = df_latlon.iloc[chukchi_sea_index]

beaufort_sea = df_latlon[(df_latlon.Lat>=68) & (df_latlon.Lat<=75) & (df_latlon.Lon>=-160) & (df_latlon.Lon<=-122)]
beaufort_sea_index = np.array(list(set(beaufort_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[beaufort_sea_index.tolist()] = 2
for idx in beaufort_sea_index.tolist():
	name_list[idx] = "beaufort_sea"
beaufort_sea = df_latlon.iloc[beaufort_sea_index]

canada_islands = df_latlon[(df_latlon.Lat>=65) & (df_latlon.Lat<=80) & (df_latlon.Lon>=-122) & (df_latlon.Lon<=-77)]
canada_islands_index = np.array(list(set(canada_islands.index.tolist())&set(ocean_array_index.tolist())))
label_array[canada_islands_index.tolist()] = 3
for idx in canada_islands_index.tolist():
	name_list[idx] = "canada_islands"
canada_islands = df_latlon.iloc[canada_islands_index]

hudson_bay = df_latlon[(df_latlon.Lat>=54) & (df_latlon.Lat<=65) & (df_latlon.Lon>=-100) & (df_latlon.Lon<=-75)]
hudson_bay_index = np.array(list(set(hudson_bay.index.tolist())&set(ocean_array_index.tolist())))
label_array[hudson_bay_index.tolist()] = 4
for idx in hudson_bay_index.tolist():
	name_list[idx] = "hudson_bay"
hudson_bay = df_latlon.iloc[hudson_bay_index]

buffin_bay = df_latlon[(df_latlon.Lat>=65) & (df_latlon.Lat<=80) & (df_latlon.Lon>=-77) & (df_latlon.Lon<=-45)]
buffin_bay_index = np.array(list(set(buffin_bay.index.tolist())&set(ocean_array_index.tolist())))
label_array[buffin_bay_index.tolist()] = 5
for idx in buffin_bay_index.tolist():
	name_list[idx] = "buffin_bay"
buffin_bay = df_latlon.iloc[buffin_bay_index]

labrador_sea = df_latlon[(df_latlon.Lat>=57.5) & (df_latlon.Lat<=65) & (df_latlon.Lon>=-77) & (df_latlon.Lon<=-45)]
labrador_sea_index = np.array(list(set(labrador_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[labrador_sea_index.tolist()] = 6
for idx in labrador_sea_index.tolist():
	name_list[idx] = "labrador_sea"
labrador_sea = df_latlon.iloc[labrador_sea_index]

greenland_sea = df_latlon[(df_latlon.Lat>=72) & (df_latlon.Lat<=80) & (((df_latlon.Lon>=-30) & (df_latlon.Lon<=0)) 
	| ((df_latlon.Lon>=0) & (df_latlon.Lon<=25)))]
greenland_sea_index = np.array(list(set(greenland_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[greenland_sea_index.tolist()] = 7
for idx in greenland_sea_index.tolist():
	name_list[idx] = "greenland_sea"
greenland_sea = df_latlon.iloc[greenland_sea_index]

norwegian_sea = df_latlon[(df_latlon.Lat>=66) & (df_latlon.Lat<=72) & (((df_latlon.Lon>=-30) & (df_latlon.Lon<=0))
	| ((df_latlon.Lon>=0) & (df_latlon.Lon<=25)))]
norwegian_sea_index = np.array(list(set(norwegian_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[norwegian_sea_index.tolist()] = 8
for idx in norwegian_sea_index.tolist():
	name_list[idx] = "norwegian_sea"
norwegian_sea = df_latlon.iloc[norwegian_sea_index]

barents_sea = df_latlon[(df_latlon.Lat>=63) & (df_latlon.Lat<=75) & (df_latlon.Lon>=25) & (df_latlon.Lon<=58)]
barents_sea_index = np.array(list(set(barents_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[barents_sea_index.tolist()] = 9
for idx in barents_sea_index.tolist():
	name_list[idx] = "barents_sea"
barents_sea = df_latlon.iloc[barents_sea_index]

kara_sea = df_latlon[(df_latlon.Lat>=63) & (df_latlon.Lat<=75) & (df_latlon.Lon>=58) & (df_latlon.Lon<=90)]
kara_sea_index = np.array(list(set(kara_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[kara_sea_index.tolist()] = 10
for idx in kara_sea_index.tolist():
	name_list[idx] = "kara_sea"
kara_sea = df_latlon.iloc[kara_sea_index]

laptev_sea = df_latlon[(df_latlon.Lat>=63) & (df_latlon.Lat<=75) & (df_latlon.Lon>=110) & (df_latlon.Lon<=142)]
laptev_sea_index = np.array(list(set(laptev_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[laptev_sea_index.tolist()] = 11
for idx in laptev_sea_index.tolist():
	name_list[idx] = "laptev_sea"
laptev_sea = df_latlon.iloc[laptev_sea_index]

east_siberian_sea = df_latlon[(df_latlon.Lat>=67) & (df_latlon.Lat<=73) & (df_latlon.Lon>=142) & (df_latlon.Lon<=180)]
east_siberian_sea_index = np.array(list(set(east_siberian_sea.index.tolist())&set(ocean_array_index.tolist())))
label_array[east_siberian_sea_index.tolist()] = 12
for idx in east_siberian_sea_index.tolist():
	name_list[idx] = "east_siberian_sea"
east_siberian_sea = df_latlon.iloc[east_siberian_sea_index]

north_polar = df_latlon[df_latlon.Lat>=70].index
north_polar_index = np.where(label_array==15)[0]
north_polar_index = np.array(list(set(north_polar_index.tolist())&set(ocean_array_index.tolist())&set(north_polar.tolist())))
label_array[north_polar_index.tolist()] = 13
for idx in north_polar_index.tolist():
	name_list[idx] = "north_polar"
north_polar = df_latlon.iloc[north_polar_index]

other_sea_index = np.where(label_array==15)[0]
other_sea_index = np.array(list(set(other_sea_index.tolist())&set(ocean_array_index.tolist())))
label_array[other_sea_index.tolist()] = 14
for idx in other_sea_index.tolist():
	name_list[idx] = "other_sea"
other_sea = df_latlon.iloc[other_sea_index]

other_land_index = np.where(label_array==15)[0]
other_land = df_latlon.iloc[other_land_index]


df_label = pd.DataFrame({
	'Label': label_array, 
	'Name': name_list
	})


data = pd.concat([df1.loc[:,["idx1", "idx2"]], df_latlon, df_label], axis=1)
#print (data.head())
#csvに書き出し
data.to_csv("latlon_ex.csv", index=False)
#print (data[data.Name=="north_polar"])
###################################################################
m = Basemap(lon_0=180,boundinglat=50,
            resolution='h',projection='npstere')
fig=plt.figure(figsize=(7.5, 7.5))

colors = plt.cm.jet(np.linspace(0, 1, 16))

#グリッドの描画
"""
x1, y1 = m(np.array(bearing_sea.Lon), np.array(bearing_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[0], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(chukchi_sea.Lon), np.array(chukchi_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[1], label=1, s=0.6, alpha=0.9)
x1, y1 = m(np.array(beaufort_sea.Lon), np.array(beaufort_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[2], label=2, s=0.6, alpha=0.9)
x1, y1 = m(np.array(canada_islands.Lon), np.array(canada_islands.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[3], label=3, s=0.6, alpha=0.9)
x1, y1 = m(np.array(hudson_bay.Lon), np.array(hudson_bay.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[4], label=4, s=0.6, alpha=0.9)
x1, y1 = m(np.array(buffin_bay.Lon), np.array(buffin_bay.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[5], label=5, s=0.6, alpha=0.9)
x1, y1 = m(np.array(labrador_sea.Lon), np.array(labrador_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[6], label=6, s=0.6, alpha=0.9)
x1, y1 = m(np.array(greenland_sea.Lon), np.array(greenland_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[7], label=7, s=0.6, alpha=0.9)
x1, y1 = m(np.array(norwegian_sea.Lon), np.array(norwegian_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[8], label=8, s=0.6, alpha=0.9)
x1, y1 = m(np.array(barents_sea.Lon), np.array(barents_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[9], label=9, s=0.6, alpha=0.9)
x1, y1 = m(np.array(kara_sea.Lon), np.array(kara_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[10], label=10, s=0.6, alpha=0.9)
x1, y1 = m(np.array(laptev_sea.Lon), np.array(laptev_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[11], label=11, s=0.6, alpha=0.9)
x1, y1 = m(np.array(east_siberian_sea.Lon), np.array(east_siberian_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[12], label=12, s=0.6, alpha=0.9)
x1, y1 = m(np.array(north_polar.Lon), np.array(north_polar.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[13], label=13, s=0.6, alpha=0.9)
"""
"""
x1, y1 = m(np.array(other_sea.Lon), np.array(other_sea.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[14], label=14, s=0.6, alpha=0.9)
x1, y1 = m(np.array(other_land.Lon), np.array(other_land.Lat), inverse=False)
m.scatter(x1, y1, marker='o', color = "k", label=15, s=0.6, alpha=0.9)
"""


x1, y1 = m(np.array(data.Lon[data.Name=="bearing_sea"]), np.array(data.Lat[data.Name=="bearing_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[0], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="chukchi_sea"]), np.array(data.Lat[data.Name=="chukchi_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[1], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="beaufort_sea"]), np.array(data.Lat[data.Name=="beaufort_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[2], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="canada_islands"]), np.array(data.Lat[data.Name=="canada_islands"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[3], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="hudson_bay"]), np.array(data.Lat[data.Name=="hudson_bay"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[4], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="buffin_bay"]), np.array(data.Lat[data.Name=="buffin_bay"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[5], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="labrador_sea"]), np.array(data.Lat[data.Name=="labrador_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[6], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="greenland_sea"]), np.array(data.Lat[data.Name=="greenland_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[7], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="norwegian_sea"]), np.array(data.Lat[data.Name=="norwegian_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[8], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="barents_sea"]), np.array(data.Lat[data.Name=="barents_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[9], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="kara_sea"]), np.array(data.Lat[data.Name=="kara_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[10], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="laptev_sea"]), np.array(data.Lat[data.Name=="laptev_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[11], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="east_siberian_sea"]), np.array(data.Lat[data.Name=="east_siberian_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[12], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="north_polar"]), np.array(data.Lat[data.Name=="north_polar"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[13], label=0, s=0.6, alpha=0.9)

"""
x1, y1 = m(np.array(data.Lon[data.Name=="other_sea"]), np.array(data.Lat[data.Name=="other_sea"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[14], label=0, s=0.6, alpha=0.9)
x1, y1 = m(np.array(data.Lon[data.Name=="other_land"]), np.array(data.Lat[data.Name=="other_land"]), inverse=False)
m.scatter(x1, y1, marker='o', color = colors[15], label=0, s=0.6, alpha=0.9)
"""

m.drawcoastlines(color = '0.15')
m.fillcontinents(color='#555555')
m.drawparallels(np.arange(-80.,101.,5.), labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,181.,15.), labels=[0,1,1,0])

plt.show()
