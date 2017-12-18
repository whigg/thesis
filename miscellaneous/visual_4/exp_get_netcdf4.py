#netcdf4データ
#日付を指定すると，データを取ってくる構造にする
import netCDF4
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, date, timezone, timedelta

df = pd.read_csv("../data/latlon_ex.csv")
df_lat = np.array(df["Lat"])
df_lon = np.array(df["Lon"])
unreliable_index = df[df.Lat<=49].index

nc = netCDF4.Dataset("interim_2mt_10u_10v_20030101-20031231.nc", "r")
time_var = nc.variables["time"]
dtime = netCDF4.num2date(time_var[:],time_var.units)

#start: ファイルのはじめの日にち
#end: 取り出したい日付
start = 20030101
end = 20030103
start_date = [start//10000, (start%10000)//100, (start%10000)%100]
end_date = [end//10000, (end%10000)//100, (end%10000)%100]
d1 = datetime(start_date[0], start_date[1], start_date[2])
d2 = datetime(end_date[0], end_date[1], end_date[2])
L = (d2-d1).days+1

nc_lon = nc["longitude"][:]
nc_lat = nc["latitude"][:]
nc_u10 = nc["u10"][:]
nc_v10 = nc["v10"][:]
nc_t2m = nc["t2m"][:]

idx_lon = df["Lon"].apply(lambda x: np.argmin(np.absolute(nc_lon-x)))
idx_lat = df["Lat"].apply(lambda x: np.argmin(np.absolute(nc_lat-x)))

day_n = L-1
#print(nc_t2m[day_n,:,:]-273)
result_t2m = nc_t2m[day_n][idx_lat, idx_lon]-273
result_u10 = nc_u10[day_n][idx_lat, idx_lon]
result_v10 = nc_v10[day_n][idx_lat, idx_lon]

result_t2m[unreliable_index] = np.nan
result_u10[unreliable_index] = np.nan
result_v10[unreliable_index] = np.nan

#######################################################################################
result_t2m = np.ma.masked_invalid(result_t2m)
result_u10 = np.ma.masked_invalid(result_u10)
result_v10 = np.ma.masked_invalid(result_v10)
result_t2m_r = np.reshape(result_t2m, (145, 145), order="F")
result_u10_r = np.reshape(result_u10, (145, 145), order="F")
result_v10_r = np.reshape(result_v10, (145, 145), order="F")

m = Basemap(lon_0=180, boundinglat=40, resolution='l', projection='npstere')
fig = plt.figure(figsize=(6, 6))
m.drawcoastlines(color='0.15')

x, y = m(df_lon, df_lat)
x1 = np.reshape(x, (145, 145), order="F")
y1 = np.reshape(y, (145, 145), order="F")
x_idx, y_idx = m(nc_lon[idx_lon], nc_lat[idx_lat])

#グリッドの描画
#m.plot(x, y, "bo", markersize=0.4)
#m.plot(x_idx, y_idx, "ro", markersize=0.4)

#t2mの可視化
#m.contourf(x1, y1, result_t2m_r, cmap=plt.cm.jet)
#m.colorbar(location="bottom")

#u10, v10の可視化
u_out, v_out = m.rotate_vector(result_u10, result_v10, df_lon, df_lat, returnxy=False)
m.quiver(x, y, u_out, v_out, angles='xy',scale_units='xy')

plt.show()

