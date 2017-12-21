

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
datapath0 = '../../../data/'

##################################################################
def load_csv(fname):
    return pd.read_csv(fname, header=None)

##################################################################
#地衡風データの処理
def read_wind(fname):
    #fname = 'ecm030101.csv'
    df_wind = load_csv(fname)
    wind = np.array(df_wind, dtype='float32')
    w_u = wind[:,0]
    w_v = wind[:,1]
    w_speed = w_u*w_u + w_v*w_v

    return w_u, w_v, w_speed

#海氷速度データの処理
def read_ice_v(fname):
    #fname = '030101.csv'
    idx0 = np.zeros(145*145)

    df_ice_wind = load_csv(fname)
    w_true = df_ice_wind[df_ice_wind<999.].dropna()
    idx_all = range(145*145)
    idx_t = np.array(w_true.index)
    idx_f = np.sort(list(set(idx_all)-set(idx_t)))

    wind = np.array(df_ice_wind, dtype='float32')
    u = wind[:,0]
    v = wind[:,1]
    u_t = idx0
    u_t[idx_t] = u[idx_t]
    u_t[idx_f] = np.nan
    v_t = idx0
    v_t[idx_t] = v[idx_t]
    v_t[idx_f] = np.nan
    speed_t = u_t*u_t+v_t*v_t

    return u_t, v_t, speed_t

#############################################################################
def write2csv(date_array,data_list,fname):
    df = np.hstack((date_array, np.array(data_list)))
    np.savetxt(fname, df, delimiter=',')

#地衡風風データの読み込み
w_file_list = sorted(glob.glob(datapath0 + 'wind_data/*.csv'))
w_date_list, w_u_list, w_v_list, w_speed_list = [], [], [], []
wind_start_year = 2003
for filename in w_file_list:
    #print (str(20) + filename[27:33])
    date = int(str(20) + filename[27:33])
    if date < (wind_start_year+1)*10000+101:
	    w_date_list.append(date)
	    w_u, w_v, w_speed = read_wind(filename)
	    w_u, w_v, w_speed = w_u.T.tolist(), w_v.T.tolist(), w_speed.T.tolist()
	    w_u_list.append(w_u)
	    w_v_list.append(w_v)
	    w_speed_list.append(w_speed)
    else:
	    w_date_array = np.array(w_date_list).reshape(len(w_date_list),1)
	    print ("wind - " + str(wind_start_year))
	    print ("\t Writing w_u to csv...")
	    w_u_csv_name = "./ts_w_u/ts_w_u_" + str(wind_start_year) + ".csv"
	    write2csv(w_date_array,w_u_list,w_u_csv_name)

	    print ("\t Writing w_v to csv...")
	    w_v_csv_name = "./ts_w_v/ts_w_v_" + str(wind_start_year) + ".csv"
	    write2csv(w_date_array,w_v_list,w_v_csv_name)

	    print ("\t Writing w_speed to csv...")
	    w_speed_csv_name = "./ts_w_speed/ts_w_speed_" + str(wind_start_year) + ".csv"
	    write2csv(w_date_array,w_speed_list,w_speed_csv_name)

	    wind_start_year += 1
	    w_date_list, w_u_list, w_v_list, w_speed_list = [], [], [], []
	    w_date_list.append(date)
	    w_u, w_v, w_speed = read_wind(filename)
	    w_u, w_v, w_speed = w_u.T.tolist(), w_v.T.tolist(), w_speed.T.tolist()
	    w_u_list.append(w_u)
	    w_v_list.append(w_v)
	    w_speed_list.append(w_speed)
#print (np.array(w_u_list))
#print (np.array(w_u_list).shape)

#csvの書き出し
#インデックスの作成
w_date_array = np.array(w_date_list).reshape(len(w_date_list),1)
print ("wind - " + str(wind_start_year))
print ("\t Writing w_u to csv...")
w_u_csv_name = "./ts_w_u/ts_w_u_" + str(wind_start_year) + ".csv"
write2csv(w_date_array,w_u_list,w_u_csv_name)

print ("\t Writing w_v to csv...")
w_v_csv_name = "./ts_w_v/ts_w_v_" + str(wind_start_year) + ".csv"
write2csv(w_date_array,w_v_list,w_v_csv_name)

print ("\t Writing w_speed to csv...")
w_speed_csv_name = "./ts_w_speed/ts_w_speed_" + str(wind_start_year) + ".csv"
write2csv(w_date_array,w_speed_list,w_speed_csv_name)


#氷の速度データの読み込み
iw_file_list = sorted(glob.glob(datapath0 + 'ice_wind_data/*.csv'))
iw_date_list, iw_u_list, iw_v_list, iw_speed_list = [], [], [], []
iwind_start_year = 2003
for filename in iw_file_list:
    #print (str(20) + filename[28:34])
    date = int(str(20) + filename[28:34])
    if date < (iwind_start_year+1)*10000+101:
	    iw_date_list.append(date)
	    iw_u, iw_v, iw_speed = read_wind(filename)
	    iw_u, iw_v, iw_speed = iw_u.T.tolist(), iw_v.T.tolist(), iw_speed.T.tolist()
	    iw_u_list.append(iw_u)
	    iw_v_list.append(iw_v)
	    iw_speed_list.append(iw_speed)
    else:
	    iw_date_array = np.array(iw_date_list).reshape(len(iw_date_list),1)
	    print ("ice wind - " + str(iwind_start_year))
	    print ("\t Writing iw_u to csv...")
	    iw_u_csv_name = "./ts_iw_u/ts_iw_u_" + str(iwind_start_year) + ".csv"
	    write2csv(iw_date_array,iw_u_list,iw_u_csv_name)

	    print ("\t Writing iw_v to csv...")
	    iw_v_csv_name = "./ts_iw_v/ts_iw_v_" + str(iwind_start_year) + ".csv"
	    write2csv(iw_date_array,iw_v_list,iw_v_csv_name)

	    print ("\t Writing iw_speed to csv...")
	    iw_speed_csv_name = "./ts_iw_speed/ts_iw_speed_" + str(iwind_start_year) + ".csv"
	    write2csv(iw_date_array,iw_speed_list,iw_speed_csv_name)

	    iwind_start_year += 1
	    iw_date_list, iw_u_list, iw_v_list, iw_speed_list = [], [], [], []
	    iw_date_list.append(date)
	    iw_u, iw_v, iw_speed = read_wind(filename)
	    iw_u, iw_v, iw_speed = iw_u.T.tolist(), iw_v.T.tolist(), iw_speed.T.tolist()
	    iw_u_list.append(iw_u)
	    iw_v_list.append(iw_v)
	    iw_speed_list.append(iw_speed)
#print (np.array(w_u_list))
#print (np.array(w_u_list).shape)

#csvの書き出し
#インデックスの作成
iw_date_array = np.array(iw_date_list).reshape(len(iw_date_list),1)
print ("ice wind - " + str(iwind_start_year))
print ("\t Writing iw_u to csv...")
iw_u_csv_name = "./ts_iw_u/ts_iw_u_" + str(iwind_start_year) + ".csv"
write2csv(iw_date_array,iw_u_list,iw_u_csv_name)

print ("\t Writing iw_v to csv...")
iw_v_csv_name = "./ts_iw_v/ts_iw_v_" + str(iwind_start_year) + ".csv"
write2csv(iw_date_array,iw_v_list,iw_v_csv_name)

print ("\t Writing iw_speed to csv...")
iw_speed_csv_name = "./ts_iw_speed/ts_iw_speed_" + str(iwind_start_year) + ".csv"
write2csv(iw_date_array,iw_speed_list,iw_speed_csv_name)


print ("\n Process completed.")
