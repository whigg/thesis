"""
[概要]
指定した日付からファイルを取り出し、時系列プロット

[ルール・注意]
・get_...関数は、その都度自分の求めたいデータに応じて書き換える
・visualize関数も、どのようなプロットをしたいかで、書き換える

[使い方]
・mainでread_ts_file関数を呼び出し、DataFrame型のdataを取得する
・そのdataをvisualize関数の引数にして、描画する

[例]
・特定の季節だけ取り出したい場合，1年以上のデータを見たい場合も，日付を指定すればいい
・違うタイプのデータはfile_type_listで指定
・同じタイプだが複数の時期を取得したい場合は，read_ts_fileをmainから複数回呼び出す
・

[TODO]
・日付の指定とデータの取得インデックスの範囲をmainから指定したい
	・ある日付から何日分という指定
	・日付が飛び飛びになっている場合(3/1, 3/5, 3/10など)
	・複数のグリッドを選択したい場合
		全部または大多数を呼び出すとき -> 平均や分散やカウントがわかればいい
		その他(例えば5つ選択など) -> get_...関数の中のindexをリストにしてfor文？

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime, date, timezone, timedelta
import method_sns_graph as m_sns

"""
#日付設定のテンプレ
d1, d2 = datetime(2003, 1, 1), datetime(2003,6, 30)
L = (d2-d1).days+1
dt = d1
days_all = []
for i in range(L):
	days_all.append(int(str(dt)[:10].replace('-', '')))
	dt = dt + timedelta(days=1)
print (days_all)
"""

def cvt_date(dt):
	# "2013-01-01" -> "20130101"
	return str(dt)[:10].replace('-', '')


def read_ts_file(file_type_list, start, end, date_col=True):
	"""
	start = 20130101
	end = 20130630
	file_type: ['wind', 'ice', 'ic0_145', 'ic0_900']
	"""
	print ("file types:")
	print ('\t{}'.format(file_type_list))

	start_date = [start//10000, (start%10000)//100, (start%10000)%100]
	end_date = [end//10000, (end%10000)//100, (end%10000)%100]
	d1 = datetime(start_date[0], start_date[1], start_date[2])
	d2 = datetime(end_date[0], end_date[1], end_date[2])
	L = (d2-d1).days+1
	dt = d1

	date_ax = []
	date_ax_str = []
	for i in range(L):
		date_ax.append(dt)
		date_ax_str.append(cvt_date(dt))
		dt = dt + timedelta(days=1)

	#data = pd.to_datetime(date_ax)
	data = pd.DataFrame(date_ax)

	#このdataに，以下のvalue_listを列結合していく
	#ex. file_type_list = ["wind", "ic0_145"]
	for file_type in file_type_list:
		if file_type == "wind":
			value_list = get_w_data(date_ax_str)

			#ここに列結合のコード
			value_column = pd.DataFrame(value_list)
			data = pd.concat([data, value_column], axis=1)

		elif file_type == "ice":
			value_list = get_iw_data(date_ax_str)

			value_column = pd.DataFrame(value_list)
			data = pd.concat([data, value_column], axis=1)

		elif file_type == "ic0_145":
			value_list = get_ic0_145_data(date_ax_str)

			value_column = pd.DataFrame(value_list)
			data = pd.concat([data, value_column], axis=1)

		elif file_type == "ic0_900":
			value_list = get_ic0_900_data(date_ax_str)

			value_column = pd.DataFrame(value_list)
			data = pd.concat([data, value_column], axis=1)

		else:
			print ("Error: No file type matched.")
	
	column_name = ["date"] + file_type_list
	data.columns = column_name
	if date_col==False:
		data = data.drop("date", axis=1)

	#いったんdataをmainに投げて、そこからvisualize関数を呼び出す
	return data


############################################################################
#データを読み込んで、知りたい値のリストを返す関数群

def get_w_data(date_ax_str):
	print ("Loading wind data ...")
	value_list = []
	for day in tqdm(date_ax_str):
		wind_file_name = "../data/wind_data/ecm" + day[2:] + ".csv"
		w_u, w_v, w_speed = m_sns.read_wind(wind_file_name)
		#ここに，取り出したいデータを作る(あるグリッドなのか，平均なのか，など)
		#あるグリッドの時
		index = 8455
		value = w_speed[index]
		value_list.append(value)

	return value_list


def get_iw_data(date_ax_str):
	print ("Loading ice wind data ...")
	value_list = []
	for day in tqdm(date_ax_str):
		ice_file_name = "../data/ice_wind_data/" + day[2:] + ".csv"
		iw_u, iw_v, iw_speed, idx_t = m_sns.read_ice_v(ice_file_name)
		#ここに，取り出したいデータを作る(あるグリッドなのか，平均なのか，など)
		#あるグリッドの時
		index = 8455
		if index in idx_t:
			value = iw_speed[index]
		else:
			value = np.nan
		value_list.append(value)

	return value_list


def get_ic0_145_data(date_ax_str):
	print ("Loading IC0_145 data ...")
	value_list = []
	for day in tqdm(date_ax_str):
		ic0_145_file_name = "../data/IC0_csv/2" + day + "A.csv"
		ic0_145, idx_t = m_sns.read_ic0(ic0_145_file_name, grid900to145="../data/grid900to145.csv")
		#ここに，取り出したいデータを作る(あるグリッドなのか，平均なのか，など)
		#あるグリッドの時
		index = 8455
		if index in idx_t:
			value = ic0_145[index]
		else:
			value = np.nan
		value_list.append(value)

	return value_list


def get_ic0_900_data(date_ax_str):
	print ("Loading IC0_900 data ...")
	value_list = []
	for day in tqdm(date_ax_str):
		ic0_900_file_name = "../data/IC0_csv/2" + day + "A.csv"
		df0 = pd.read_csv(ic0_900_file_name, header=None)
		ic0_900 = np.array(df0, dtype='float32')
		#ここに，取り出したいデータを作る(あるグリッドなのか，平均なのか，など)
		#あるグリッドの時
		value = ic0_900[45]
		value_list.append(value)

	return value_list

############################################################################


def visualize(data):
	"""
	DataFrame型のdataをプロットする

	[参考]
	http://sinhrks.hatenablog.com/entry/2015/11/15/222543
	"""

	#普通の時系列プロット
	
	tmp = pd.to_datetime(data["date"])
	data["date"] = data.index
	data.index = tmp
	data = data.rename(columns={'date': 'idx'})

	#data[["wind", "ice"]].plot(figsize=(16,4), alpha=0.5)
	"""
	ax = data.wind.plot(figsize=(16,4), ylim=(0, 30), color="blue" )
	ax2 = ax.twinx()
	data.ice.plot( ax=ax2, ylim=(0, 0.8), color="red" )
	"""


	plt.show()

	#時間軸が違う場合(share axis)










