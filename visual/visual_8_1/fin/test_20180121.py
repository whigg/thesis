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
import matplotlib.gridspec as gridspec
import random
import itertools
import matplotlib.dates as mdates



y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

for year in y_list:
	for month in month_list:
		print("******************** {} ***************".format(year + month))
		gw_file_list = sorted(glob.glob("../data/csv_w/ecm" + year + month + "*.csv"))
		del gw_file_list[0]
		if month != "12":
			next_year = year
		else:
			next_year = str(int(year)+1)
			if len(next_year) == 1:
				next_year = "0" + next_year
		gw_file_list.append("../data/csv_w/ecm" + next_year + month_list[(int(month)+1)%12-1] + "01.csv")
		iw_file_list = sorted(glob.glob("../data/csv_iw/" + year + month + "*.csv"))

		for i in range(len(gw_file_list)):
			print("{}, {}".format(gw_file_list[i][17:], iw_file_list[i][15:]))


print("\n\n\n")



"""
y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
m_list = [["01", "02", "03"], ["04", "05", "06"], ["07", "08", "09"], ["10", "11", "12"]]
for year in y_list:
	for im, m3 in enumerate(m_list):
		print("******************** {}, {} ***************".format(year, m3))
		gw_file_list = sorted(glob.glob("../data/csv_w/ecm" + year + m3[0] + "*.csv") + \
			glob.glob("../data/csv_w/ecm" + year + m3[1] + "*.csv") + \
			glob.glob("../data/csv_w/ecm" + year + m3[2] + "*.csv"))
		del gw_file_list[0]
		if m3[2] != "12":
			next_year = year
			next_month = str(int(m3[2])+1)
			if len(next_month) == 1:
				next_month = "0" + next_month
		else:
			next_year = str(int(year)+1)
			if len(next_year) == 1:
				next_year = "0" + next_year
			next_month = "01"
		gw_file_list.append("../data/csv_w/ecm" + next_year + next_month + "01.csv")
		iw_file_list = sorted(glob.glob("../data/csv_iw/" + year + m3[0] + "*.csv") + \
			glob.glob("../data/csv_iw/" + year + m3[1] + "*.csv") + \
			glob.glob("../data/csv_iw/" + year + m3[2] + "*.csv"))

		for i in range(len(gw_file_list)):
			print("{}, {}".format(gw_file_list[i][17:], iw_file_list[i][15:]))
"""

#print("\n\n\n")


y_list = ["03", "04", "05", "06", "07", "08", "09", "10", "13", "14", "15", "16"]
month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
for im, month in enumerate(month_list):
	print("******************** {} ***************".format(month))
	gw_file_list = []
	iw_file_list = []
	for iy, year in enumerate(y_list):
		gw_tmp = glob.glob("../data/csv_w/ecm" + year + month + "*.csv")
		del gw_tmp[0]
		if month == "12":
			next_month = "01"
			next_year = str(int(year)+1)
			if len(next_year) == 1:
				next_year = "0" + next_year
		else:
			next_month = month_list[im+1]
			next_year = year
		tmp = "../data/csv_w/ecm" + next_year + next_month + "01.csv"
		gw_tmp = gw_tmp + [tmp]
		gw_file_list = gw_file_list + gw_tmp
		#print(gw_file_list)
		iw_file_list = iw_file_list + glob.glob("../data/csv_iw/" + year + month + "*.csv")
	#gw_file_list = sorted(gw_file_list)
	iw_file_list = sorted(iw_file_list)

	for i in range(len(gw_file_list)):
		print("{}, {}".format(gw_file_list[i][17:], iw_file_list[i][15:]))





