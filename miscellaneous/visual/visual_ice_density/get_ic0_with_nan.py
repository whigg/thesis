"""
IC0の全データをcsvに書き出す(欠損値はnanにしている)
書き出した結果は(lon,lat,ic0)になっている
csvのlon, latはポーラーステレオ座標なので注意
可視化の際にreshapeするときはorder='F'では「ない」ので，注意
"""

from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

#緯度経度情報の作成
coverage = np.array([[45.0,35.64], [-45.0,35.64], [135.0,35.64], [-135.0,35.64]])
lons = coverage[:,0]
lats = coverage[:,1]
m = Basemap(lon_0=180, boundinglat=40, resolution='l', projection='npstere')
x,y = m(lons, lats)
x = np.linspace(min(x), max(x), 900)[::-1]
y = np.linspace(min(y), max(y), 900)
xx,yy = np.meshgrid(x, y)
grids = np.vstack([xx.ravel(), yy.ravel()]).T[-1::-1]
x_ = grids[:,0]
y_ = grids[:,1]

np.savetxt("./IC0_csv/latlon_info.csv", np.c_[x_, y_], delimiter=',')

#IC0の書き出し
datapath0 = "../../../data/"
file_list = sorted(glob.glob(datapath0 + 'IC0/*.dat'))
for filename in tqdm(file_list):
	fp = open(filename,'rb')
	ary = np.fromfile(fp, '<h', -1)
	fp.close()

	ic0 = np.zeros(900**2)
	ic0[ary<0] = np.nan
	ic0[ary>=0] = ary[ary>=0]

	savename = "./IC0_csv/" + filename[23:33] + '.csv'
	#data = np.c_[x_, y_, ic0]
	np.savetxt(savename, ic0, delimiter=',')


