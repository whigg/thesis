import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd

###########################################################################################
"""
windのバイナリ読み込み
pcolormeshの修正
"""
fname = "ecm090623.ads60"
dt = np.dtype([("u","<f"), ("v","<f"), ("slp","<f")])
fd = open(fname,"r")
result = np.fromfile(fd, dtype=dt, count=-1)
data = result.tolist()
data1 = np.array([list(data[i]) for i in range(len(data))])

w_u = data1[:,0]
w_v = data1[:,1]
w_speed = np.sqrt(w_u**2+w_v**2)
slp = data1[:,2]
slp1 = np.reshape(slp, (145,145), order="F")


m = Basemap(lon_0=180, boundinglat=50, resolution='i', projection='npstere')
fig = plt.figure(figsize=(6.5, 6.5))

df_latlon = pd.read_csv("latlon_ex.csv")
lon = df_latlon.Lon
lat = df_latlon.Lat
lon = np.array(lon)
lat = np.array(lat)
x, y = m(lon, lat)

x1 = np.reshape(x, (145,145), order="F")
y1 = np.reshape(y, (145,145), order="F")


dx = (x1[0,1]-x1[0,0])/2
dy = (y1[1,0]-y1[0,0])/2
dx1 = (x1[1,0]-x1[0,0])/2
dy1 = (y1[0,1]-y1[0,0])/2



x2 = np.linspace(x1[0,0], x1[144,0], 145)
y2 = np.linspace(y1[0,0], y1[0,144], 145)
xx, yy = np.meshgrid(x2, y2)
xx, yy = xx.T, yy.T

dx2 = (xx[1,0]-xx[0,0])/2
dy2 = (yy[0,1]-yy[0,0])/2

m.drawcoastlines(color = '0.15')
m.plot(xx[144,0], yy[144,0], "bo")

xx = np.hstack([xx, xx[:,0].reshape(145,1)])
xx_ex = np.vstack([xx, (xx[144,:] + (xx[1,0]-xx[0,0]))])
yy = np.vstack([yy, yy[0,:]])
yy_ex = np.hstack([(yy[:,0].reshape(146,1) + (yy[0,0]-yy[0,1])), yy])



m.drawcoastlines(color = '0.15')
m.pcolormesh(xx_ex-dx1, yy_ex+dy1, slp1, cmap=plt.cm.jet)
m.quiver(x, y, w_u, w_v, w_speed)
plt.show()


"""
m.drawcoastlines(color = '0.15')
m.pcolormesh(x1-dx1, y1+dy1, slp1, cmap=plt.cm.jet)
m.quiver(x, y, w_u, w_v, w_speed)
plt.show()
"""


###########################################################################################
#colormapの修正

cm_angle = generate_cmap([
    [0, 0, 96/255], 
    [0/255, 0/255, 255/255], 
    [108/255, 108/255, 108/255], 
    [255/255, 0/255, 0/255], 
    [96/255, 0, 0]
    ])





###########################################################################################
# 可視化の保存先の部分の修正

import os

try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))




###########################################################################################
#流氷データの読み込み
import csv, glob, os, tarfile
from tqdm import tqdm

def readfile(fname):
	dt = np.dtype([("u","<f"), ("v","<f")])
	fd = open(fname,"r")
	result = np.fromfile(fd, dtype=dt, count=-1)
	return result

#read each file in 'input_binary' and save the csv file to './output_csv'
files_y = [os.path.basename(r) for r in glob.glob('input_binary/*')]

for y in tqdm(files_y):
	files = [os.path.basename(r) for r in glob.glob('./input_binary/' + y + '/*')]
	for f in files:
		fname = './input_binary/' + y + '/' + f
		result = readfile(fname)
		sname = './output_csv/' + f[0:6] + '.csv'
		np.savetxt(sname, result, delimiter=',')







