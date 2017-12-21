import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from mpl_toolkits.basemap import Basemap
import pandas as pd

df0 = pd.read_csv('data/ecm030101.csv', header=None)
wind = np.array(df0, dtype='float32')

df1 = pd.read_csv('data/latlon.csv', header=None)
latlon = np.array(df1, dtype='float32')
"""
df2 = np.loadtxt('latlon.csv', delimiter=',')
lons2 = np.reshape(df2[:,2], (145,145), order = 'F')
print (type(lons2[1,1]))
lats2 = np.reshape(df2[:,3], (145,145), order = 'F')
"""

map = Basemap(projection='npstere',boundinglat=60,lon_0=90, resolution='i')
"""
map = Basemap(llcrnrlon=-150.7, llcrnrlat=28., urcrnrlon=150.1, urcrnrlat=39.5,
              projection='lcc', lat_1=30., lat_2=60., lat_0=34.83158, lon_0=-98.)
"""

"""
u10 = np.reshape(wind[:,0], (145,145), order = 'F')
v10 = np.reshape(wind[:,1], (145,145), order = 'F')
speed = np.sqrt(u10*u10 + v10*v10)

lons = np.reshape(latlon[:,2], (145,145), order = 'F')
lats = np.reshape(latlon[:,3], (145,145), order = 'F')
"""

u10 = wind[:,0]
v10 = wind[:,1]
speed = np.sqrt(u10*u10 + v10*v10)
lons = latlon[:,2]
lats = latlon[:,3]

print ("\n\n\n")

print (lons)
print (type(lons))
print (lons.shape)
print (lons.dtype)
print (lons[1])
print (type(lons[1]))
print ("\n\n\n")
"""
print (lats)
print (type(lats))
print (lats.shape)
print (lats.dtype)
print ("\n\n\n")
"""

x, y = map(lons, lats)

#g = np.arange(0, 145, 1)
#print (x.shape)
g = np.arange(0, 145*145, 1)
points = np.meshgrid(g)



print ("x: ")
print (x)
print (x.shape)
print ("\n\n\n")

print ("y: ")
print (y)
print (y.shape)
print ("\n\n\n")

print ("g:")
print (g)
print (len(g))
print ("\n\n\n")

print ("points")
print (points)
print (points[0].shape)
print ("\n\n\n")






map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
map.drawcoastlines(color = '0.15')

map.quiver(x[points], y[points], 
    u10[points], v10[points], speed[points],
    cmap=plt.cm.autumn)

plt.show()

