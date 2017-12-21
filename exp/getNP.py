from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('../data/latlon.csv', header=None)
latlon = np.array(df1, dtype='float32')
lon = latlon[:,2]
lat = latlon[:,3]

lon4 = [lon[0],lon[20125]]
lat4 = [lat[0],lat[20125]]


m = Basemap(lon_0=180,boundinglat=50,
            resolution='l',projection='npstere')
x, y = m(lat4, lon4)

print (x)
print (y)
