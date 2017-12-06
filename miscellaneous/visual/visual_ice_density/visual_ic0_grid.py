import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd

coverage = np.array([[45.0,35.64],[-45.0,35.64],[135.0,35.64],[-135.0,35.64]])
#############################################################################
m = Basemap(lon_0=180,boundinglat=40,
            resolution='l',projection='npstere')
fig=plt.figure(figsize=(10,10))

#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
m.drawcoastlines(color = '0.15')


lons = coverage[:,0]
lats = coverage[:,1]
#lons,latの段階で何度ずらして作成するのもあり

x,y = m(lons,lats)
"""
x: [  1315178.9078919   10568199.85683821   1315178.90789189  10568199.8568382 ]
y: [ 10568199.8568382   10568199.85683819   1315178.90789189   1315178.90789189]
"""

x = np.linspace(min(x), max(x), 900)
y = np.linspace(min(y), max(y), 900)

#x,y = np.meshgrid(x, y)

x = x[:]
y = y[:3]

m.plot(x,y,'bo', markersize=10)

plt.show()

