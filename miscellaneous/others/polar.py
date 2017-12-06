from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# setup north polar stereographic basemap.
# The longitude lon_0 is at 6-o'clock, and the
# latitude circle boundinglat is tangent to the edge
# of the map at lon_0. Default value of lat_ts
# (latitude of true scale) is pole.

df0 = pd.read_csv('ecm030101.csv', header=None)
wind = np.array(df0, dtype='float32')

df1 = pd.read_csv('latlon.csv', header=None)
latlon = np.array(df1, dtype='float32')

s_lon = latlon[:,3][0:3000]
s_lat = latlon[:,2][0:3000]
#print (s_lon)
#print (s_lat)

m = Basemap(projection='npstere',boundinglat=50,lon_0=180,resolution='l')
"""
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary(fill_color='aqua')
"""

# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,10.))
m.drawmeridians(np.arange(-180.,181.,20.))

m.bluemarble()


# draw tissot's indicatrix to show distortion.
#ax = plt.gca()

"""
print (m.ymax)
print (m.xmax)
for y in np.linspace(m.ymax/20,19*m.ymax/20,10):
    for x in np.linspace(m.xmax/20,19*m.xmax/20,10):
        lon, lat = m(x,y,inverse=True)
        print (lon, lat)
        poly = m.tissot(lon,lat,0.3,100,\
                        facecolor='green',zorder=5,alpha=0.5)
"""

#print (latlon[:,3][0],latlon[:,2][0])
#lons, lats = m(-135,86,inverse=False)
lons, lats = m(s_lon,s_lat,inverse=False)

#print (lon)
m.plot(lons,lats,'bo', markersize=0.3)


"""
for y in s_lat:
    for x in s_lon:
        #lon, lat = m(x,y)
        #print (lon,lat)
        poly = m.tissot(x,y,0.7,100,\
                        facecolor='green',zorder=5,alpha=0.5)
"""

plt.title("North Polar Stereographic Projection")
plt.show()