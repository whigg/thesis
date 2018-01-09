#https://stackoverflow.com/questions/32853787/matplotlib-basemap-coastal-coordinates

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np


def distance_from_coast(lon,lat,resolution='l',degree_in_km=111.12):
    plt.ioff()

    m = Basemap(projection='robin',lon_0=0,resolution=resolution)
    coast = m.drawcoastlines()

    coordinates = np.vstack(coast.get_segments())
    lons,lats = m(coordinates[:,0],coordinates[:,1],inverse=True)

    dists = np.sqrt((lons-lon)**2+(lats-lat)**2)

    if np.min(dists)*degree_in_km<1:
      return True
    else:
      return False

print(distance_from_coast(-117.2547,32.8049,resolution="i"))