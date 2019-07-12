'''
From https://pypi.python.org/pypi/pyshp:

A record in a shapefile contains the attributes for each shape in the
collection of geometry. Records are stored in the dbf file. The link between
geometry and attributes is the foundation of all geographic information systems.
This critical link is implied by the order of shapes and corresponding records
in the shp geometry file and the dbf attribute file.

The field names of a shapefile are available as soon as you read a shapefile.
You can call the "fields" attribute of the shapefile as a Python list. Each
field is a Python list with the following information:

* Field name: the name describing the data at this column index.
* Field type: the type of data at this column index. Types can be: Character,
Numbers, Longs, Dates, or Memo. The "Memo" type has no meaning within a
GIS and is part of the xbase spec instead.
* Field length: the length of the data found at this column index. Older GIS
software may truncate this length to 8 or 11 characters for "Character"
fields.
* Decimal length: the number of decimal places found in "Number" fields.

To see the fields for the Reader object above (sf) call the "fields"
attribute:
'''

## plt.show()


import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import pandas as pd

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-60, -44]
latLims = [39, 56]
proj = 'merc'
decim_scale = 4



fig_name = 'map_bathym_nafo.png'

## ---- Bathymetry ---- ####
#v = np.linspace(0, 1400, 8)
v = [100, 200, 400, 1000]

# Load data
dataset = netCDF4.Dataset(dataFile)

# Extract variables
x = dataset.variables['x_range']
y = dataset.variables['y_range']
spacing = dataset.variables['spacing']

# Compute Lat/Lon
nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir

lon = np.linspace(x[0],x[-1],nx)
lat = np.linspace(y[0],y[-1],ny)

# Reshape data
zz = dataset.variables['z']
Z = zz[:].reshape(ny, nx)

# Reduce data according to Region params
lon = lon[::decim_scale]
lat = lat[::decim_scale]
Z = Z[::decim_scale, ::decim_scale]


## ---- NAFO divisions ---- ##
# (see map_david_paper.py)
xlon = [-59.778, -47.529, -42, -55.712]
xlat = [55.33, 55.33, 52.25, 52.25]
div2J = {'lat' : xlat, 'lon' : xlon}
xlon = [-55.425, -55.425, -42, -42, -53.466]
xlat = [51.583, 52.25, 52.25, 49.25, 49.25]
div3K = {'lat' : xlat, 'lon' : xlon}
xlon = [-53.466, -46.5, -46.5, -54.5, -54.198]
xlat = [49.25, 49.25, 46, 46, 46.815]
div3L = {'lat' : xlat, 'lon' : xlon}
xlon = [-51, -46.5, -46.5, -50, -51, -51]
xlat = [46, 46, 39, 39, 39.927, 46]
div3N = {'lat' : xlat, 'lon' : xlon}
xlon = [-54.5, -51, -51, -54.4, -54.5]
xlat = [46, 46, 39.927, 43.064, 46]
div3O = {'lat' : xlat, 'lon' : xlon}
xlon = [-57.523, -58.82, -54.5, -54.5, -54.2]
xlat = [47.631, 46.843, 43.064, 46, 46.817]
div3Ps = {'lat' : xlat, 'lon' : xlon}


## ---- plot ---- ##
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='h')
x,y = m(*np.meshgrid(lon,lat))
c = m.contourf(x, y, np.flipud(-Z), v, cmap=plt.cm.PuBu_r, extend="min");
cc = m.contour(x, y, np.flipud(-Z), v, colors='grey');
plt.clabel(cc, inline=1, fontsize=10, colors='k', fmt='%d')
m.fillcontinents(color='tan');
m.drawparallels([40, 45, 50, 55], labels=[1,0,0,0], fontsize=12, fontweight='normal');
m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');

        
# Draw NAFO divisions
x,y = m(np.array(div3K['lon']), np.array(div3K['lat']))
m.plot(x,y,color='black')
ax.text(np.mean(x), np.mean(y), '3K', fontsize=12, color='black', fontweight='bold')

x,y = m(np.array(div3L['lon']), np.array(div3L['lat']))
m.plot(x,y,color='black')
ax.text(np.mean(x), np.mean(y), '3L', fontsize=12, color='black', fontweight='bold')

x,y = m(np.array(div3N['lon']), np.array(div3N['lat']))
m.plot(x,y,color='black')
ax.text(np.mean(x), np.mean(y), '3N', fontsize=12, color='black', fontweight='bold')

x,y = m(np.array(div3O['lon']), np.array(div3O['lat']))
m.plot(x,y,color='black')
ax.text(np.mean(x)*.9, np.mean(y), '3O', fontsize=12, color='black', fontweight='bold')

x,y = m(np.array(div3Ps['lon']), np.array(div3Ps['lat']))
m.plot(x,y,color='black')
ax.text(np.mean(x)*.7, np.mean(y)*.95, '3Ps', fontsize=12, color='black', fontweight='bold')

x,y = m(np.array(div2J['lon']), np.array(div2J['lat']))
m.plot(x,y,color='black')
ax.text(np.mean(x), np.mean(y), '2J', fontsize=12, color='black', fontweight='bold')


#### ---- Save Figure ---- ####
#plt.suptitle('Fall surveys', fontsize=16)
fig.set_size_inches(w=9, h=12)
#fig.tight_layout() 
fig.set_dpi(200)
fig.savefig(fig_name)


