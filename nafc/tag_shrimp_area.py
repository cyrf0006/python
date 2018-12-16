'''
/home/cyrf0006/research/tag_lebris
Snipet code to extract shrimp fishing area from shapefile

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
import shapefile 
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#from bokeh.plotting import figure, save
#import geopandas as gpd


## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-70, -40]
latLims = [40, 65]
proj = 'merc'
decim_scale = 4
fig_name = 'Shrimp_area.png'

## ---- Bathymetry ---- ####
v = np.linspace(-4000, 0, 9)

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



#myshp = open('/home/cyrf0006/research/tag_lebris/SFA4to6new.shp', 'rb')
#mydbf = open('/home/cyrf0006/research/tag_lebris/SFA4to6new.dbf', 'rb')
myshp = open('/home/cyrf0006/research/tag_lebris/SFA4to7_polygons.shp', 'rb')
mydbf = open('/home/cyrf0006/research/tag_lebris/SFA4to7_polygons.dbf', 'rb')
r = shapefile.Reader(shp=myshp, dbf=mydbf)
records = r.records()
shapes = r.shapes()

# Fill dictionary with NAFO divisions
shrimp_area = {}
for idx, rec in enumerate(records):
    if rec[-1] == '':
        continue
    else:
        shrimp_area[rec[-1]] = np.array(shapes[idx].points)
        
polygon4 = Polygon(shrimp_area[4])
polygon5 = Polygon(shrimp_area[5])
polygon6 = Polygon(shrimp_area[6])
polygon7 = Polygon(shrimp_area[7])


fig = plt.figure(1)
#m = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution=None)
m = Basemap(projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='l')
x,y = m(*np.meshgrid(lon,lat))
c = m.contourf(x, y, np.flipud(Z), v, cmap=plt.cm.PuBu_r, extend="min");
#c = m.contourf(x, y, np.flipud(Z), v, cmap=plt.cm.PuRd_r, extend="min");
m.fillcontinents(color='grey');
m.drawparallels(np.arange(10,70,10), labels=[1,0,0,0], fontsize=12, fontweight='bold');
m.drawmeridians(np.arange(-80, 5, 10), labels=[0,0,0,1], fontsize=12, fontweight='bold');
cb = plt.colorbar(c)
for l in cb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(10)
cb.set_label('Depth (m)', fontsize=13, fontweight='bold')
#plt.title("AZMP-NL standard lines", fontsize=13, fontweight='bold')

coords = np.array(polygon4.boundary.coords.xy)
x,y = m(coords[0,:], coords[1,:])
m.plot(x,y,color='black')
plt.text(np.mean(x), np.mean(y), 'SFA-4', fontsize=10, color='black', fontweight='bold')

coords = np.array(polygon5.boundary.coords.xy)
x,y = m(coords[0,:], coords[1,:])
m.plot(x,y,color='black')
plt.text(np.mean(x), np.mean(y), 'SFA-5', fontsize=10, color='black', fontweight='bold')

coords = np.array(polygon6.boundary.coords.xy)
x,y = m(coords[0,:], coords[1,:])
m.plot(x,y,color='black')
plt.text(np.mean(x), np.mean(y), 'SFA-6', fontsize=10, color='black', fontweight='bold')

coords = np.array(polygon7.boundary.coords.xy)
x,y = m(coords[0,:], coords[1,:])
m.plot(x,y,color='black')
plt.text(np.mean(x), np.mean(y), 'SFA-7', fontsize=10, color='black', fontweight='bold')


#### ---- Save Figure ---- ####
fig.set_size_inches(w=8, h=9)
fig.set_dpi(200)
fig.savefig(fig_name)
#plt.show()

