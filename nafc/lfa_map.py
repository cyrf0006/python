'''
/home/cyrf0006/research/lobster
Snipet code to extract lobster fishing area from shapefile
(see alsp sfa_map.py)

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

import os
import netCDF4
#from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import shapefile 
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#from bokeh.plotting import figure, save
#import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cmocean

## ---- Region parameters ---- ##
lon_0 = -50
lat_0 = 50
lonLims = [-61, -47]
latLims = [43, 53]
lonLims = [-61, -51] # LFAs truncated
latLims = [46, 53]
#proj = 'merc'
decim_scale = 4
fig_name = 'map_lfa.png'
v = np.linspace(50, 500,10)
vtext = np.linspace(100, 500,5)

## --------- Get Bathymetry -------- ####
print('Get bathy...')
dataFile = '/home/cyrf0006/data/GEBCO/GEBCO_2014_1D.nc' # Maybe find a better way to handle this file
# Load data
dataset = netCDF4.Dataset(dataFile)
x = [-179-59.75/60, 179+59.75/60] 
y = [-89-59.75/60, 89+59.75/60]
spacing = dataset.variables['spacing']
# Compute Lat/Lon
nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir
lon = np.linspace(x[0],x[-1],nx)
lat = np.linspace(y[0],y[-1],ny)
# interpolate data on regular grid (temperature grid)
# Reshape data
zz = dataset.variables['z']
Z = zz[:].reshape(ny, nx)
Z = np.flipud(Z) # <------------ important!!!
# Reduce data according to Region params
idx_lon = np.where((lon>=lonLims[0]) & (lon<=lonLims[1]))
idx_lat = np.where((lat>=latLims[0]) & (lat<=latLims[1]))
Z = Z[idx_lat[0][0]:idx_lat[0][-1]+1, idx_lon[0][0]:idx_lon[0][-1]+1]
lon = lon[idx_lon[0]]
lat = lat[idx_lat[0]]

lon = lon[::decim_scale]
lat = lat[::decim_scale]
Z = Z[::decim_scale, ::decim_scale]
print(' -> Done!')

## --------- GetShapefiles -------- ####
myshp = open('/home/cyrf0006/research/PeopleStuff/CoughlanStuff/LobsterFishingAreas.shp', 'rb')
mydbf = open('/home/cyrf0006/research/PeopleStuff/CoughlanStuff/LobsterFishingAreas.dbf', 'rb')
r = shapefile.Reader(shp=myshp, dbf=mydbf)
records = r.records()
shapes = r.shapes()

# Fill dictionary with NAFO divisions
lobster_area = {}
for idx, rec in enumerate(records):
    if rec[-1] == '':
        continue
    else:
        lobster_area[rec[-1]] = np.array(shapes[idx].points)


lfa = ['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13A', '13B', '14A', '14B', '14C']                        
polygon3 = Polygon(lobster_area['3'])
polygon4 = Polygon(lobster_area['4'])
polygon5 = Polygon(lobster_area['5'])
polygon6 = Polygon(lobster_area['6'])
polygon7 = Polygon(lobster_area['7'])
polygon8 = Polygon(lobster_area['8'])
polygon9 = Polygon(lobster_area['9'])
polygon10 = Polygon(lobster_area['10'])
polygon11 = Polygon(lobster_area['11'])
polygon12 = Polygon(lobster_area['12'])
polygon13A = Polygon(lobster_area['13A'])
polygon13B = Polygon(lobster_area['13B'])
polygon14A = Polygon(lobster_area['14A'])
polygon14B = Polygon(lobster_area['14B'])
polygon14C = Polygon(lobster_area['14C'])


# plot data using cartopy
projection=ccrs.Mercator()
extent = [lonLims[0], lonLims[1], latLims[0], latLims[1]]

fig = plt.figure(figsize=(13.3, 10))                      
ax = fig.add_subplot(111, projection=projection)

lon_labels = np.arange(-60, -40, 1)
lat_labels = np.arange(40, 60, 1)

gl = ax.gridlines(draw_labels=True, xlocs=lon_labels, ylocs=lat_labels)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

ax.set_extent(extent, crs=ccrs.PlateCarree())

coastline_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                        edgecolor='k', alpha=1,
                                        facecolor=cfeature.COLORS['land'], zorder=100)
ax.add_feature(coastline_10m)
c = plt.contourf(lon, lat, -Z, v, cmap=cmocean.cm.deep, extend="both",  zorder=1, transform=ccrs.PlateCarree());
#c = plt.contourf(lon, lat, -Z, v, cmap=cmocean.cm.deep, zorder=1, transform=ccrs.PlateCarree());
cc = plt.contour(lon, lat, -Z, v, colors='lightgrey', zorder=10, linewidths=.5, transform=ccrs.PlateCarree());
#ccc = plt.contour(lon, lat, -Z, vtext, colors='lightgrey', zorder=10, linewidths=.5, transform=ccrs.PlateCarree());
#plt.clabel(ccc, inline=True, fontsize=10, fmt='%i')

for i in lfa:
    polygon = Polygon(lobster_area[i])
    coords = np.array(polygon.boundary.coords.xy)
    plt.plot(coords[0,:], coords[1,:],
         color='black', linestyle='-',
         transform=ccrs.PlateCarree(), zorder=20
         )

plt.text(-52, 50.5, '3', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-52, 49.5, '4', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-52, 48.9, '5', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-52, 48.4, '6', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-52, 47.9, '7', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-52, 47.0, '8', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-54, 46.5, '9', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-55, 46.5, '10', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-57, 46.5, '11', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-59, 47.25, '12', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-59.8, 48.1, '13A', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-59.5, 49.0, '13B', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-58.5, 50, '14A', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-57.5, 51, '14B', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-56.5, 51.75, '14C', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-55, 52.3, 'North East Coast', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-52.5, 46.3, 'Avalon', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-58.5, 47.8, 'South Coast', fontsize=22, transform=ccrs.Geodetic(), zorder=200)
plt.text(-59.75, 50, 'West Coast', rotation=45, fontsize=22, transform=ccrs.Geodetic(), zorder=200)



# Add data pts
df_W = pd.read_pickle('sst_west_coast_method2.pkl')
df_S = pd.read_pickle('sst_south_coast_method2.pkl')
df_NE = pd.read_pickle('sst_northeast_coast_method2.pkl')
df_A = pd.read_pickle('sst_avalon_method2.pkl')
# Reset index (set time)
df_W.reset_index(inplace=True)
df_S.reset_index(inplace=True)
df_NE.reset_index(inplace=True)
df_A.reset_index(inplace=True)
df_W.set_index('time', inplace=True) 
df_S.set_index('time', inplace=True) 
df_NE.set_index('time', inplace=True)
df_A.set_index('time', inplace=True)
# Daily averages
df_W = df_W[df_W.index=='2019-10-10']
df_S = df_S[df_S.index=='2019-10-10']
df_NE = df_NE[df_NE.index=='2019-10-10']
df_A = df_A[df_A.index=='2019-10-10']
plt.scatter(df_W.lon.values, df_W.lat.values,
         color='gray', alpha=.7,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(df_S.lon.values, df_S.lat.values,
         color='green', alpha=.7,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(df_NE.lon.values, df_NE.lat.values,
         color='red', alpha=.7,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(df_A.lon.values, df_A.lat.values,
         color='orange', alpha=.7,
         transform=ccrs.PlateCarree(), zorder=20
         )



# Save figure
fig.set_size_inches(w=18, h=12)
fig.savefig(fig_name, dpi=200)
print('Figure trimmed!')
os.system('convert -trim ' + fig_name + ' ' + fig_name)


