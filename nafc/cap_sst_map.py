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
import xarray as xr
import h5py
import azmp_utils as azu    

## ---- Region parameters ---- ##
lon_0 = -50
lat_0 = 50
dataFile = '/home/cyrf0006/data/GEBCO/GEBCO_2014_1D.nc'
lonLims = [-62, -45] # 2J3KLNO
latLims = [45, 57]

decim_scale = 4
fig_name = 'map_sst_capelin.png'
v = np.linspace(100, 5000, 10)
vtext = np.linspace(1000, 5000, 5)

## ---- Load NAFO divisions ---- ##
nafo_div = azu.get_nafo_divisions()
# Polygons
polygon2J = Polygon(zip(nafo_div['2J']['lon'], nafo_div['2J']['lat']))
polygon3K = Polygon(zip(nafo_div['3Kx']['lon'], nafo_div['3Kx']['lat']))
polygon3L = Polygon(zip(nafo_div['3L']['lon'], nafo_div['3L']['lat']))
polygon3N = Polygon(zip(nafo_div['3N']['lon'], nafo_div['3N']['lat']))
polygon3O = Polygon(zip(nafo_div['3O']['lon'], nafo_div['3O']['lat']))
polygon3Ps = Polygon(zip(nafo_div['3Ps']['lon'], nafo_div['3Ps']['lat']))

## ---- get NL shelf ---- ##
NLshelf = azu.get_NLshelf('/home/cyrf0006/github/AZMP-NL/data/NLshelf_definition.npy')
polygonShelf = Polygon(NLshelf)


## --------- Get Bathymetry -------- ####

print('Load and grid bathymetry')
# h5 file
h5_outputfile = 'capelin_bathymetry2.h5'
if os.path.isfile(h5_outputfile):
     print([h5_outputfile + ' exists! Reading directly'])
     h5f = h5py.File(h5_outputfile,'r')
     lon = h5f['lon'][:]
     lat = h5f['lat'][:]
     Z = h5f['Z'][:]
     h5f.close()

else:
    # Extract variables
    dataset = netCDF4.Dataset(dataFile)
    x = [-179-59.75/60, 179+59.75/60] # to correct bug in 30'' dataset?
    y = [-89-59.75/60, 89+59.75/60]
    spacing = dataset.variables['spacing']

    # Compute Lat/Lon
    nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
    ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir
    lon = np.linspace(x[0],x[-1],nx)
    lat = np.linspace(y[0],y[-1],ny)
    ## interpolate data on regular grid (temperature grid)
    ## Reshape data
    zz = dataset.variables['z']
    Z = zz[:].reshape(ny, nx)
    Z = np.flipud(Z) # <------------ important!!!
    # Reduce data according to Region params
    idx_lon = np.where((lon>=lonLims[0]) & (lon<=lonLims[1]))
    idx_lat = np.where((lat>=latLims[0]) & (lat<=latLims[1]))
    Z = Z[idx_lat[0][0]:idx_lat[0][-1]+1, idx_lon[0][0]:idx_lon[0][-1]+1]
    lon = lon[idx_lon[0]]
    lat = lat[idx_lat[0]]

    # Save data for later use
    h5f = h5py.File(h5_outputfile, 'w')
    h5f.create_dataset('lon', data=lon)
    h5f.create_dataset('lat', data=lat)
    h5f.create_dataset('Z', data=Z)
    h5f.close()
    print(' -> Done!')

lon = lon[::decim_scale]
lat = lat[::decim_scale]
Z = Z[::decim_scale, ::decim_scale]
print(' -> Done!')


# plot data using cartopy
projection=ccrs.Mercator()
extent = [lonLims[0], lonLims[1], latLims[0], latLims[1]]

fig = plt.figure(figsize=(13.3, 10))                      
ax = fig.add_subplot(111, projection=projection)

lon_labels = np.arange(0, 25, 5)
lat_labels = np.arange(55, 75, 5)

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
cc = plt.contour(lon, lat, -Z, v, colors='lightgrey', zorder=10, linewidths=.5, transform=ccrs.PlateCarree());

coords2J = np.array(polygon2J.boundary.coords.xy)
coords3K = np.array(polygon3K.boundary.coords.xy)
coords3L = np.array(polygon3L.boundary.coords.xy)
coordsShelf = np.array(polygonShelf.boundary.coords.xy)

plt.plot(coords2J[0,:], coords2J[1,:],
         color='black', linestyle='-',
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot(coords3K[0,:], coords3K[1,:],
         color='black', linestyle='-',
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot(coords3L[0,:], coords3L[1,:],
         color='black', linestyle='-',
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot(coordsShelf[0,:], coordsShelf[1,:],
         color='black', linestyle='-',
         transform=ccrs.PlateCarree(), zorder=20
         )
# Add data pts
df_2J = pd.read_pickle('sst_2J_method2.pkl')
df_3K = pd.read_pickle('sst_3K_method2.pkl')
df_3L = pd.read_pickle('sst_3L_method2.pkl')
# Reset index (set time)
df_2J.reset_index(inplace=True)
df_3K.reset_index(inplace=True)
df_3L.reset_index(inplace=True)
df_2J.set_index('time', inplace=True) 
df_3K.set_index('time', inplace=True) 
df_3L.set_index('time', inplace=True) 
# Daily averages
df_2J = df_2J[df_2J.index=='2019-10-10']
df_3K = df_3K[df_3K.index=='2019-10-10']
df_3L = df_3L[df_3L.index=='2019-10-10']
plt.scatter(df_2J.lon.values, df_2J.lat.values,
         color='orange', alpha=.7,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(df_3K.lon.values, df_3K.lat.values,
         color='green', alpha=.7,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(df_3L.lon.values, df_3L.lat.values,
         color='gray', alpha=.7,
         transform=ccrs.PlateCarree(), zorder=20
         )
# Save figure
fig.set_size_inches(w=18, h=12)
fig.savefig(fig_name, dpi=200)
print('Figure trimmed!')
os.system('convert -trim ' + fig_name + ' ' + fig_name)


