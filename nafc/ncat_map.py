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
lonLims = [-58, -45]
latLims = [47, 55]
#lonLims = [-54, -48] # zoom
#latLims = [48, 54.5]
decim_scale = 1
fig_name = 'map_ncat.png'
v = np.linspace(50, 500,10)
vtext = np.linspace(100, 500,5)
## v = np.linspace(100, 1000,10)
## vtext = np.linspace(200, 1000, 5)

# Original VEMCO
#idx_shelf = np.array([0,1,3,4,6,7,9,11,12])
#idx_channel = np.array([0,5,6,10,12,16])

# New 45 TidBits
idx_shelf = np.array([0,2,4,5,7,8,10,12,13,16,18,19,22,23,24,26,28,29,32,33,36,39,40,43,46,47,48,49,50,51,52,53,54,55,56,57])
idx_channel = np.array([0,3,5,6,8,10,11,14,16])

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


## --------- Get gates -------- ####
df_shelf = pd.read_excel('/home/cyrf0006/research/NCAT/Northern Cod Survey - Buoy Locations - July 2nd 2019.xls', header=0, sheet_name='Continental Shelf Bouys')
df_channel = pd.read_excel('/home/cyrf0006/research/NCAT/Northern Cod Survey - Buoy Locations - July 2nd 2019.xls', header=0, sheet_name='Channel Arrays')


## --------- Add depths to DataFrames -------- ####
## -> For shelf
lon_shelf = df_shelf['Longitude DD'].values
lat_shelf = df_shelf['Latitude DD'].values
z = np.full_like(lon_shelf, 'nan')
couple = []
z_vec = []

# build lat/lon vector couples
for id_ln, ln in enumerate(lon):
    for id_lt, lt in enumerate(lat):
        couple.append([ln, lt])
        z_vec.append(Z[id_lt, id_ln])
couple = np.array(couple)
z_vec = np.array(z_vec)

# Minimize the position
for idx, ln in enumerate(lon_shelf):
    z_idx = np.argmin(np.abs(lon_shelf[idx]-couple[:,0]) + np.abs(lat_shelf[idx]-couple[:,1]))
    z[idx] = -z_vec[z_idx]

    #print(lon_shelf[idx], lat_shelf[idx], couple[z_idx])
    
df_shelf['depth']=z 
df_shelf.to_csv('shelf_gates.csv')  
df_shelf.iloc[idx_shelf].to_csv('thermistor_shelf_gates.csv')

## -> For channels
lon_channel = df_channel['Longitude (DD)'].values
lat_channel = df_channel['Latitude (DD)'].values
z = np.full_like(lon_channel, 'nan')
couple = []
z_vec = []

# build lat/lon vector couples
for id_ln, ln in enumerate(lon):
    for id_lt, lt in enumerate(lat):
        couple.append([ln, lt])
        z_vec.append(Z[id_lt, id_ln])
couple = np.array(couple)
z_vec = np.array(z_vec)

# Minimize the position
for idx, ln in enumerate(lon_channel):
    z_idx = np.argmin(np.abs(lon_channel[idx]-couple[:,0]) + np.abs(lat_channel[idx]-couple[:,1]))
    z[idx] = -z_vec[z_idx]

    #print(lon_channel[idx], lat_channel[idx], couple[z_idx])
    
df_channel['depth']=z 
df_channel.to_csv('channel_gates.csv')  
df_channel.iloc[idx_channel].to_csv('thermistor_channel_gates.csv')



## --------- plot data using cartopy -------- ####
projection=ccrs.Mercator()
extent = [lonLims[0], lonLims[1], latLims[0], latLims[1]]

fig = plt.figure(figsize=(13.3, 10))                      
ax = fig.add_subplot(111, projection=projection)

lon_labels = np.arange(lonLims[0]-2, lonLims[1]+2, 2)
lat_labels = np.arange(latLims[0]-2, latLims[1]+2, 2)

gl = ax.gridlines(draw_labels=True, xlocs=lon_labels, ylocs=lat_labels)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'fontsize':15}
gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'fontsize':15}

ax.set_extent(extent, crs=ccrs.PlateCarree())

coastline_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                        edgecolor='k', alpha=1,
                                        facecolor=cfeature.COLORS['land'], zorder=100)
ax.add_feature(coastline_10m)
c = plt.contourf(lon, lat, -Z, v, cmap=cmocean.cm.deep, extend="both",  zorder=1, transform=ccrs.PlateCarree());
#c = plt.contourf(lon, lat, -Z, v, cmap=cmocean.cm.deep, zorder=1, transform=ccrs.PlateCarree());
#cc = plt.contour(lon, lat, -Z, v, colors='lightgrey', zorder=10, linewidths=.5, transform=ccrs.PlateCarree());
ccc = plt.contour(lon, lat, -Z, vtext, colors='lightgrey', zorder=10, linewidths=.5, transform=ccrs.PlateCarree());
plt.clabel(ccc, inline=True, fontsize=10, fmt='%i')

# Add Gates Locations
plt.scatter(df_channel['Longitude (DD)'].values, df_channel['Latitude (DD)'].values,
         color='red', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(df_shelf['Longitude DD'].values, df_shelf['Latitude DD'].values,
         color='red', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(df_channel['Longitude (DD)'].values[idx_channel]  , df_channel['Latitude (DD)'].values[idx_channel],
         color='pink', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(df_shelf['Longitude DD'].values[idx_shelf]  , df_shelf['Latitude DD'].values[idx_shelf]  ,
         color='pink', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )


# Add Gates Labels
label_fontsize = 15
for channel in df_channel.Channel.unique():
        lons = df_channel[df_channel.Channel==channel]['Longitude (DD)']    
        lats = df_channel[df_channel.Channel==channel]['Latitude (DD)']    
        depths = df_channel[df_channel.Channel==channel]['depth']
        plt.text(lons[lons.index.min()], lats[lats.index.min()], channel+'  ',
                 fontsize=label_fontsize, horizontalalignment='right',
                 transform=ccrs.Geodetic(), zorder=200, color='red')
        
for gate in df_shelf.Gate.unique():
        lons = df_shelf[df_shelf.Gate==gate]['Longitude DD']    
        lats = df_shelf[df_shelf.Gate==gate]['Latitude DD']    
        depths = df_shelf[df_shelf.Gate==gate]['depth']
        plt.text(lons[lons.index.max()], lats[lats.index.max()], '  gate ' + str(gate) + ' ' + str(np.int32(depths)) +'m',
                 fontsize=label_fontsize, horizontalalignment='left',
                 transform=ccrs.Geodetic(), zorder=200, color='red')

        
        
# Save figure
fig.set_size_inches(w=18, h=12)
fig.savefig(fig_name, dpi=200)
print('Figure trimmed!')
os.system('convert -trim ' + fig_name + ' ' + fig_name)


