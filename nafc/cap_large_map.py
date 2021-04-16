'''
This script generates a large map for zoom.
This is inspired from nafo_map_csas.py.
It goes in a subplot with cap_beach_map.py

see /home/cyrf0006/research/capelin/map

Frederic.Cyr@dfo-mpo.gc.ca
October 2020
'''

import os
import netCDF4
import h5py                                                                
os.environ['PROJ_LIB'] = '/home/cyrf0006/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import pandas as pd
import cmocean
import cmocean.cm as cmo
import azmp_utils as azu

import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GEBCO_2014_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-70, -40]
latLims = [38, 65]
proj = 'merc'
decim_scale = 4
fig_name = 'cap_map_large.png'
lightdeep = cmocean.tools.lighten(cmocean.cm.deep, .9)

zoom_lon = [-60, -60, -52, -52, -60]
zoom_lat = [46, 53, 53, 46, 46]

## ---- Bathymetry ---- ####
v = np.linspace(0, 3500, 8)

print('Load and grid bathymetry')
# h5 file
h5_outputfile = 'cap_bathymetry_large.h5'
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

    # Save data for later use
    h5f = h5py.File(h5_outputfile, 'w')
    h5f.create_dataset('lon', data=lon)
    h5f.create_dataset('lat', data=lat)
    h5f.create_dataset('Z', data=Z)
    h5f.close()
    print(' -> Done!')

## ---- NAFO divisions ---- ##
nafo_div = azu.get_nafo_divisions()

## ---- draw map ---- ##
print('--- Now plot ---')
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection=ccrs.Mercator())
ax.set_extent([lonLims[0], lonLims[1], latLims[0], latLims[1]], crs=ccrs.PlateCarree())
ax.add_feature(cpf.NaturalEarthFeature('physical', 'coastline', '50m', edgecolor='k', alpha=0.7, linewidth=0.6, facecolor='black'), zorder=1)
m=ax.gridlines(linewidth=0.5, color='black', draw_labels=True, alpha=0.5)
m.xlabels_top=False
m.ylabels_right=False
m.xlocator = mticker.FixedLocator(np.arange(-80, 5, 5))
m.ylocator = mticker.FixedLocator(np.arange(10,70,5))
m.xformatter = LONGITUDE_FORMATTER
m.yformatter = LATITUDE_FORMATTER
m.ylabel_style = {'size': 7, 'color': 'black', 'weight':'bold'}
m.xlabel_style = {'size': 7, 'color': 'black', 'weight':'bold'}
lightdeep = cmocean.tools.lighten(cmo.deep, 0.5)
c = plt.contourf(lon, lat, -Z, v, transform=ccrs.PlateCarree(), cmap=lightdeep, extend='max', zorder=5)
cc = plt.contour(lon, lat, -Z, [250, 500, 1000, 2000, 3000, 4000, 5000], colors='silver', linewidths=.5, transform=ccrs.PlateCarree(), zorder=10)
plt.clabel(cc, inline=True, fontsize=7, fmt='%i')

# Add NAFO divisions
div_toplot = ['2G', '2H', '2J', '3K', '3L', '3N', '3O', '3Ps', '3M']
#div_toplot = ['2J', '3K', '3L','3Ps']
for div in div_toplot:
    div_lon = nafo_div[div]['lon']
    div_lat = nafo_div[div]['lat']
    ax.plot(div_lon, div_lat, 'black', linewidth=2, transform=ccrs.PlateCarree(), zorder=100)
    if (div == '3Ps') | (div == '2H'):
        ax.text(np.mean(div_lon), np.mean(div_lat)+.3, div, horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
    else:
        ax.text(np.mean(div_lon), np.mean(div_lat), div, horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)

# add zoom box    
ax.plot(zoom_lon, zoom_lat, 'red', linewidth=2, transform=ccrs.PlateCarree(), zorder=100)
ax.plot([-60, -40], [53, 65], 'red', linewidth=2, transform=ccrs.PlateCarree(), zorder=100)
ax.plot([-60, -40], [46, 38], 'red', linewidth=2, transform=ccrs.PlateCarree(), zorder=100)


# add other text
#ax.text(-56, 48, 'Newfoundland', horizontalalignment='center', verticalalignment='center', fontsize=10, color='white', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
#ax.text(-58, 52.5, 'Labrador', horizontalalignment='center', verticalalignment='center', fontsize=10, color='white', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(-55.5, 60, 'Labrador Sea', horizontalalignment='left', verticalalignment='bottom', fontsize=10, color='white', fontweight='bold', transform=ccrs.PlateCarree(), zorder=100)

#### ---- Save Figure ---- ####
fig.set_size_inches(w=10, h=12)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
#plt.show()

# Montgage:
os.system('montage  cap_map_large.png cap_map_beach.png -tile 2x1 -geometry +10+10  -background white  cap_map_subplot.png') 
