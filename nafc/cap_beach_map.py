'''
Beaches map for capelin paper.
It goes in a subplot with cap_large_map.py

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

import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GEBCO_2014_1D.nc'
lon_0 = -50
lat_0 = 50
#lonLims = [-60, -52]
#latLims = [46, 52]
lonLims = [-60, -52]
latLims = [46, 53]
proj = 'merc'
decim_scale = 4
beachFile = 'lat_long_all_spawning_beaches.xlsx'
v = np.linspace(0, 500, 11)
s27 = [47.54667,-52.58667]
fig_name = 'cap_map_beach.png'

CB = [47.75, -52.95]
TB = [48.2, -53.2]
BB = [48.75, -53.4]
ND = [49.7, -55.3]
WB = [50.2, -56.35]
BV = [47.55, -56.3]
FB = [47.3, -55.6]
PB = [47.1, -54.5]
SM = [46.8, -53.8]

## ---- Bathymetry ---- ##
print('Load and grid bathymetry')
# h5 file
h5_outputfile = 'cap_bathymetry.h5'
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


## ---- Station info ---- ##
df = pd.read_excel(beachFile)

## ---- draw map ---- ##
print('--- Now plot ---')
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection=ccrs.Mercator())
ax.set_extent([lonLims[0], lonLims[1], latLims[0], latLims[1]], crs=ccrs.PlateCarree())
ax.add_feature(cpf.NaturalEarthFeature('physical', 'coastline', '50m', edgecolor='k', alpha=0.7, linewidth=0.6, facecolor='black'), zorder=1)
m=ax.gridlines(linewidth=0.5, color='black', draw_labels=True, alpha=0.5)
m.xlabels_top=False
m.ylabels_right=False
m.xlocator = mticker.FixedLocator([-60, -58, -56, -54, -52])
m.ylocator = mticker.FixedLocator([46, 48, 50, 52, 54])
m.xformatter = LONGITUDE_FORMATTER
m.yformatter = LATITUDE_FORMATTER
m.ylabel_style = {'size': 7, 'color': 'black', 'weight':'bold'}
m.xlabel_style = {'size': 7, 'color': 'black', 'weight':'bold'}
lightdeep = cmocean.tools.lighten(cmo.deep, 0.5)
c = plt.contourf(lon, lat, -Z, v, transform=ccrs.PlateCarree(), cmap=lightdeep, extend='max', zorder=5)
cc = plt.contour(lon, lat, -Z, [100, 500], colors='silver', linewidths=.5, transform=ccrs.PlateCarree(), zorder=10)
plt.clabel(cc, inline=True, fontsize=7, fmt='%i')

# plot Beaches
beach = df.Short_name.values
lats = df.Lat.values
lons = df.Long.values
ax.plot(lons, lats, '.', color='palevioletred', transform=ccrs.PlateCarree(), markersize=10, zorder=10)

# research Beaches
res_beach = df[(df.Short_name=="Bryant's Cove") | (df.Short_name=="Bellevue Beach")]
ax.plot(res_beach.Long, res_beach.Lat, '*', color='green', transform=ccrs.PlateCarree(), markersize=15, zorder=10)

# Plot stn 27
ax.plot(s27[1], s27[0], '*', color='red', transform=ccrs.PlateCarree(), markersize=15, zorder=10)

# Identify name of the Bays
ax.text(CB[1], CB[0], 'CB', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(TB[1], TB[0], 'TB', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(BB[1], BB[0], 'BB', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(ND[1], ND[0], 'ND', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(WB[1], WB[0], 'WB', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(FB[1], FB[0], 'FB', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(PB[1], PB[0], 'PB', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(SM[1], SM[0], 'SM', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
#ax.text(BV[1], BV[0], 'BV', horizontalalignment='center', verticalalignment='center', fontsize=10, color='black', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(-56, 48.6, 'Newfoundland', horizontalalignment='center', verticalalignment='center', fontsize=10, color='white', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)
ax.text(-57, 52.15, 'Labrador', horizontalalignment='center', verticalalignment='center', fontsize=10, color='white', fontweight='bold', transform=ccrs.PlateCarree(), zorder=10)

# Custom legend
import matplotlib.lines as mlines
legend_elements = [mlines.Line2D([],[], marker='.',linestyle='None', color='palevioletred', markersize=15, label='Diary Beach'),
                   mlines.Line2D([],[], marker='*',linestyle='None', color='green', markersize=15, label='Research Beach'),
                   mlines.Line2D([],[], marker='*',linestyle='None', color='red', markersize=15, label='Stn 27')]
ax.legend(handles=legend_elements)


# Save    
fig.set_size_inches(w=10, h=12)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# Montgage:
os.system('montage  cap_map_large.png cap_map_beach.png -tile 2x1 -geometry +10+10  -background white  cap_map_subplot.png') 
