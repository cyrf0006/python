'''
A new script for glider processing of preswot mission.

Ideally, a lot of material from here should be put in a function (e.g. track projection)
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import seawater
import pyglider as pg
import gsw
import cmocean as cmo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import netCDF4
import os
from math import radians, sin, cos


## ---- Region parameters ---- ##
# SeaExplorer limits & target
origin =[38.9425, 2.96987]
target = [37.88908, 2.64376]
timeLims = [pd.to_datetime('2018-05-04 22:00:00'),  pd.to_datetime('2018-05-10 00:00:00')] # transect1
#timeLims = [pd.to_datetime('2018-05-09 20:0:00'),  pd.to_datetime('2018-05-13 23:30:00')] # transect2
#timeLims = [pd.to_datetime('2018-05-13 19:00:00'), pd.to_datetime('2018-05-15 14:00:00')] # transect3

# Slocum limits and target
origin_sl =[38.9425, 2.91]
target_sl = [37.88908, 2.5847]
timeLims_sl = [pd.to_datetime('2018-05-05 13:00:00'),  pd.to_datetime('2018-05-09 20:00:00')] # transect1
#timeLims_sl = [pd.to_datetime('2018-05-09 13:00:00'),  pd.to_datetime('2018-05-15 8:00:00')] # transect2
#timeLims_sl = []

lonLims = [2, 3.5]
latLims = [37.5, 39.6]
decim_scale = 1
fig_name = 'map_preswot.png'
v = np.linspace(50, 500,10)
vtext = np.linspace(100, 500,5)

## --------- Get Bathymetry -------- ####
## print('Get bathy...')
## dataFile = '/home/cyrf0006/data/GEBCO/GEBCO_2014_1D.nc' # Maybe find a better way to handle this file
## # Load data
## dataset = netCDF4.Dataset(dataFile)
## x = [-179-59.75/60, 179+59.75/60] 
## y = [-89-59.75/60, 89+59.75/60]
## spacing = dataset.variables['spacing']
## # Compute Lat/Lon
## nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
## ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir
## lon = np.linspace(x[0],x[-1],nx)
## lat = np.linspace(y[0],y[-1],ny)
## # interpolate data on regular grid (temperature grid)
## # Reshape data
## zz = dataset.variables['z']
## Z = zz[:].reshape(ny, nx)
## Z = np.flipud(Z) # <------------ important!!!
## # Reduce data according to Region params
## idx_lon = np.where((lon>=lonLims[0]) & (lon<=lonLims[1]))
## idx_lat = np.where((lat>=latLims[0]) & (lat<=latLims[1]))
## Z = Z[idx_lat[0][0]:idx_lat[0][-1]+1, idx_lon[0][0]:idx_lon[0][-1]+1]
## lon = lon[idx_lon[0]]
## lat = lat[idx_lat[0]]

## lon = lon[::decim_scale]
## lat = lat[::decim_scale]
## Z = Z[::decim_scale, ::decim_scale]
## print(' -> Done!')


## ---- Load both dataset ---- ##
dsSX = xr.open_dataset('/home/cyrf0006/data/gliders_data/netcdf_preswot/SeaExplorer/SEA003_20180503_l2.nc')
dsSL = xr.open_dataset('/home/cyrf0006/data/gliders_data/netcdf_preswot/Slocum/dep0023_sdeep00_scb-sldeep000_L2_2018-05-03_data_dt.nc')
latVec_orig = dsSX['latitude'].values
lonVec_orig = dsSX['longitude'].values
latVec_orig_sl = dsSL['latitude'].values
lonVec_orig_sl = dsSL['longitude'].values

# Lat/Lon SeaExplorer
dsSX = dsSX.sel(time=slice(timeLims[0], timeLims[1]))
latVec = dsSX['latitude'].values
lonVec = dsSX['longitude'].values
Z = dsSX.depth.values
# Lat/Lon Slocum
dsSL = dsSL.sel(time=slice(timeLims_sl[0], timeLims_sl[1]))
latVec_sl = dsSL['latitude'].values
lonVec_sl = dsSL['longitude'].values
Z_sl = dsSL.depth.values

## ---- Load vertically averaged currents ---- ##
df_c = pd.read_csv('mean_currents.csv', sep=';')
df_c.set_index('yo_number', drop=True, inplace=True)
ulat = df_c.start_arrow_lat.values
ulon = df_c.start_arrow_lon.values
u_norm = df_c['average current'].values
ux = np.full_like(ulat, 0)
uy = np.full_like(ulat, 0)
for idx, alpha in enumerate(df_c.bearing):
    ux[idx] = u_norm[idx]*cos(alpha) # angle from north...
    uy[idx] = u_norm[idx]*sin(alpha)


## ---- Find projection ---- ##
import coord_list
## 1. SeaExplorer
interval = 20.0 #meters
azimuth = coord_list.calculateBearing(origin[0], origin[1], target[0], target[1]) # this works but angle above not.
coords = np.array(coord_list.main(interval,azimuth,origin[0], origin[1], target[0], target[1]))
lats = coords[:,0]
lons = coords[:,1]

I2 = np.argmin(np.abs(latVec-target[0]) + np.abs(lonVec-target[1]))
I1 = np.argmin(np.abs(latVec-origin[0]) + np.abs(lonVec-origin[1]))
theIndex = np.arange(np.min([I1, I2]), np.max([I1, I2]))
del I1, I2
distVec = np.full_like(theIndex, np.nan, dtype=np.double)  
new_lat = np.full_like(theIndex, np.nan, dtype=np.double)  
new_lon = np.full_like(theIndex, np.nan, dtype=np.double)  
for re_idx, idx in enumerate(theIndex):
    idx_nearest = np.argmin(np.abs(latVec[idx]-lats) + np.abs(lonVec[idx]-lons))
    new_lat[re_idx] = coords[idx_nearest,0]
    new_lon[re_idx] = coords[idx_nearest,1]

    d = seawater.dist([origin[0], lats[idx_nearest]], [origin[1], lons[idx_nearest]])
    distVec[re_idx] = d[0]
    
## 2. Slocum
azimuth_sl = coord_list.calculateBearing(origin_sl[0], origin_sl[1], target_sl[0], target_sl[1])
coords_sl = np.array(coord_list.main(interval,azimuth_sl,origin_sl[0], origin_sl[1], target_sl[0], target_sl[1]))
lats_sl = coords_sl[:,0]
lons_sl = coords_sl[:,1]

I2 = np.argmin(np.abs(latVec_sl-target_sl[0]) + np.abs(lonVec_sl-target_sl[1]))
I1 = np.argmin(np.abs(latVec_sl-origin_sl[0]) + np.abs(lonVec_sl-origin_sl[1]))
theIndex_sl = np.arange(np.min([I1, I2]), np.max([I1, I2]))

distVec_sl = np.full_like(theIndex_sl, np.nan, dtype=np.double)  
new_lat_sl = np.full_like(theIndex_sl, np.nan, dtype=np.double)  
new_lon_sl = np.full_like(theIndex_sl, np.nan, dtype=np.double)  
for re_idx, idx in enumerate(theIndex_sl):
    idx_nearest = np.argmin(np.abs(latVec_sl[idx]-lats_sl) + np.abs(lonVec_sl[idx]-lons_sl))
    new_lat_sl[re_idx] = coords_sl[idx_nearest,0]
    new_lon_sl[re_idx] = coords_sl[idx_nearest,1]

    d = seawater.dist([origin_sl[0], lats_sl[idx_nearest]], [origin_sl[1], lons_sl[idx_nearest]])
    distVec_sl[re_idx] = d[0]
    

## --------- plot map using cartopy -------- ####
projection=ccrs.Mercator()
extent = [lonLims[0], lonLims[1], latLims[0], latLims[1]]

fig = plt.figure(figsize=(13.3, 10))                      
ax = fig.add_subplot(111, projection=projection)

lon_labels = np.arange(lonLims[0]-2, lonLims[1]+2, .5)
lat_labels = np.arange(latLims[0]-2, latLims[1]+2, .5)

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
#c = plt.contourf(lon, lat, -Z, v, cmap=cmocean.cm.deep, extend="both",  zorder=1, transform=ccrs.PlateCarree());
#ccc = plt.contour(lon, lat, -Z, vtext, colors='lightgrey', zorder=10, linewidths=.5, transform=ccrs.PlateCarree());
#plt.clabel(ccc, inline=True, fontsize=10, fmt='%i')

# Add Glider tracks
#plt.plot(lonVec[theIndex], latVec[theIndex], '.m') 
#plt.plot(new_lon, new_lat, '.r')
plt.plot(lonVec_orig, latVec_orig,
         color='darkorange', linewidth=2,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot(lonVec_orig_sl, latVec_orig_sl,
         color='gold', linewidth=2,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot([origin[1], target[1]], [origin[0], target[0]],
         color='black', 
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot([origin_sl[1], target_sl[1]], [origin_sl[0], target_sl[0]],
         color='black',
         transform=ccrs.PlateCarree(), zorder=20
         )
## plt.plot(lonVec[theIndex], latVec[theIndex], '.r', alpha=.5,
##          transform=ccrs.PlateCarree(), zorder=20
         ## )
plt.plot(new_lon, new_lat, '.r', alpha=1,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot(new_lon_sl, new_lat_sl, '.m', alpha=1,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.legend(['SeaExplorer track', 'Slocum track', 'projection line', 'projection line', 'SeaExplorer projected T1', 'Slocum projected T1'], fontsize=12, loc='lower right')

# Plot current arrows
plt.quiver(ulon[30:46], ulat[30:46], ux[30:46], uy[30:46], transform=ccrs.PlateCarree(), zorder=50)
    
# Save figure
fig.set_size_inches(w=18, h=12)
fig.savefig(fig_name, dpi=200)
print('Figure trimmed!')
os.system('convert -trim ' + fig_name + ' ' + fig_name)
