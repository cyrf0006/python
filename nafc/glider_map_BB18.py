'''
Bottom temperature maps from custom reference period and region.
This script uses Gebco 30sec bathymetry to reference the bottom after a 3D interpolation of the temperature field.
To do:

- maybe add a max depth to be more computationally efficient
- Flag wrong data...
'''

import netCDF4
import xarray as xr
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import interp1d  # to remove NaNs in profiles
from scipy.interpolate import RegularGridInterpolator as rgi
import cmocean

## ---- Glider data ---- ##
ds = xr.open_dataset('/home/cyrf0006/data/glider_sdata/SEA022/20181106/netcdf/SEA022_20181106_l2.nc')
ds = xr.open_dataset('/home/cyrf0006/data/gliders_data/SEA024/20190807/netcdf/SEA024_20190807_l2.nc')

lons = np.array(ds.longitude)
lats = np.array(ds.latitude)

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-55, -47] # Maximum MPA zone
latLims = [46, 52]
#lonLims = [-60, -50] # For test
#latLims = [42, 48]
proj = 'merc'
#spring = True
spring = False
fig_name = 'map_glider_BB18.png'
zmax = 1000 # do try to compute bottom temp below that depth
dz = 5 # vertical bins
dc = .1
lon_reg = np.arange(lonLims[0]+dc/2, lonLims[1]-dc/2, dc)
lat_reg = np.arange(latLims[0]+dc/2, latLims[1]-dc/2, dc)
lon_grid, lat_grid = np.meshgrid(lon_reg,lat_reg)
stationFile = '/home/cyrf0006/github/AZMP-NL/data/STANDARD_SECTIONS.xlsx'

## ---- Station info ---- ##
df = pd.read_excel(stationFile)
#get the values for a given column
sections = df['SECTION'].values
stations = df['STATION'].values
stationLat = df['LAT'].values
stationLon = df['LONG.1'].values
idx = df.SECTION[df.SECTION=="BONAVISTA"].index.tolist()
stationLat = stationLat[idx]
stationLon = stationLon[idx]
stations = stations[idx]
stations = [x.replace('-', '') for x in stations]


## ---- Bathymetry ---- ####
print('Load and grid bathymetry')
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
# interpolate data on regular grid (temperature grid)
lon_grid_bathy, lat_grid_bathy = np.meshgrid(lon,lat)
lon_vec_bathy = np.reshape(lon_grid_bathy, lon_grid_bathy.size)
lat_vec_bathy = np.reshape(lat_grid_bathy, lat_grid_bathy.size)
z_vec = np.reshape(Z, Z.size)
Zitp = griddata((lon_vec_bathy, lat_vec_bathy), z_vec, (lon_grid, lat_grid), method='linear')
print(' -> Done!')

## ---- Plot map ---- ##
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='f')
levels = np.linspace(0, 4000, 81)
xi, yi = m(*np.meshgrid(lon_reg, lat_reg))
#lon_casts, lat_casts = m(lons[idx], lats[idx])
#c = m.pcolor(xi, yi, Tbot, levels, cmap=plt.cm.RdBu_r, extend='both')
x,y = m(*np.meshgrid(lon,lat))
c = m.contourf(x, y, -Z, levels, cmap=cmocean.cm.deep, extend='both')
cc = m.contour(x, y, -Z, [100, 300, 500, 1000, 4000], colors='grey');
plt.clabel(cc, inline=1, fontsize=10, fmt='%d')
m.fillcontinents(color='tan');

# Plot glider track
x, y = m(lons, lats)
m.scatter(x,y, s=10, marker='.',color='r')

# Plot BB stations
x, y = m(stationLon, stationLat)
m.scatter(x,y,30,marker='o',color='k')
for i, stn in enumerate(stations):
    if stn in ['BB01', 'BB05', 'BB10', 'BB15']:
        plt.text(x[i], y[i], stn+' ', horizontalalignment='right', verticalalignment='bottom', fontsize=10, color='k', fontweight='bold')


m.drawparallels(np.arange(latLims[0],latLims[1]), labels=[1,0,0,0], fontsize=12, fontweight='normal')
m.drawmeridians(np.arange(lonLims[0],lonLims[1]), labels=[0,0,0,1], fontsize=12, fontweight='normal')


#### ---- Save Figure ---- ####
fig.set_size_inches(w=8, h=8)
fig.set_dpi(200)
fig.savefig(fig_name)

