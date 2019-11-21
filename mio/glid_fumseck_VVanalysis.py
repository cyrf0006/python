'''
A new script for glider processing.
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
import os

# ** This should also be in a separate function.
# SeaExplorer limits & target
target =[43.204, 8.543]
origin = [43.73, 8.35]
timeLims = [pd.to_datetime('2019-05-01 9:00:00'),  pd.to_datetime('2019-05-04 05:00:00')] # transect1
#timeLims = [pd.to_datetime('2019-05-04 04:00:00'),  pd.to_datetime('2019-05-06 04:00:00')] # transect2

# profiles to flag:
idx_flag = [39, 40, 81]


v1 = np.arange(27,30, .2)
ZMAX = 300
XLIM = 60
lonLims = [7, 10]
latLims = [42, 45]

## ---- Load both dataset ---- ##
ds = xr.open_dataset('/home/cyrf0006/data/gliders_data/SEA003/20190501/netcdf/SEA003_20190501_l1.nc')
latVec_orig = ds['latitude'].values
lonVec_orig = ds['longitude'].values
pres_orig = ds['pressure'].values
time_orig = ds['time'].values
pdtime = pd.to_datetime(time_orig)

# DataFrame
pres  = pd.Series(pres_orig, index=pdtime)
pres.dropna(inplace=True)
W = pres.diff()
df = pd.concat([pres, W], axis=1, keys={'pressure', 'vert_vel'})      

# HERE!!!!!! NEED TO SMOOTH




df.plot()
df.diff().plot()

plt.plot(df.diff()**2)  
plt.plot([timeLims[0], timeLims[0]], [0,600], '--k')
plt.plot([timeLims[1], timeLims[1]], [0,600], '--k')


# Lat/Lon SeaExplorer
ds = ds.sel(time=slice(timeLims[0], timeLims[1]))
latVec = ds['latitude'].values
lonVec = ds['longitude'].values
Z = ds.depth.values


# dPdz.mean()
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(dPdz.index, dPdz.columns, dPdz.rolling(50, axis=1, min_periods=25).mean().T)
plt.clim([0.7, 1.1])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, 600])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Mean Vertical velocity (dB/m)')
fig_name = 'fumseck_transect1_vv_mean.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
