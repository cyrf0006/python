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


## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-66.5, -47] # Maximum MPA zone
latLims = [40, 62]
#lonLims = [-60, -50] # For test
#latLims = [42, 48]
proj = 'merc'
#spring = True
spring = False
fig_name = 'map_bottom_temp.png'
zmax = 1000 # do try to compute bottom temp below that depth
dz = 5 # vertical bins
dc = .1
lon_reg = np.arange(lonLims[0]+dc/2, lonLims[1]-dc/2, dc)
lat_reg = np.arange(latLims[0]+dc/2, latLims[1]-dc/2, dc)
lon_grid, lat_grid = np.meshgrid(lon_reg,lat_reg)
season = 'fall'
z_sst = 15

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

## ---- Get CTD data --- ##
print('Get historical data')
ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/*.nc')
# Selection of a subset region
ds = ds.where((ds.longitude>lonLims[0]) & (ds.longitude<lonLims[1]), drop=True)
ds = ds.where((ds.latitude>latLims[0]) & (ds.latitude<latLims[1]), drop=True)
# Select time (save several options here)
# Select time (save several options here)
if season == 'summer':
    ds = ds.sel(time=ds['time.season']=='JJA')
elif season == 'spring':
    ds = ds.sel(time=ds['time.season']=='MAM')
elif season == 'fall':
    ds = ds.sel(time=ds['time.season']=='SON')
else:
    print('!! no season specified, used them all! !!')
    
ds = ds.sel(time=ds['time.year']>=1971)
#ds = ds.sel(time=ds['time.year']<=1998)
# Vertical binning (on dataset; slower here as we don't need it)
#bins = np.arange(dz/2.0, df_temp.columns.max(), dz)
#ds = ds.groupby_bins('level', bins).mean(dim='level')
# Restrict max depth to zmax defined earlier
ds = ds.sel(level=ds['level']<z_sst)

# Vertical binning (on dataArray; more appropriate here
da_temp = ds['temperature']
bins = np.arange(dz/2.0, ds.level.max(), dz)
da_temp = da_temp.groupby_bins('level', bins).mean(dim='level')
#To Pandas Dataframe
df_temp = da_temp.to_pandas()
df_temp.columns = bins[0:-1] #rename columns with 'bins'
#df_temp.to_pickle('T_2000-2017.pkl')
print(' -> Done!')


## --- fill 3D cube --- ##  
print('Fill regular cube')
lons = np.array(ds.longitude)
lats = np.array(ds.latitude)
z = df_temp.columns.values
V = np.full((lat_reg.size, lon_reg.size, z.size), np.nan)

# Aggregate on regular grid
for i, xx in enumerate(lon_reg):
    for j, yy in enumerate(lat_reg):    
        idx = np.where((lons>=xx-dc/2) & (lons<xx+dc/2) & (lats>=yy-dc/2) & (lats<yy+dc/2))
        tmp = np.array(df_temp.iloc[idx].mean(axis=0))
        idx_good = np.argwhere((~np.isnan(tmp)) & (tmp<30))
        if np.size(idx_good)==1:
            V[j,i,:] = np.array(df_temp.iloc[idx].mean(axis=0))
        elif np.size(idx_good)>1: # vertical interpolation between pts
            #V[j,i,:] = np.interp((z), np.squeeze(z[idx_good]), np.squeeze(tmp[idx_good]))  <--- this method propagate nans below max depth (extrapolation)
            interp = interp1d(np.squeeze(z[idx_good]), np.squeeze(tmp[idx_good]))  # <---------- Pay attention here, this is a bit unusual, but seems to work!
            idx_interp = np.arange(np.int(idx_good[0]),np.int(idx_good[-1]+1))
            V[j,i,idx_interp] = interp(z[idx_interp]) # interpolate only where possible (1st to last good idx)

                   
# horizontal interpolation at each depth
lon_grid, lat_grid = np.meshgrid(lon_reg,lat_reg)
lon_vec = np.reshape(lon_grid, lon_grid.size)
lat_vec = np.reshape(lat_grid, lat_grid.size)
for k, zz in enumerate(z):
    # Meshgrid 1D data (after removing NaNs)
    tmp_grid = V[:,:,k]
    tmp_vec = np.reshape(tmp_grid, tmp_grid.size)
    print 'interpolate depth layer ' + np.str(k) + ' / ' + np.str(z.size) 
    # griddata (after removing nans)
    idx_good = np.argwhere(~np.isnan(tmp_vec))
    if idx_good.size: # will ignore depth where no data exist
        LN = np.squeeze(lon_vec[idx_good])
        LT = np.squeeze(lat_vec[idx_good])
        TT = np.squeeze(tmp_vec[idx_good])
        zi = griddata((LN, LT), TT, (lon_grid, lat_grid), method='linear')
        V[:,:,k] = zi
    else:
        continue
print(' -> Done!')    

# mask using bathymetry (I don't think it is necessary, but make nice figures)
for i, xx in enumerate(lon_reg):
    for j,yy in enumerate(lat_reg):
        if Zitp[j,i] > -10: # remove shallower than 10m
            V[j,i,:] = np.nan

# getting bottom temperature
print('Getting SST')
idx_sst = np.squeeze(np.where(bins<z_sst))
#SST = V[:,:,idx_sst].mean(axis=2)
SST = V.mean(axis=2)
print(' -> Done!')    
            
## ---- Plot map ---- ##
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='f')
levels = np.linspace(0, 15, 16)
xi, yi = m(*np.meshgrid(lon_reg, lat_reg))
#lon_casts, lat_casts = m(lons[idx], lats[idx])
c = m.contourf(xi, yi, SST, levels, cmap=plt.cm.RdBu_r, extend='both')
#c = m.pcolor(xi, yi, Tbot, levels, cmap=plt.cm.RdBu_r, extend='both')
x,y = m(*np.meshgrid(lon,lat))
cc = m.contour(x, y, -Z, [100, 500, 1000, 4000], colors='grey');
m.fillcontinents(color='tan');

m.drawparallels([40, 45, 50, 55, 60], labels=[1,0,0,0], fontsize=12, fontweight='normal');
m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');

# Draw NAFO divisions
## x,y = m(np.array(div3K['lon']), np.array(div3K['lat']))
## m.plot(x,y,color='black')
## ax.text(np.mean(x), np.mean(y), '3K', fontsize=12, color='black', fontweight='bold')
## x,y = m(np.array(div3L['lon']), np.array(div3L['lat']))
## m.plot(x,y,color='black')
## ax.text(np.mean(x), np.mean(y), '3L', fontsize=12, color='black', fontweight='bold')
## x,y = m(np.array(div3N['lon']), np.array(div3N['lat']))
## m.plot(x,y,color='black')
## ax.text(np.mean(x), np.mean(y), '3N', fontsize=12, color='black', fontweight='bold')
## x,y = m(np.array(div3O['lon']), np.array(div3O['lat']))
## m.plot(x,y,color='black')
## ax.text(np.mean(x)*.9, np.mean(y), '3O', fontsize=12, color='black', fontweight='bold')
## x,y = m(np.array(div3Ps['lon']), np.array(div3Ps['lat']))
## m.plot(x,y,color='black')
## ax.text(np.mean(x)*.7, np.mean(y)*.95, '3Ps', fontsize=12, color='black', fontweight='bold')
## x,y = m(np.array(div2J['lon']), np.array(div2J['lat']))
## m.plot(x,y,color='black')
## ax.text(np.mean(x), np.mean(y), '2J', fontsize=12, color='black', fontweight='bold')

cax = plt.axes([0.85,0.15,0.04,0.7])
cb = plt.colorbar(c, cax=cax, ticks=levels)
cb.set_label(r'$\rm T(^{\circ}C)$', fontsize=12, fontweight='normal')
#plt.subplots_adjust(left=.07, bottom=.07, right=.93, top=.9, wspace=.2, hspace=.2)


#### ---- Save Figure ---- ####
#plt.suptitle('Spring surveys', fontsize=16)
fig.set_size_inches(w=6, h=9)
#fig.tight_layout() 
fig.set_dpi(200)
fig.savefig('sst.png')

#### ---- Save CSV data ---- ####
# Replace NaNs by -9999
idx_nan = np.where(np.isnan(SST))
SST[idx_nan] = -9999
# define header
SST_flip = np.flipud(SST)
#header = '{0:^12s} {1:^6s}\n{2:^12s} {3:^6s}\n{4:^12s} {5:^6s}\n{6:^12s} {7:^6s}\n{8:^12s} {9:^6s}\n{10:^12s} {11:^6s}\n'.format('ncols', '195', 'nrows', '220', 'xllcorner', '293500', 'yllcorner', '40000', 'cellsize', '1', 'NODATA_value', '-9999')
header = '{0:^1s} {1:^1s}\n{2:^1s} {3:^1s}\n{4:^1s} {5:^1s}\n{6:^1s} {7:^1s}\n{8:^1s} {9:^1s}\n{10:^1s} {11:^1s}'.format('NCOLS', np.str(lon_reg.size), 'NROWS', np.str(lat_reg.size), 'XLLCORNER', np.str(lon_reg.min()), 'YLLCORNER', np.str(lat_reg.min()), 'CELLSIZE', np.str(dc), 'NODATA_VALUE', '-9999')

# Save
np.savetxt("sst.asc", SST_flip, delimiter=" ", header=header, fmt='%5.2f', comments='')
