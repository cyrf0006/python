# -*- coding: utf-8 -*-
'''
Bottom salinity maps for Ocean Acidification work

To generate bottom climato:

import numpy as np
import azmp_utils as azu
dc = .25
lonLims = [-71, -42] # Olvia's 2014 paper
latLims = [41, 55.5]
lon_reg = np.arange(lonLims[0]+dc/2, lonLims[1]-dc/2, dc)
lat_reg = np.arange(latLims[0]+dc/2, latLims[1]-dc/2, dc)
azu.get_bottomS_climato('/home/cyrf0006/data/dev_database/*.nc', lon_reg, lat_reg, season='fall', zlims=[10, 3000], h5_outputfile='Sbot_climato_OA_fall_0.25.h5') 

* see: /home/cyrf0006/AZMP/state_reports/bottomT

'''

import os
import netCDF4
import h5py
import xarray as xr
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import interp1d  # to remove NaNs in profiles
from scipy.interpolate import RegularGridInterpolator as rgi
import azmp_utils as azu
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon as PP

def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = PP( xy, facecolor=[.8, .8, .8])
    plt.gca().add_patch(poly)


dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
proj = 'merc'
zmax = 100 # do try to compute surface temp below that depth
zmin = 0
zsurf = 20 # average between zmin and zsurf
dz = 5 # vertical bins
season = 'fall'
year = '2014'

if season=='spring':
    climato_file = 'Sbot_climato_OA_spring_0.25.h5'
elif season=='fall':
    climato_file = 'Sbot_climato_OA_fall_0.25.h5'
elif season=='summer':
    climato_file = 'Sbot_climato_OA_summer_0.25.h5'
year_file = '/home/cyrf0006/data/dev_database/' + year + '.nc'

year_file = '/home/cyrf0006/data/dev_database/' + year + '.nc'

## ---- Load Climato data ---- ##    
print('Load ' + climato_file)
h5f = h5py.File(climato_file, 'r')
Sbot_climato = h5f['Sbot'][:]
lon_reg = h5f['lon_reg'][:]
lat_reg = h5f['lat_reg'][:]
Zitp = h5f['Zitp'][:]
h5f.close()

## ---- Derive some parameters ---- ##    
lon_0 = np.round(np.mean(lon_reg))
lat_0 = np.round(np.mean(lat_reg))
lonLims = [lon_reg[0], lon_reg[-1]]
latLims = [lat_reg[0], lat_reg[-1]]
lon_grid, lat_grid = np.meshgrid(lon_reg,lat_reg)
dc = np.diff(lon_reg[0:2])

## ---- NAFO divisions ---- ##
nafo_div = azu.get_nafo_divisions()

## ---- Get CTD data --- ##
print('Get ' + year_file)
ds = xr.open_mfdataset(year_file)

# Remome problematic datasets
print('!!Remove MEDBA data!!')
print('  ---> I Should be improme because I remove good data!!!!')
ds = ds.where(ds.instrument_ID!='MEDBA', drop=True)

# Selection of a subset region
ds = ds.where((ds.longitude>lonLims[0]) & (ds.longitude<lonLims[1]), drop=True)
ds = ds.where((ds.latitude>latLims[0]) & (ds.latitude<latLims[1]), drop=True)
# Select time (save several options here)
if season == 'summer':
    #ds = ds.sel(time=ds['time.season']=='JJA')
    ds = ds.sel(time=((ds['time.month']>=7)) & ((ds['time.month']<=9)))
elif season == 'spring':
    #ds = ds.sel(time=ds['time.season']=='MAM')
    ds = ds.sel(time=((ds['time.month']>=4)) & ((ds['time.month']<=6)))
elif season == 'fall':
    #ds = ds.sel(time=ds['time.season']=='SON')
    ds = ds.sel(time=((ds['time.month']>=10)) & ((ds['time.month']<=12)))
else:
    print('!! no season specified, used them all! !!')
    

# Vertical binning (on dataset; slower here as we don't need it)
# Restrict max depth to zmax defined earlier
ds = ds.sel(level=ds['level']<zmax)
lons = np.array(ds.longitude)
lats = np.array(ds.latitude)
# Vertical binning (on dataArray; more appropriate here
da_sal = ds['salinity']
bins = np.arange(dz/2.0, ds.level.max(), dz)
da_sal = da_sal.groupby_bins('level', bins).mean(dim='level')
#To Pandas Dataframe
df_sal = da_sal.to_pandas()
df_sal.columns = bins[0:-1] #rename columns with 'bins'
# Remove empty columns & drop coordinates (for cast identification on map)
idx_empty_rows = df_sal.isnull().all(1).nonzero()[0]
df_sal = df_sal.dropna(axis=0,how='all')
lons = np.delete(lons,idx_empty_rows)
lats = np.delete(lats,idx_empty_rows)
#df_temp.to_pickle('T_2000-2017.pkl')
print(' -> Done!')


## --- fill 3D cube --- ##  
print('Fill regular cube')
z = df_sal.columns.values
V = np.full((lat_reg.size, lon_reg.size, z.size), np.nan)

# Aggregate on regular grid
for i, xx in enumerate(lon_reg):
    for j, yy in enumerate(lat_reg):    
        idx = np.where((lons>=xx-dc/2) & (lons<xx+dc/2) & (lats>=yy-dc/2) & (lats<yy+dc/2))
        tmp = np.array(df_sal.iloc[idx].mean(axis=0))
        idx_good = np.argwhere(~np.isnan(tmp))
        if np.size(idx_good)==1:
            V[j,i,:] = np.array(df_sal.iloc[idx].mean(axis=0))
        elif np.size(idx_good)>1: # vertical interpolation between pts
            interp = interp1d(np.squeeze(z[idx_good]), np.squeeze(tmp[idx_good]))
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
    #print 'interpolate depth layer ' + np.str(k) + ' / ' + np.str(z.size) 
    # griddata (after removing nans)
    idx_good = np.argwhere(~np.isnan(tmp_vec))
    if idx_good.size>3: # will ignore depth where no data exist
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

# Get Tsurface
Ssurf = V[:,:,np.squeeze(np.where(z<=zsurf))].mean(axis=2)


# Contour of data to mask
contour_mask = np.load('/home/cyrf0006/AZMP/state_reports/bottomT/100m_contour_labrador.npy')
polygon_mask = Polygon(contour_mask)


# Salinity anomaly:
#anom = Ssurf-Ssurf_climato


## ## ---- Plot Anomaly ---- ##
## fig, ax = plt.subplots(nrows=1, ncols=1)
## m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution= 'i')
## #levels = np.linspace(-1, 1, 6)
## levels = np.array([-1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1])
## xi, yi = m(*np.meshgrid(lon_reg, lat_reg))
## c = m.contourf(xi, yi, anom, levels, cmap=plt.cm.RdBu_r, extend='both')
## cc = m.contour(xi, yi, -Zitp, [100, 500, 1000, 4000], colors='grey');
## plt.clabel(cc, inline=1, fontsize=10, fmt='%d')
## if season=='fall':
##     plt.title('Fall Surface Salinity ' + year + ' Anomaly')
## elif season=='spring':
##     plt.title('Spring Surface Salinity ' + year + ' Anomaly')
## else:
##     plt.title('Surface Salinity ' + year + '  Anomaly')
## m.fillcontinents(color='tan');
## m.drawparallels([40, 45, 50, 55, 60], labels=[0,0,0,0], fontsize=12, fontweight='normal');
## m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');
## cax = fig.add_axes([0.16, 0.05, 0.7, 0.025])
## cb = plt.colorbar(c, cax=cax, orientation='horizontal')
## cb.set_label(r'$\rm S$', fontsize=12, fontweight='normal')
## ## div_toplot = ['2J', '3K', '3L', '3N', '3O', '3Ps']
## ## for div in div_toplot:
## ##     div_lon, div_lat = m(nafo_div[div]['lon'], nafo_div[div]['lat'])
## ##     m.plot(div_lon, div_lat, 'k', linewidth=2)
## ##     ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold')    
## # Save Figure
## fig.set_size_inches(w=6, h=5)
## fig.set_dpi(200)
## outfile = 'surface_sal_anomaly_' + season + '_' + year + '.png'
## fig.savefig(outfile)
## os.system('convert -trim ' + outfile + ' ' + outfile)
## # Save French Figure
## plt.sca(ax)
## if season=='fall':
##     plt.title(u'Anomalie de salinité au fond - Automne ' + year )
## elif season=='spring':
##     plt.title(u'Anomalie de salinité au fond - Printemp ' + year )
## else:
##     plt.title(u'Anomalie de salinité au fond ' + year )
## fig.set_size_inches(w=6, h=5)
## fig.set_dpi(300)
## outfile = 'surface_sal_anomaly_' + season + '_' + year + '_FR.png'
## fig.savefig(outfile)
## os.system('convert -trim ' + outfile + ' ' + outfile)

## ---- Plot Salinity ---- ##
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution= 'i')
levels = np.linspace(30, 36, 13)
#levels = np.linspace(30, 36, 7)
xi, yi = m(*np.meshgrid(lon_reg, lat_reg))
c = m.contourf(xi, yi, Ssurf, levels, cmap=plt.cm.RdBu_r, extend='both')
cc = m.contour(xi, yi, -Zitp, [100, 500, 1000, 4000], colors='grey');
plt.clabel(cc, inline=1, fontsize=10, fmt='%d')
if season=='fall':
    plt.title('Fall Surface Salinity ' + year)
elif season=='spring':
    plt.title('Spring Surface Salinity ' + year)
else:
    plt.title('Surface Salinity ' + year)
m.fillcontinents(color='tan');
m.drawparallels([40, 45, 50, 55, 60], labels=[1,0,0,0], fontsize=12, fontweight='normal');
m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');
x, y = m(lons, lats)
m.scatter(x,y, s=10, marker='.',color='k')
cax = fig.add_axes([0.16, 0.05, 0.7, 0.025])
cb = plt.colorbar(c, cax=cax, orientation='horizontal')
cb.set_label(r'$\rm S$', fontsize=12, fontweight='normal')
# Save Figure
fig.set_size_inches(w=6, h=5)
fig.set_dpi(200)
outfile = 'surface_sal_' + season + '_' + year + '.png'
fig.savefig(outfile)
os.system('convert -trim ' + outfile + ' ' + outfile)
## # Save French Figure
## plt.sca(ax)
## if season=='fall':
##     plt.title(u'Salinité au fond - Automne ' + year )
## elif season=='spring':
##     plt.title(u'Salinité au fond - Printemp ' + year )
## else:
##     plt.title(u'Salinité au fond ' + year )
## fig.set_size_inches(w=6, h=5)
## fig.set_dpi(300)
## outfile = 'surface_sal_' + season + '_' + year + '_FR.png'
## fig.savefig(outfile)
## os.system('convert -trim ' + outfile + ' ' + outfile)

## ## ---- Plot Climato ---- ##
## fig, ax = plt.subplots(nrows=1, ncols=1)
## m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution= 'i')
## levels = np.linspace(30, 36, 13)
## #levels = np.linspace(30, 36, 7)
## xi, yi = m(*np.meshgrid(lon_reg, lat_reg))
## c = m.contourf(xi, yi, Ssurf_climato, levels, cmap=plt.cm.RdBu_r, extend='both')
## cc = m.contour(xi, yi, -Zitp, [100, 500, 1000, 4000], colors='grey');
## plt.clabel(cc, inline=1, fontsize=10, fmt='%d')
## if season=='fall':
##     plt.title('Fall Surface Salinity Climatology')
## elif season=='spring':
##     plt.title('Spring Surface Salinity Climatology')
## else:
##     plt.title('Surface Salinity Climatology')
## m.fillcontinents(color='tan');
## m.drawparallels([40, 45, 50, 55, 60], labels=[1,0,0,0], fontsize=12, fontweight='normal');
## m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');
## cax = fig.add_axes([0.16, 0.05, 0.7, 0.025])
## cb = plt.colorbar(c, cax=cax, orientation='horizontal')
## cb.set_label(r'$\rm S$', fontsize=12, fontweight='normal')
## ## div_toplot = ['2J', '3K', '3L', '3N', '3O', '3Ps']
## ## for div in div_toplot:
## ##     div_lon, div_lat = m(nafo_div[div]['lon'], nafo_div[div]['lat'])
## ##     m.plot(div_lon, div_lat, 'k', linewidth=2)
## ##     ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold')    
## # Save Figure
## fig.set_size_inches(w=6, h=5)
## fig.set_dpi(200)
## outfile = 'surface_sal_climato_' + season + '_' + year + '.png'
## fig.savefig(outfile)
## os.system('convert -trim ' + outfile + ' ' + outfile)
## # Save French Figure
## plt.sca(ax)
## if season=='fall':
##     plt.title(u'Climatoligie de salinité au fond - Automne ' + year )
## elif season=='spring':
##     plt.title(u'Climatologie de salinité au fond - Printemp ' + year )
## else:
##     plt.title(u'Climatologie de salinité au fond ' + year )
## fig.set_size_inches(w=6, h=5)
## fig.set_dpi(300)
## outfile = 'surface_sal_climato_' + season + '_' + year + '_FR.png'
## fig.savefig(outfile)
## os.system('convert -trim ' + outfile + ' ' + outfile)


## # Convert to a subplot
## os.system('montage surface_sal_climato_' + season + '_' + year + '.png surface_sal_' + season + '_' + year + '.png surface_sal_anomaly_' + season + '_' + year + '.png  -tile 3x1 -geometry +10+10  -background white  surfaceS_' + season + year + '.png') 
## # French
## os.system('montage surface_sal_climato_' + season + '_' + year + '_FR.png surface_sal_' + season + '_' + year + '_FR.png surface_sal_anomaly_' + season + '_' + year + '_FR.png  -tile 3x1 -geometry +10+10  -background white  surfaceS_' + season + year + '_FR.png') 

