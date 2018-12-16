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
import gsw
from scipy import stats

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-60, -44] # FishHab region
latLims = [39, 56]
proj = 'merc'
decim_scale = 4
#spring = True
spring = False
fig_name = 'map_bottom_temp.png'
zmax = 100 # do try to compute bottom temp below that depth
dz = 5 # vertical bins
dc = .5
lon_reg = np.arange(lonLims[0]+dc/2, lonLims[1]-dc/2, dc)
lat_reg = np.arange(latLims[0]+dc/2, latLims[1]-dc/2, dc)
lon_grid, lat_grid = np.meshgrid(lon_reg,lat_reg)
season = 'summer'

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
Zbathy = zz[:].reshape(ny, nx)
Zbathy = np.flipud(Zbathy) # <------------ important!!!
# Reduce data according to Region params
idx_lon = np.where((lon>=lonLims[0]) & (lon<=lonLims[1]))
idx_lat = np.where((lat>=latLims[0]) & (lat<=latLims[1]))
Zbathy = Zbathy[idx_lat[0][0]:idx_lat[0][-1]+1, idx_lon[0][0]:idx_lon[0][-1]+1]
lon = lon[idx_lon[0]]
lat = lat[idx_lat[0]]
# Reduce data according to Region params
lon = lon[::decim_scale]
lat = lat[::decim_scale]
Zbathy = Zbathy[::decim_scale, ::decim_scale]

# interpolate data on regular grid (temperature grid)
lon_grid_bathy, lat_grid_bathy = np.meshgrid(lon,lat)
lon_vec_bathy = np.reshape(lon_grid_bathy, lon_grid_bathy.size)
lat_vec_bathy = np.reshape(lat_grid_bathy, lat_grid_bathy.size)
z_vec = np.reshape(Zbathy, Zbathy.size)
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
#ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/2017.nc')
# Selection of a subset region
ds = ds.where((ds.longitude>lonLims[0]) & (ds.longitude<lonLims[1]), drop=True)
ds = ds.where((ds.latitude>latLims[0]) & (ds.latitude<latLims[1]), drop=True)
# Select time (save several options here)
if season == 'summer':
    ds = ds.sel(time=((ds['time.month']>=7)) & ((ds['time.month']<=9)))
elif season == 'spring':
    ds = ds.sel(time=((ds['time.month']>=4)) & ((ds['time.month']<=6)))
elif season == 'fall':
    ds = ds.sel(time=((ds['time.month']>=10)) & ((ds['time.month']<=12)))
elif season == 'icefree':
    ds = ds.sel(time=((ds['time.month']>=3)) & ((ds['time.month']<=12)))
else:
    print('!! no season specified, used them all! !!')

                
ds = ds.sel(time=ds['time.year']>=1998)
# Vertical binning (on dataset; slower here as we don't need it)
#bins = np.arange(dz/2.0, df_temp.columns.max(), dz)
#ds = ds.groupby_bins('level', bins).mean(dim='level')
# Restrict max depth to zmax defined earlier
ds = ds.sel(level=ds['level']<zmax)

# Vertical binning (on dataArray; more appropriate here
da_temp = ds['temperature']
bins = np.arange(dz/2.0, ds.level.max(), dz)
da_temp = da_temp.groupby_bins('level', bins).mean(dim='level')
#To Pandas Dataframe
df_temp = da_temp.to_pandas()
df_temp.columns = bins[0:-1] #rename columns with 'bins'

da_sal = ds['salinity']
bins = np.arange(dz/2.0, ds.level.max(), dz)
da_sal = da_sal.groupby_bins('level', bins).mean(dim='level')
#To Pandas Dataframe
df_sal = da_sal.to_pandas()
df_sal.columns = bins[0:-1] #rename columns with 'bins'

da_lat = ds['latitude']
df_lat = da_lat.to_pandas()
da_lon = ds['longitude']
df_lon = da_lon.to_pandas()
print(' -> Done!')


## --- Compute N2 trend for each pixel --- ##  
print('Compute N2 trends for each pixel')
index_unique = df_temp.index.year.unique()

lons = np.array(ds.longitude)
lats = np.array(ds.latitude)
z = df_temp.columns.values
Tmap = np.full((lat_reg.size, lon_reg.size), np.nan)
Smap = np.full((lat_reg.size, lon_reg.size), np.nan)
Nmap = np.full((lat_reg.size, lon_reg.size), np.nan)


# Aggregate on regular grid
for i, xx in enumerate(lon_reg): # space loop
    print xx
    for j, yy in enumerate(lat_reg):
        df_Ttmp = df_temp[(df_lon>=xx-dc/2) & (df_lon<=xx+dc/2) & (df_lat>=yy-dc/2) & (df_lat<=yy+dc/2)]
        df_Stmp = df_sal[(df_lon>=xx-dc/2) & (df_lon<=xx+dc/2) & (df_lat>=yy-dc/2) & (df_lat<=yy+dc/2)]
        df_Ttmp = df_Ttmp.resample('A').mean()
        df_Stmp = df_Stmp.resample('A').mean()

        TVec = []
        SVec = []
        NVec = []
        yearVec = []
        for t, year in enumerate(index_unique): #time loop
            T = df_Ttmp[df_Ttmp.index.year==year].values.squeeze()
            S = df_Stmp[df_Stmp.index.year==year].values.squeeze()

            if T.size < 1: # no data for this year
                pass
            else:            
                Z = df_Ttmp.columns.values
                T[(~np.isnan(T)) & (~np.isnan(S))]
                TT = T[(~np.isnan(T)) & (~np.isnan(S))]
                SS = S[(~np.isnan(T)) & (~np.isnan(S))]
                ZZ = Z[(~np.isnan(T)) & (~np.isnan(S))]
                
                SA = gsw.SA_from_SP(SS,ZZ,xx,yy)
                CT = gsw.CT_from_t(SA,TT,ZZ)
                N2, pmid = gsw.Nsquared(SA, CT, ZZ, yy)

                yearVec.append(year)
                TVec.append(np.nanmean(TT))
                SVec.append(np.nanmean(SS))
                NVec.append(np.sqrt(np.nanmean(N2)))


            # compute trends if serie long enough
            x = np.array(yearVec)
            y = np.array(TVec)
            idx = np.where(~np.isnan(y))
            if np.size(idx)>5:
                m, b, r_value, p_value, std_err = stats.linregress(x[idx], y[idx])
                Tmap[j,i] = m
            y = np.array(SVec)
            idx = np.where(~np.isnan(y))
            if np.size(idx)>5:
                m, b, r_value, p_value, std_err = stats.linregress(x[idx], y[idx])
                Smap[j,i] = m            
            y = np.array(NVec)
            idx = np.where(~np.isnan(y))
            if np.size(idx)>5:
                m, b, r_value, p_value, std_err = stats.linregress(x[idx], y[idx])
                Nmap[j,i] = m            

print(' -> Done!')    


## ---- Plot map ---- ##
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='i')
levels = np.linspace(-.0005, .0005, 9)
#levels = 20
xi, yi = m(*np.meshgrid(lon_reg, lat_reg))
#lon_casts, lat_casts = m(lons[idx], lats[idx])
#c = m.contourf(xi, yi, Nmap, levels, cmap=plt.cm.RdBu_r, extend='both')
c = m.pcolor(xi, yi, Nmap, cmap=plt.cm.RdBu_r, vmin=levels.min(), vmax=levels.max())
x,y = m(*np.meshgrid(lon,lat))
cc = m.contour(x, y, -Zbathy, [100, 500, 1000, 4000], colors='grey');
m.fillcontinents(color='tan');

m.drawparallels([40, 45, 50, 55, 60], labels=[1,0,0,0], fontsize=12, fontweight='normal');
m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');
#plt.title(r'Trends in N $\rm (s^{-1}) yr^{-1}$')
plt.title(r'$\rm \frac{dN}{dt} (s^{-1} yr^{-1})$')

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
#cb = plt.colorbar(c, cax=cax, ticks=levels)
cb = plt.colorbar(c, cax=cax)
#cb.set_label(r'$\rm N(s^{-1})$', fontsize=12, fontweight='normal')

fig.set_size_inches(w=7, h=9)
#fig.tight_layout() 
fig.set_dpi(200)
fig.savefig('map_Nmap.png')



# Salinity
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='i')
levels = np.linspace(-.05, .05, 11)
#levels = 20
xi, yi = m(*np.meshgrid(lon_reg, lat_reg))
#lon_casts, lat_casts = m(lons[idx], lats[idx])
#c = m.contourf(xi, yi, Smap, levels, cmap=plt.cm.RdBu_r, extend='both')
c = m.pcolor(xi, yi, Smap, cmap=plt.cm.RdBu_r, vmin=levels.min(), vmax=levels.max())
x,y = m(*np.meshgrid(lon,lat))
cc = m.contour(x, y, -Zbathy, [100, 500, 1000, 4000], colors='grey');
m.fillcontinents(color='tan');

m.drawparallels([40, 45, 50, 55, 60], labels=[1,0,0,0], fontsize=12, fontweight='normal');
m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');
plt.title(r'$\rm \frac{dS}{dt} (yr^{-1})$')

cax = plt.axes([0.85,0.15,0.04,0.7])
#cb = plt.colorbar(c, cax=cax, ticks=levels)
cb = plt.colorbar(c, cax=cax)
#cb.set_label(r'$\rm S_{trend} (yr^{-1})$', fontsize=12, fontweight='normal')

fig.set_size_inches(w=7, h=9)
#fig.tight_layout() 
fig.set_dpi(200)
fig.savefig('map_Smap.png')


# Temperature
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='i')
levels = np.linspace(-.4, .4, 9)
#levels = 20
xi, yi = m(*np.meshgrid(lon_reg, lat_reg))
#lon_casts, lat_casts = m(lons[idx], lats[idx])
#c = m.contourf(xi, yi, Tmap, levels, cmap=plt.cm.RdBu_r, extend='both')
c = m.pcolor(xi, yi, Tmap, cmap=plt.cm.RdBu_r, vmin=levels.min(), vmax=levels.max())
x,y = m(*np.meshgrid(lon,lat))
cc = m.contour(x, y, -Zbathy, [100, 500, 1000, 4000], colors='grey');
m.fillcontinents(color='tan');

m.drawparallels([40, 45, 50, 55, 60], labels=[1,0,0,0], fontsize=12, fontweight='normal');
m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');
plt.title(r'$\rm \frac{dT}{dt} (^{\circ}C yr^{-1})$')

cax = plt.axes([0.85,0.15,0.04,0.7])
#cb = plt.colorbar(c, cax=cax, ticks=levels)
cb = plt.colorbar(c, cax=cax)
#cb.set_label(r'$\rm T_{trend} (^{\circ}C yr^{-1})$', fontsize=12, fontweight='normal')

fig.set_size_inches(w=7, h=9)
#fig.tight_layout() 
fig.set_dpi(200)
fig.savefig('map_Tmap.png')


#convert -trim map_Tmap_spring_0 map_Tmap_spring_0

#montage map_Tmap_spring_0.5.png map_Smap_spring_0.5.png map_Nmap_spring_0.5.png map_Tmap_summer_0.5.png map_Smap_summer_0.5.png map_Nmap_summer_0.5.png map_Tmap_fall_0.5.png map_Smap_fall_0.5.png map_Nmap_fall_0.5.png -tile 3x3 -geometry +1+1  -background white  physical_trends_map.png

