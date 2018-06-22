'''
This is a test with xarray, see if I can manipulate a lot of netCDF files.
Try this script in /home/cyrf0006/research/AZMP_database
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata # here should remove nans or empty profiles
import netCDF4

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)


# This is a dataset
ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/*.nc')


# Select a depth range
## ds = ds.sel(level=ds['level']<500)
## ds = ds.sel(level=ds['level']>10)

## # Selection of a subset region
#ds = ds.where((ds.longitude>-58) & (ds.longitude<-46), drop=True)
## #ds = ds.where((ds.latitude>50) & (ds.latitude<55), drop=True)
#ds = ds.where((ds.latitude>42) & (ds.latitude<56), drop=True)

# Select only one year
#ds = ds.sel(time=ds['time.year']>1984)
ds = ds.sel(time=ds['time.year']>=1980)
ds = ds.sel(time=ds['time.year']<=1998)
year_unique = np.unique(ds['time.year'])
no_years = np.float(np.size(year_unique))

# Select only summer
#ds = ds.sel(time=ds['time.season']=='JJA')


## --- Caculate no of pts --- ##
lons = np.array(ds.longitude)
lats = np.array(ds.latitude)

# Meshgrid lat/lon
dc = 0.5
x = np.arange(np.min(lons), np.max(lons), dc)
y = np.arange(np.min(lats), np.max(lats), dc)  
lon, lat = np.meshgrid(x,y)
density = np.full(lon.shape, np.nan)

for ix, ll in enumerate(x):
    for iy, lt in enumerate(y):
        idx = np.where((lons>=ll-dc/2) & (lons<ll+dc/2) & (lats>=lt-dc/2) & (lats<lt+dc/2))
        if idx[0].size > 0:            
            density[iy, ix] = idx[0].size

 
## ---- Bathymetry ---- ####
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
decim_scale = 4
v = np.linspace(0, 4000, 9)

# Load data
dataset = netCDF4.Dataset(dataFile)

# Extract variables
x = dataset.variables['x_range']
y = dataset.variables['y_range']
spacing = dataset.variables['spacing']

# Compute Lat/Lon
nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir

lonz = np.linspace(x[0],x[-1],nx)
latz = np.linspace(y[0],y[-1],ny)

# Reshape data
zz = dataset.variables['z']
Z = zz[:].reshape(ny, nx)

# Reduce data according to Region params
lonz = lonz[::decim_scale]
latz = latz[::decim_scale]
Z = Z[::decim_scale, ::decim_scale]


## ---- Plot ---- ## 
lon_0 = lons.mean()
lat_0 = lats.mean()
lon_0 = -50
lat_0 = 50
#lonLims = [-58, -46]
#latLims = [42, 56]
lonLims = [-70, -40]
latLims = [40, 65]

fig = plt.figure()
proj = 'merc'
m = Basemap(projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='l')

x,y = m(*np.meshgrid(lonz,latz))
c = m.contour(x, y, np.flipud(-Z), v, colors='darkgrey');

#c = m.contourf(x, y, np.flipud(Z), v, cmap=plt.cm.PuRd_r, extend="min");
m.fillcontinents(color='grey');

#v = np.arange(np.floor(np.min(tmax)), np.ceil(np.max(tmax))+1)
xi, yi = m(lon, lat)
lon_casts, lat_casts = m(lons[idx], lats[idx])
#cs = m.scatter(xi, yi, c=np.log10(density), alpha=1, s=10, cmap=plt.cm.YlOrRd)
#cs = m.scatter(xi, yi, c=np.log10(density), alpha=1, s=10, cmap=plt.cm.OrRd, vmin=0, vmax=4)
cs = m.pcolor(xi, yi, np.log10(density/no_years), vmin=0, vmax=2)

m.fillcontinents(color='grey');

# Add Colorbar
cbar = m.colorbar(cs, location='right')
#cbar.set_label(r'$\rm log_{10}(no. observ.)$')
cbar.set_label(r'Average no. of casts per year', fontsize=15, fontweight='bold')
cbar.set_ticks([0, .5, 1, 1.5, 2])
cbar.set_ticklabels(['0', '3', '10', '30', '100']) 


# Add Grid Lines
m.drawparallels(np.arange(latLims[0], latLims[1], 5.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(lonLims[0], lonLims[1], 5.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Title
#plt.title('Number of observations (1950-1980)', fontsize=16, fontweight='bold')
plt.title('Number of observations (1980-1998)', fontsize=16, fontweight='bold')
#plt.title('Density of observations (1998-2017)', fontsize=16, fontweight='bold')

#plt.show()

fig.set_size_inches(w=9,h=9)
fig_name = 'map_obs_density.png'
fig.set_dpi(300)
fig.savefig(fig_name)
