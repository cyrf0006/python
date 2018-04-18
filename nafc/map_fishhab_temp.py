'''
Temperature maps based on historical trawl data. David Belanger ACCASP's project on ground fish.
'''

## plt.show()


import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import pandas as pd
from scipy.interpolate import griddata # here should remove nans or empty profiles

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/Data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-60, -44]
latLims = [39, 56]
proj = 'merc'
decim_scale = 4
#spring = True
spring = False


fig_name = 'map_trawls_temp.png'

## ---- Bathymetry ---- ####
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

# Reshape data
zz = dataset.variables['z']
Z = zz[:].reshape(ny, nx)

# Reduce data according to Region params
lon = lon[::decim_scale]
lat = lat[::decim_scale]
Z = Z[::decim_scale, ::decim_scale]


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


## ---- Get trawl data --- ##
df = pd.read_csv('empirical.df_18Jan2018.csv',delimiter=',', parse_dates={'Datetime': [1, 2, 3]})                
df.index = pd.to_datetime(df.Datetime)
df.drop(columns=['Datetime'])
#df_spring = df[(df.index.month>=4) & (df.index.month<=6)]
#df_fall = df[(df.index.month>=10) & (df.index.month<=12)]
df_spring = df[df.season=='Spring']
df_fall = df[df.season=='Fall']



## -----> Start loop <----- ##
edges = np.arange(1980, 2016, 5)
fig, axes = plt.subplots(nrows=2, ncols=4)
for idx in np.arange(edges.size-1):

        
    ## ---- Interpolate temperature ---- ##
    if spring:
        df_5yr = df_spring[(df_spring.index.year>=edges[idx]) & (df_spring.index.year<edges[idx+1])]
        v = np.arange(-1, 8)
    else:
        df_5yr = df_fall[(df_fall.index.year>=edges[idx]) & (df_fall.index.year<edges[idx+1])]
        v = np.arange(-1, 8)

    temp = np.array(df_5yr['temp'])
    lons = np.array(df_5yr['long'])
    lats = np.array(df_5yr['lat'])

    # Meshgrid 1D data (after removing NaNs)
    idx_nan = np.argwhere(~np.isnan(temp))
    x = np.arange(np.min(lons[idx_nan]), np.max(lons[idx_nan]), .2)
    y = np.arange(np.min(lats[idx_nan]), np.max(lats[idx_nan]), .2)  
    lon_temp, lat_temp = np.meshgrid(x,y)

    # griddata
    LN = np.squeeze(lons[idx_nan])
    LT = np.squeeze(lats[idx_nan])
    TT = np.squeeze(temp[idx_nan])
    temp_itp = griddata((LN, LT), TT, (lon_temp, lat_temp), method='linear')

    ## ---- plot ---- ##
    ax = axes.flat[idx]
    print idx
    m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='l')

    xi, yi = m(lon_temp, lat_temp)
    lon_casts, lat_casts = m(lons[idx], lats[idx])
    c = m.contourf(xi,yi,temp_itp, v, cmap=plt.cm.RdBu_r, extend='both')

    x,y = m(*np.meshgrid(lon,lat))
    cc = m.contour(x, y, np.flipud(-Z), [100, 500, 1000, 4000], colors='grey');
    m.fillcontinents(color='tan');

    if (idx==0) | (idx==4):
        m.drawparallels([40, 45, 50, 55], labels=[1,0,0,0], fontsize=12, fontweight='normal');
    else:
        m.drawparallels([40, 45, 50, 55], labels=[0,0,0,0], fontsize=12, fontweight='normal');

    m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');

    # title
    ax.set_title(np.str(edges[idx]) + '-' + np.str(edges[idx+1]))
    
    # Draw NAFO divisions
    x,y = m(np.array(div3K['lon']), np.array(div3K['lat']))
    m.plot(x,y,color='black')
    ax.text(np.mean(x), np.mean(y), '3K', fontsize=12, color='black', fontweight='bold')

    x,y = m(np.array(div3L['lon']), np.array(div3L['lat']))
    m.plot(x,y,color='black')
    ax.text(np.mean(x), np.mean(y), '3L', fontsize=12, color='black', fontweight='bold')

    x,y = m(np.array(div3N['lon']), np.array(div3N['lat']))
    m.plot(x,y,color='black')
    ax.text(np.mean(x), np.mean(y), '3N', fontsize=12, color='black', fontweight='bold')

    x,y = m(np.array(div3O['lon']), np.array(div3O['lat']))
    m.plot(x,y,color='black')
    ax.text(np.mean(x)*.9, np.mean(y), '3O', fontsize=12, color='black', fontweight='bold')

    x,y = m(np.array(div3Ps['lon']), np.array(div3Ps['lat']))
    m.plot(x,y,color='black')
    ax.text(np.mean(x)*.7, np.mean(y)*.95, '3Ps', fontsize=12, color='black', fontweight='bold')

    x,y = m(np.array(div2J['lon']), np.array(div2J['lat']))
    m.plot(x,y,color='black')
    ax.text(np.mean(x), np.mean(y), '2J', fontsize=12, color='black', fontweight='bold')


ax = axes.flat[7]
ax.axis('off')
cax = plt.axes([0.73,0.08,0.014,0.35])
cb = plt.colorbar(c, cax=cax, ticks=v)
cb.set_label(r'$\rm T(^{\circ}C)$', fontsize=12, fontweight='normal')
plt.subplots_adjust(left=.07, bottom=.07, right=.93, top=.9, wspace=.2, hspace=.2)

#### ---- Save Figure ---- ####
if spring:
    plt.suptitle('Spring surveys', fontsize=16)
else:
    plt.suptitle('Fall surveys', fontsize=16)
#plt.suptitle('Spring surveys', fontsize=16)
fig.set_size_inches(w=12, h=9)
#fig.tight_layout() 
fig.set_dpi(200)
fig.savefig(fig_name)


