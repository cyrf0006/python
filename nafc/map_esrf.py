'''
A temptative map for ATACOM proposal'''


import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import pandas as pd
import xarray as xr
import cmocean
import os
## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
AZMP_station_file = '/home/cyrf0006/github/AZMP-NL/data/STANDARD_SECTIONS.xlsx'
AZMP_fixedstation_file = '/home/cyrf0006/research/proposals/atacom/AZMP_fixed_stations.xls'
AZMP_carbonate_file = '/home/cyrf0006/github/AZMP-NL/data/HUDSON2014_surveys_with_ancillary_data_31Aug2015.xlsx'
lon_0 = -50
lat_0 = 50
lonLims = [-70, -40]
latLims = [40, 65]
#lonLims = [-80, -30]
#latLims = [30, 61]
proj= 'merc'
decim_scale = 10
fig_name = 'map_azmp_esrf.png'

## ---- Bathymetry ---- ####
#v = np.linspace(-3500, 0, 36)
v = np.linspace(0, 3500, 36)


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

## ---- AZMP Station info ---- ##
df = pd.read_excel(AZMP_station_file)
# delete some sections
indexNames = df[(df.SECTION == 'STATION 27') | (df.SECTION == 'SMITH SOUND') | (df.SECTION == 'FUNK ISLAND') | (df.SECTION == 'RYANS BAY') | (df.SECTION == 'HUDSON STRAIT') | (df.SECTION == 'FROBISHER BAY')].index
df.drop(indexNames , inplace=True)
df.reset_index(inplace=True) 
azmp_stationLat = df['LAT'].values
azmp_stationLon = df['LONG.1'].values

index_SEGB = df.SECTION[df.SECTION=="SOUTHEAST GRAND BANK"].index.tolist()
index_FC = df.SECTION[df.SECTION=="FLEMISH CAP"].index.tolist()
index_BB = df.SECTION[df.SECTION=="BONAVISTA"].index.tolist()
index_WB = df.SECTION[df.SECTION=="WHITE BAY"].index.tolist()
index_SI = df.SECTION[df.SECTION=="SEAL ISLAND"].index.tolist()
index_MB = df.SECTION[df.SECTION=="MAKKOVIK BANK"].index.tolist()
index_BI = df.SECTION[df.SECTION=="BEACH ISLAND"].index.tolist()
index_SESPB = df.SECTION[df.SECTION=="SOUTHEAST ST PIERRE BANK"].index.tolist()
index_SWSPB = df.SECTION[df.SECTION=="SOUTHWEST ST PIERRE BANK"].index.tolist()
index_CS = df.SECTION[df.SECTION=="CABOT STRAIT"].index.tolist()
index_HLX = df.SECTION[df.SECTION=="HALIFAX"].index.tolist()
index_LB = df.SECTION[df.SECTION=="LOUISBOURG"].index.tolist()
index_BrB = df.SECTION[df.SECTION=="BROWNS BANK"].index.tolist()




## ---- AZMP Carbonates data  ---- ##
df = pd.read_excel(AZMP_carbonate_file)
cc_stationLat = df['Latitude'].values
cc_stationLon = df['Longitude'].values

## ---- AZMP Fixed Stations  ---- ##
df = pd.read_excel(AZMP_fixedstation_file)
fix_stationLat = df['latitude'].values
fix_stationLon = -df['longitude'].values

## ---- plot ---- ##
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='l')

# add AZMP
x, y = m(fix_stationLon,fix_stationLat)
m.scatter(x,y,20,marker='o',color='r', zorder=20)
x, y = m(azmp_stationLon,azmp_stationLat)
m.scatter(x,y,5,marker='o',color='orange', zorder=10)
x, y = m(cc_stationLon,cc_stationLat)
plt.legend(['AZMP buoys & fixed stations ', 'AZMP hydrographic sections'])
x, y = m(cc_stationLon,cc_stationLat)
m.scatter(x,y,5,marker='o',color='orange', zorder=10)


# plot stations
x, y = m(azmp_stationLon[index_SEGB],azmp_stationLat[index_SEGB])
#m.scatter(x,y,3,marker='o',color='orange')
plt.text(x[-1], y[-1], ' SEGB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_FC],azmp_stationLat[index_FC])
plt.text(x[-1], y[-1], ' FC', horizontalalignment='left', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_BB],azmp_stationLat[index_BB])
plt.text(x[-1], y[-1], ' BB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_WB],azmp_stationLat[index_WB])
plt.text(x[-1], y[-1], ' WB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_SI],azmp_stationLat[index_SI])
plt.text(x[-1], y[-1], ' SI', horizontalalignment='left', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_MB],azmp_stationLat[index_MB])
plt.text(x[-1], y[-1], ' MB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_BI],azmp_stationLat[index_BI])
plt.text(x[-1], y[-1], ' BI', horizontalalignment='left', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_SESPB],azmp_stationLat[index_SESPB])
plt.text(x[-1], y[-1], 'SESPB ', horizontalalignment='center', verticalalignment='top', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_SWSPB],azmp_stationLat[index_SWSPB])
plt.text(x[-1], y[-1], 'SWSPB ', horizontalalignment='center', verticalalignment='top', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_CS],azmp_stationLat[index_CS])
plt.text(x[-1], y[-1], 'CS ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_HLX],azmp_stationLat[index_HLX])
plt.text(x[-1], y[-1], 'HLX ', horizontalalignment='right', verticalalignment='top', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_LB],azmp_stationLat[index_LB])
plt.text(x[-1], y[-1], 'LB ', horizontalalignment='right', verticalalignment='top', fontsize=10, color='orange', fontweight='bold')
x, y = m(azmp_stationLon[index_BrB],azmp_stationLat[index_BrB])
plt.text(x[-1], y[-1], 'BrB ', horizontalalignment='right', verticalalignment='top', fontsize=10, color='orange', fontweight='bold')



x,y = m(*np.meshgrid(lon,lat))
c = m.contourf(x, y, np.flipud(-Z), v, cmap=cmocean.cm.deep, extend="max", alpha=.9);
cc = m.contour(x, y, np.flipud(-Z), [100, 500, 1000, 3000, 4000], colors='lightgrey', linewidths=.5);
plt.clabel(cc, inline=1, fontsize=10, colors='gray', fmt='%d')
ccc = m.contour(x, y, np.flipud(-Z), [0], colors='black');
#c = m.contourf(x, y, np.flipud(Z), v, cmap=plt.cm.PuBu_r, extend="min");
m.fillcontinents(color='peru', alpha=.8);
m.drawparallels(np.arange(10,70,10), labels=[1,0,0,0], fontsize=12, fontweight='bold');
m.drawmeridians(np.arange(-80, 5, 10), labels=[0,0,0,1], fontsize=12, fontweight='bold');
plt.title('AZMP hydrographic stations fixed observatories')


#### ---- Save Figure ---- ####
fig.set_size_inches(w=9, h=9)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

