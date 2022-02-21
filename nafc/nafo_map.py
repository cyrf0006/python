'''
Script created to plot a map with SST boxes on it.

To extract the box lat/lon, I did this:
df = pd.read_csv('/home/cyrf0006/github/AZMP-NL/utils/SST_boxes_coordinates.csv', header=None)
# from ftp://ftp.dfo-mpo.gc.ca/bometrics/noaa/noaa-dat_stats.in.all (removed 1st line)
box_name = df.iloc[::2]
box_coord = df.iloc[1::2]
box_name = box_name.reset_index(drop=True)
box_coord = box_coord.reset_index(drop=True)
df = pd.concat([box_name, box_coord], axis=1)
df.to_csv('SST_boxes_coords.csv')
box_name.to_csv('SST_boxes_coords_name.csv')  # <---- Easier to play with
box_coord.to_csv('SST_boxes_coords_coords.csv')

Then I opened it and copy-pasted it in /home/cyrf0006/github/AZMP-NL/utils/SST_boxes.xslx

Check this for nafo boxes:
/home/cyrf0006/AZMP/state_reports/2018/nafo.html
'''

import os
import netCDF4
os.environ['PROJ_LIB'] = '/home/cyrf0006/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import pandas as pd
import os
import cmocean
import azmp_utils as azu

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GEBCO_2014_1D.nc'#GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
#lonLims = [-60, -40]
#latLims = [38, 56]
lonLims = [-75, -40]
latLims = [38, 65]
proj = 'merc'
decim_scale = 4
stationFile = '/home/cyrf0006/github/AZMP-NL/data/STANDARD_SECTIONS.xlsx'
fig_name = 'map_nafo.png'
AZMP_airTemp_file = '/home/cyrf0006/github/AZMP-NL/utils/airTemp_sites_nafo.xlsx'
lightdeep = cmocean.tools.lighten(cmocean.cm.deep, .9)
v = np.linspace(0, 3500, 36)
v = np.linspace(0, 5400, 55) # Olivia's

## ---- Load SST boxes ---- ##
df_box = pd.read_excel('/home/cyrf0006/github/AZMP-NL/utils/SST_boxes.xslx')
#df_box = df_box[df_box.region=='NL']
df_box = df_box[(df_box.region=='NL') | (df_box.region=='NS')]

## --------- Get Bathymetry -------- ####
bathy_file = 'nafo_map_bathy.npz'
if os.path.isfile(bathy_file):
    print('Load saved bathymetry!')
    npzfile = np.load(bathy_file)
    lat = npzfile['lat']  
    lon = npzfile['lon']  
    Z = npzfile['Z']

else:
    print('Get bathy...')
    dataFile = '/home/cyrf0006/data/GEBCO/GEBCO_2014_1D.nc' # Maybe find a better way to handle this file
    # Load data
    dataset = netCDF4.Dataset(dataFile)
    x = [-179-59.75/60, 179+59.75/60] # to correct bug in 30'' dataset?
    y = [-89-59.75/60, 89+59.75/60]
    spacing = dataset.variables['spacing']
    # Compute Lat/Lon
    nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
    ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir
    lon = np.linspace(x[0],x[-1],nx)
    lat = np.linspace(y[0],y[-1],ny)
    # interpolate data on regular grid 
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
    lon = lon[::decim_scale]
    lat = lat[::decim_scale]
    Z = Z[::decim_scale, ::decim_scale]
    np.savez(bathy_file, lat=lat, lon=lon, Z=Z)
    print(' -> Done!')

## ---- AZMP air temp Stations  ---- ##
df = pd.read_excel(AZMP_airTemp_file)
air_stationLat = df['latitude'].values
air_stationLon = df['longitude'].values

## ---- NAFO divisions ---- ##
nafo_div = azu.get_nafo_divisions()

## ---- Station info ---- ##
import pandas as pd
df = pd.read_excel(stationFile)
#print the column names
print(df.columns)
#get the values for a given column
sections = df['SECTION'].values
stations = df['STATION'].values
stationLat = df['LAT'].values
stationLon = df['LONG.1'].values

index_SEGB = df.SECTION[df.SECTION=="SOUTHEAST GRAND BANK"].index.tolist()
index_FC = df.SECTION[df.SECTION=="FLEMISH CAP"].index.tolist()
index_BB = df.SECTION[df.SECTION=="BONAVISTA"].index.tolist()
index_WB = df.SECTION[df.SECTION=="WHITE BAY"].index.tolist()
index_SI = df.SECTION[df.SECTION=="SEAL ISLAND"].index.tolist()
index_MB = df.SECTION[df.SECTION=="MAKKOVIK BANK"].index.tolist()
index_BI = df.SECTION[df.SECTION=="BEACH ISLAND"].index.tolist()
index_FI = df.SECTION[df.SECTION=="FUNK ISLAND"].index.tolist()
index_S27 = df.SECTION[df.SECTION=="STATION 27"].index.tolist()
index_SESPB = df.SECTION[df.SECTION=="SOUTHEAST ST PIERRE BANK"].index.tolist()
index_SWSPB = df.SECTION[df.SECTION=="SOUTHWEST ST PIERRE BANK"].index.tolist()
index_SS = df.SECTION[df.SECTION=="SMITH SOUND"].index.tolist()
index_CSL = df.SECTION[df.SECTION=="CABOT STRAIT"].index.tolist()
index_LL = df.SECTION[df.SECTION=="LOUISBOURG"].index.tolist()
index_HL = df.SECTION[df.SECTION=="HALIFAX"].index.tolist()
index_BBL = df.SECTION[df.SECTION=="BROWNS BANK"].index.tolist()

## ---- plot ---- ##
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='h')
x,y = m(*np.meshgrid(lon,lat))
c = plt.contourf(x, y, -Z, v, cmap=lightdeep, extend='max', zorder=1)
cc = plt.contour(x, y, -Z, [100, 500, 1000, 2000, 3000, 4000, 5000], colors='lightgrey', linewidths=.5, zorder=10)
plt.clabel(cc, inline=1, fontsize=10, colors='lightgray', fmt='%d')
ccc = m.contour(x, y, -Z, [1000], colors='k', linestyle='--', linewidths=1, zorder=12);

m.fillcontinents(color='tan');
m.drawparallels(np.arange(10,70,5), labels=[1,0,0,0], fontsize=12, fontweight='bold');
m.drawmeridians(np.arange(-80, 5, 5), labels=[0,0,0,1], fontsize=12, fontweight='bold');

# plot AZMP stations
x, y = m(stationLon[index_SEGB],stationLat[index_SEGB])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' SEGB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_FC],stationLat[index_FC])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' FC', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_BB],stationLat[index_BB])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' BB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_WB],stationLat[index_WB])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' WB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_SI],stationLat[index_SI])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' SI', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_MB],stationLat[index_MB])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' MB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_BI],stationLat[index_BI])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' BI', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_S27],stationLat[index_S27])
m.scatter(x[0],y[0],100,marker='*',color='r', zorder=10)
plt.text(x[0], y[0], ' S27', horizontalalignment='left', verticalalignment='bottom', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_SESPB],stationLat[index_SESPB])
m.scatter(x,y,3,marker='o',color='r', zorder=20)
plt.text(x[-1], y[-1], 'SESPB ', horizontalalignment='left', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_SWSPB],stationLat[index_SWSPB])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], 'SWSPB ', horizontalalignment='center', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
## m.scatter(x,y,3,marker='o',color='lightcoral')
## plt.text(x[0], y[0], 'SS ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='lightcoral', fontweight='bold')
x, y = m(stationLon[index_CSL],stationLat[index_CSL])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], 'CSL ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_LL],stationLat[index_LL])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' LL', horizontalalignment='left', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_HL],stationLat[index_HL])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' HL', horizontalalignment='left', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_BBL],stationLat[index_BBL])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], ' BBL', horizontalalignment='left', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
x, y = m(-63.317, 44.267)
m.scatter(x,y,100,marker='*',color='r', zorder=10)
plt.text(x, y, 'HLX-2  ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(-66.85, 44.93)
m.scatter(x,y,100,marker='*',color='r', zorder=10)
plt.text(x, y, 'Prince 5  ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
## # plot SST_boxes
## abbr = df_box.abbr.values
## lat_min = df_box.lat_min.values
## lat_max = df_box.lat_max.values
## lon_min = df_box.lon_min.values
## lon_max = df_box.lon_max.values

# add air temperature stations
x, y = m(air_stationLon,air_stationLat)
plt.text(x[0], y[0], '  St. John''s ', horizontalalignment='right', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[1], y[1], '  Bonavista ', horizontalalignment='right', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[2], y[2], '   Cartwright  ', horizontalalignment='right', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[3], y[3], '  Iqaluit  ', horizontalalignment='left', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[4], y[4], ' Nuuk  ', horizontalalignment='left', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[5], y[5], ' Sydney ', horizontalalignment='right', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[6], y[6], 'Sable Is.  ', horizontalalignment='left', verticalalignment='top', fontsize=13, rotation=-45, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[7], y[7], ' Halifax  ', horizontalalignment='left', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[8], y[8], ' St. John ', horizontalalignment='right', verticalalignment='bottom', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[9], y[9], ' Boston ', horizontalalignment='right', verticalalignment='bottom', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[10], y[10], ' Yarmouth ', horizontalalignment='right', verticalalignment='bottom', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
m.scatter(x,y,20,marker='o',color='saddlebrown', zorder=10)

# Add NAFO divisions
div_toplot = ['2G', '2H', '2J', '3K', '3L', '3N', '3M', '3O', '3Ps', '4Vn', '4Vs', '4W', '4X']
for div in div_toplot:
    div_lon, div_lat = m(nafo_div[div]['lon'], nafo_div[div]['lat'])
    m.plot(div_lon, div_lat, 'black', linewidth=2)
    if (div == '3M') | (div == '4X'):
        ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold', zorder=100, verticalalignment='top', horizontalalignment='right')    
    elif (div == '4Vn') | (div == '3Ps') | (div == '3M') | (div == '4X'):
        ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold', zorder=100, verticalalignment='top', horizontalalignment='center')
    else:
        ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold', zorder=100)    

#### ---- Save Figure ---- ####
fig.set_size_inches(w=10, h=12)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
#plt.show()

