'''
This script generates the map used in CSAS ResDoc.
It is similar to another script used for NAFO SCR doc: nafo_map.py

Check this for nafo boxes:
/home/cyrf0006/AZMP/state_reports/2018/nafo.html

Frederic.Cyr@dfo-mpo.gc.ca
June 2019
'''


import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import pandas as pd
import os
import cmocean
import azmp_utils as azu

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
#lonLims = [-60, -40]
#latLims = [38, 56]
lonLims = [-70, -40]
latLims = [38, 65]
proj = 'merc'
decim_scale = 4
stationFile = '/home/cyrf0006/github/AZMP-NL/data/STANDARD_SECTIONS.xlsx'
fig_name = 'map_csas.png'
AZMP_airTemp_file = '/home/cyrf0006/github/AZMP-NL/utils/airTemp_sites.xlsx'
lightdeep = cmocean.tools.lighten(cmocean.cm.deep, .9)

## ---- Load SST boxes ---- ##
df_box = pd.read_excel('/home/cyrf0006/github/AZMP-NL/utils/SST_boxes.xslx')
#df_box = df_box[df_box.region=='NL']
df_box = df_box[(df_box.region=='NL')]

## ---- Bathymetry ---- ####
v = np.linspace(0, 3500, 36)
#v = np.linspace(-4000, 0, 9)
v = np.linspace(0, 5400, 55) # Olivia's
#v = np.append([0,25,50], np.linspace(100, 3500, 35))


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
print df.columns
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
## index_SS = df.SECTION[df.SECTION=="SMITH SOUND"].index.tolist()
## index_CSL = df.SECTION[df.SECTION=="CABOT STRAIT"].index.tolist()
## index_LL = df.SECTION[df.SECTION=="LOUISBOURG"].index.tolist()
## index_HL = df.SECTION[df.SECTION=="HALIFAX"].index.tolist()
## index_BBL = df.SECTION[df.SECTION=="BROWNS BANK"].index.tolist()

## ---- plot ---- ##
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='h')
x,y = m(*np.meshgrid(lon,lat))
c = plt.contourf(x, y, np.flipud(-Z), v, cmap=lightdeep, extend='max', zorder=1)
cc = plt.contour(x, y, np.flipud(-Z), [100, 500, 1000, 2000, 3000, 4000, 5000], colors='lightgrey', linewidths=.5, zorder=1)
plt.clabel(cc, inline=1, fontsize=10, colors='k', fmt='%d', zorder=1 )
#plt.clabel(cc, inline=True, fontsize=8, fmt='%d', zorder=100)
#c = m.contourf(x, y, np.flipud(-Z), v, cmap=cmocean.cm.deep, extend="max");
#cc = m.contour(x, y, np.flipud(-Z), [100, 500, 1000, 3000, 4000], colors='lightgrey', linewidths=.5);
#plt.clabel(cc, inline=1, fontsize=10, colors='gray', fmt='%d')
#ccc = m.contour(x, y, np.flipud(-Z), [0], colors='black');

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
## x, y = m(stationLon[index_FI],stationLat[index_FI])
## m.scatter(x,y,3,marker='o',color='lightcoral')
## plt.text(x[-1], y[-1], ' FI', horizontalalignment='left', verticalalignment='center', fontsize=10, color='lightcoral', fontweight='bold')
x, y = m(stationLon[index_S27],stationLat[index_S27])
# m.scatter(x,y,3,marker='o',color='lightcoral')
m.scatter(x[0],y[0],100,marker='*',color='r', zorder=10)
plt.text(x[0], y[0], ' S27', horizontalalignment='left', verticalalignment='bottom', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_SESPB],stationLat[index_SESPB])
m.scatter(x,y,3,marker='o',color='r', zorder=20)
plt.text(x[-1], y[-1], 'SESPB ', horizontalalignment='left', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_SWSPB],stationLat[index_SWSPB])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], 'SWSPB ', horizontalalignment='center', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_SS],stationLat[index_SS])
## m.scatter(x,y,3,marker='o',color='lightcoral')
## plt.text(x[0], y[0], 'SS ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='lightcoral', fontweight='bold')
## x, y = m(stationLon[index_CSL],stationLat[index_CSL])
## m.scatter(x,y,3,marker='o',color='r')
## plt.text(x[-1], y[-1], 'CSL ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_LL],stationLat[index_LL])
## m.scatter(x,y,3,marker='o',color='r')
## plt.text(x[-1], y[-1], ' LL', horizontalalignment='left', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_HL],stationLat[index_HL])
## m.scatter(x,y,3,marker='o',color='r')
## plt.text(x[-1], y[-1], ' HL', horizontalalignment='left', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_BBL],stationLat[index_BBL])
## m.scatter(x,y,3,marker='o',color='r')
## plt.text(x[-1], y[-1], ' BBL', horizontalalignment='left', verticalalignment='top', fontsize=10, color='r', fontweight='bold')
## x, y = m(-63.317, 44.267)
## m.scatter(x,y,100,marker='*',color='r', zorder=10)
## plt.text(x, y, 'S2  ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
## x, y = m(-66.85, 44.93)
## m.scatter(x,y,100,marker='*',color='r', zorder=10)
## plt.text(x, y, 'Prince 5  ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')


# plot SST_boxes
abbr = df_box.abbr.values
lat_min = df_box.lat_min.values
lat_max = df_box.lat_max.values
lon_min = df_box.lon_min.values
lon_max = df_box.lon_max.values
for idx, name in enumerate(abbr):
    xbox = np.array([lon_min[idx], lon_max[idx], lon_max[idx], lon_min[idx], lon_min[idx]])
    ybox = np.array([lat_min[idx], lat_min[idx], lat_max[idx], lat_max[idx], lat_min[idx]])
    x, y = m(xbox, ybox)
    if (name=='CLS'):
        m.plot(x,y,color='w')
        x, y = m(xbox, ybox+1.5)
        plt.text(x.mean(), y.mean(), name, horizontalalignment='center', verticalalignment='center', fontsize=10, color='w', fontweight='bold')
    elif (name=='GS') | (name=='OK') | (name=='NCLS') | (name=='NENL'):
        m.plot(x,y,color='dimgray')
        x, y = m(xbox, ybox+.5)
        plt.text(x.mean(), y.mean(), name, horizontalalignment='center', verticalalignment='center', fontsize=10, color='dimgray', fontweight='bold')
    elif (name=='HS'):
        m.plot(x,y,color='dimgray')
        x, y = m(xbox, ybox+.75)
        plt.text(x.mean(), y.mean(), name, horizontalalignment='center', verticalalignment='center', fontsize=10, color='dimgray', fontweight='bold')
    elif (name=='BRA'):
        m.plot(x,y,color='w')
        plt.text(x.mean(), y.mean(), name, horizontalalignment='center', verticalalignment='center', fontsize=10, color='w', fontweight='bold')
    else:
        m.plot(x,y,color='dimgray')
        plt.text(x.mean(), y.mean(), name, horizontalalignment='center', verticalalignment='center', fontsize=10, color='dimgray', fontweight='bold')

#    m.plot(x,y,color='dimgray')
#    plt.text(x.mean(), y.mean(), name, horizontalalignment='center', verticalalignment='center', fontsize=10, color='dimgray', fontweight='bold')
# add names


# add air temperature stations
x, y = m(air_stationLon,air_stationLat)
plt.text(x[0], y[0], '  St. John''s ', horizontalalignment='right', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[1], y[1], '  Bonavista ', horizontalalignment='right', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[2], y[2], '   Cartwright  ', horizontalalignment='right', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[3], y[3], '  Iqaluit  ', horizontalalignment='left', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
plt.text(x[4], y[4], ' Nuuk  ', horizontalalignment='left', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
## plt.text(x[5], y[5], ' Sydney ', horizontalalignment='right', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
## plt.text(x[6], y[6], 'Sable Is.  ', horizontalalignment='left', verticalalignment='top', fontsize=13, rotation=-45, color='saddlebrown', fontweight='bold', zorder=10)
## plt.text(x[7], y[7], ' Halifax  ', horizontalalignment='left', verticalalignment='center', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
## plt.text(x[8], y[8], ' St. John ', horizontalalignment='right', verticalalignment='bottom', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
## plt.text(x[9], y[9], ' Boston ', horizontalalignment='right', verticalalignment='bottom', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
## plt.text(x[10], y[10], ' Yarmouth ', horizontalalignment='right', verticalalignment='bottom', fontsize=13, color='saddlebrown', fontweight='bold', zorder=10)
m.scatter(x,y,20,marker='o',color='saddlebrown', zorder=10)

# Add NAFO divisions
div_toplot = ['2G', '2H', '2J', '3K', '3L', '3N', '3M', '3O', '3Ps']
for div in div_toplot:
    div_lon, div_lat = m(nafo_div[div]['lon'], nafo_div[div]['lat'])
    m.plot(div_lon, div_lat, 'black', linewidth=2)
    if (div == '3M') | (div == '4X'):
        ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold', zorder=100, verticalalignment='top', horizontalalignment='right')    
    elif (div == '3M'):
        ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold', zorder=100, verticalalignment='top', horizontalalignment='center')
    elif (div == '3Ps'):
        ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold', zorder=100, verticalalignment='bottom', horizontalalignment='right')
    else:
        ax.text(np.mean(div_lon), np.mean(div_lat), div, fontsize=12, color='black', fontweight='bold', zorder=100)    

#### ---- Save Figure ---- ####
fig.set_size_inches(w=10, h=12)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
#plt.show()

