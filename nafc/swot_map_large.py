
import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint

import math
import matplotlib.cm as cmx
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import os

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-70, -40]
latLims = [40, 65]
proj = 'merc'
decim_scale = 4
stationFile = '/home/cyrf0006/github/AZMP-NL/data//STANDARD_SECTIONS.xlsx'
fig_name = 'map_swot_large.png'
ephem = '/home/cyrf0006/AZMP/utils/ephem_calval.txt'
swot_kml = 'SWOT_Science_sept2015_Swath_10_60.kml'

## ---- Bathymetry ---- ####
v = np.linspace(-4000, 0, 9)

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

## ---- Station info ---- ##
import pandas as pd
df = pd.read_excel(stationFile)
#print the column names
#print(df.columns)
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

## ---- Ephemerides ---- ##
eph = np.genfromtxt(ephem, dtype=str)
swot_lon = eph[:,1].astype(np.float)-180.0
swot_lat = eph[:,2].astype(np.float)

asign = np.sign(swot_lon)
signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
signchange = np.squeeze(np.array(np.where(signchange==1), int))

swot_segment_lat = []
swot_segment_lon = []
idx_end = 0
for g in signchange:
    idx_beg = idx_end+1
    idx_end = g
    swot_segment_lat.append(swot_lat[idx_beg:idx_end])
    swot_segment_lon.append(swot_lon[idx_beg:idx_end])
swot_segment_lat.append(swot_lat[idx_end:-1])
swot_segment_lon.append(swot_lon[idx_end:-1])
    
## ---- plot ---- ##
fig = plt.figure(1)
#m = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution=None)
m = Basemap(projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='l')
x,y = m(*np.meshgrid(lon,lat))
c = m.contourf(x, y, np.flipud(Z), v, cmap=plt.cm.PuBu_r, extend="min");
cc = m.contour(x, y, np.flipud(-Z), [100, 500, 1000, 3000, 4000], colors='lightgrey', linewidths=.5);
plt.clabel(cc, inline=1, fontsize=10, colors='gray', fmt='%d')
#c = m.contourf(x, y, np.flipud(Z), v, cmap=plt.cm.PuRd_r, extend="min");
m.fillcontinents(color='gray');
m.drawparallels(np.arange(10,70,10), labels=[1,0,0,0], fontsize=12, fontweight='bold');
m.drawmeridians(np.arange(-80, 5, 10), labels=[0,0,0,1], fontsize=12, fontweight='bold');
m.drawstates()
#cb = plt.colorbar(c, orientation='horizontal')
## for l in cb.ax.yaxis.get_ticklabels():
##     l.set_weight("bold")
##     l.set_fontsize(10)
## cb.set_label('Depth (m)', fontsize=13, fontweight='bold')
plt.title("SWOT & NL shelf circulation", fontsize=13, fontweight='bold')

# plot stations
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
## x, y = m(stationLon[index_S27],stationLat[index_S27])
## m.scatter(x,y,3,marker='o',color='lightcoral')
## plt.text(x[-1], y[-1], ' S27', horizontalalignment='left', verticalalignment='center', fontsize=10, color='lightcoral', fontweight='bold')
x, y = m(stationLon[index_SESPB],stationLat[index_SESPB])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], 'SESPB ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_SWSPB],stationLat[index_SWSPB])
m.scatter(x,y,3,marker='o',color='r')
plt.text(x[-1], y[-1], 'SWSPB ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_SS],stationLat[index_SS])
## m.scatter(x,y,3,marker='o',color='lightcoral')
## plt.text(x[0], y[0], 'SS ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='lightcoral', fontweight='bold')

x, y = m(stationLon[index_S27],stationLat[index_S27])
m.scatter(x[0],y[0],40,marker='*',color='r')
plt.text(x[0], y[0], '  Stn-27', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')



## # Plot swot path (comment if not wanted)
for i in range(0,len(swot_segment_lat)):
    x_swot, y_swot = m(swot_segment_lon[i], swot_segment_lat[i])
    m.plot(x_swot, y_swot, color='green', alpha=.4, linewidth=20)   
    
# Draw zoomed box
box_x = [-56, -48, -48, -56, -56]
box_y = [52, 52, 43, 43, 52]
x, y = m(box_x, box_y)
m.plot(x, y, '--k', alpha=.9, linewidth=3)   

## # Plot Gulf Stream
## x_text, y_text = m(-65,40)
## x_gs, y_gs = m(-57, 43)
## plt.annotate("", xy=(x_gs, y_gs), xytext=(x_text, y_text), size=20, arrowprops=dict(arrowstyle="fancy",facecolor='orange'))
## x, y = m(-60, 41)
## plt.text(x, y, 'Gulf Stream', horizontalalignment='left', verticalalignment='center', fontsize=10, color='orange', fontweight='bold')

## # Plot Lab Current
## x_text, y_text = m(-55,56)
## x_gs, y_gs = m(-49, 52)
## plt.annotate("", xy=(x_gs, y_gs), xytext=(x_text, y_text), size=30, arrowprops=dict(arrowstyle="fancy",facecolor='cornflowerblue'))
## x, y = m(-50, 55)
## plt.text(x, y, 'Lab. Current', horizontalalignment='left', verticalalignment='center', fontsize=10, color='cornflowerblue', fontweight='bold')

# Plot Inshore Lab Current
x_text, y_text = m(-63,60)
x_gs, y_gs = m(-61, 58)
plt.annotate("", xy=(x_gs, y_gs), xytext=(x_text, y_text), size=20, arrowprops=dict(arrowstyle="fancy",facecolor='cornflowerblue'))

x_text, y_text = m(-60,57)
x_gs, y_gs = m(-57, 55)
plt.annotate("", xy=(x_gs, y_gs), xytext=(x_text, y_text), size=20, arrowprops=dict(arrowstyle="fancy",facecolor='cornflowerblue'))

x_text, y_text = m(-55.5,54)
x_gs, y_gs = m(-54, 52)
plt.annotate("", xy=(x_gs, y_gs), xytext=(x_text, y_text), size=20, arrowprops=dict(arrowstyle="fancy",facecolor='cornflowerblue'))

x_text, y_text = m(-53.7,51)
x_gs, y_gs = m(-53, 49)
plt.annotate("", xy=(x_gs, y_gs), xytext=(x_text, y_text), size=20, arrowprops=dict(arrowstyle="fancy",facecolor='cornflowerblue'))

x_text, y_text = m(-51.5,48.7)
x_gs, y_gs = m(-49.5, 48.2)
plt.annotate("", xy=(x_gs, y_gs), xytext=(x_text, y_text), size=20, arrowprops=dict(arrowstyle="fancy",facecolor='cornflowerblue'))

x_text, y_text = m(-52.5,47)
x_gs, y_gs = m(-53, 46)
plt.annotate("", xy=(x_gs, y_gs), xytext=(x_text, y_text), size=20, arrowprops=dict(arrowstyle="fancy",facecolor='cornflowerblue'))

x, y = m(-54, 53)
plt.text(x, y, 'Inshore Lab. current', horizontalalignment='left', verticalalignment='center', fontsize=10, color='cornflowerblue', fontweight='bold')

x, y = m(-62, 52.5)
plt.text(x, y, 'Labrador', horizontalalignment='left', verticalalignment='center', fontsize=9, color='black')
x, y = m(-59, 48.5)
plt.text(x, y, 'Newfoundland', horizontalalignment='left', verticalalignment='center', fontsize=9, color='black')
x, y = m(-48, 63)
plt.text(x, y, 'Greenland', horizontalalignment='left', verticalalignment='center', fontsize=9, color='black')



#### ---- Save Figure ---- ####
fig.set_size_inches(w=8, h=9)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

os.system('montage  map_swot_large.png map_swot_zoom.png -tile 2x1 -geometry +10+10  -background white  map_swot_subplot.png') 

