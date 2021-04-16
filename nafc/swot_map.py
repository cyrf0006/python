
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
lonLims = [-56, -48]
latLims = [43, 52]
proj = 'merc'
decim_scale = 4
stationFile = '/home/cyrf0006/github/AZMP-NL/data//STANDARD_SECTIONS.xlsx'
fig_name = 'map_swot_zoom.png'
ephem = '/home/cyrf0006/AZMP/utils/ephem_calval.txt'
swot_kml = 'SWOT_Science_sept2015_Swath_10_60.kml'

## ---- Bathymetry ---- ####
v = np.linspace(-400, 0, 9)

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
m = Basemap(projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='h')
x,y = m(*np.meshgrid(lon,lat))
c = m.contourf(x, y, np.flipud(Z), v, cmap=plt.cm.PuBu_r, extend="min");
cc = m.contour(x, y, np.flipud(-Z), [50, 100, 200, 400], colors='lightgrey', linewidths=.5);
plt.clabel(cc, inline=1, fontsize=10, colors='gray', fmt='%d')
m.fillcontinents(color='grey');
m.drawparallels(np.arange(10,70,2), labels=[1,0,0,0], fontsize=12, fontweight='bold');
m.drawmeridians(np.arange(-80, 5, 2), labels=[0,0,0,1], fontsize=12, fontweight='bold');
#cb = plt.colorbar(c, orientation='horizontal')
## for l in cb.ax.yaxis.get_ticklabels():
##     l.set_weight("bold")
##     l.set_fontsize(10)
## cb.set_label('Depth (m)', fontsize=13, fontweight='bold')
plt.title("SWOT CalVal plan", fontsize=13, fontweight='bold')

# plot stations
x, y = m(stationLon[index_SEGB],stationLat[index_SEGB])
m.scatter(x,y,3,marker='o',color='r', zorder=10)
#plt.text(x[-1], y[-1], ' SEGB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
x, y = m(stationLon[index_FC],stationLat[index_FC])
m.scatter(x,y,3,marker='o',color='r', zorder=10)
#plt.text(x[-1], y[-1], ' FC', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
uctd3 = [x[5], y[5]]
x, y = m(stationLon[index_BB],stationLat[index_BB])
m.scatter(x,y,3,marker='o',color='r', zorder=10)
#plt.text(x[-1], y[-1], ' BB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
M1 = [x[2], y[2]]
M2 = [x[4], y[4]]
glid1 = [x[0], y[0]]
glid2 = [x[10], y[10]]
uctd2 = [x[3], y[3]]
x, y = m(stationLon[index_WB],stationLat[index_WB])
m.scatter(x,y,3,marker='o',color='r', zorder=10)
#plt.text(x[-1], y[-1], ' WB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
uctd1 = [x[7], y[7]]
x, y = m(stationLon[index_SI],stationLat[index_SI])
## m.scatter(x,y,3,marker='o',color='r', zorder=10)
## plt.text(x[-1], y[-1], ' SI', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_MB],stationLat[index_MB])
## m.scatter(x,y,3,marker='o',color='r', zorder=10)
## plt.text(x[-1], y[-1], ' MB', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_BI],stationLat[index_BI])
## m.scatter(x,y,3,marker='o',color='r', zorder=10)
#plt.text(x[-1], y[-1], ' BI', horizontalalignment='left', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_FI],stationLat[index_FI])
## m.scatter(x,y,3,marker='o',color='lightcoral')
## plt.text(x[-1], y[-1], ' FI', horizontalalignment='left', verticalalignment='center', fontsize=10, color='lightcoral', fontweight='bold')
## x, y = m(stationLon[index_S27],stationLat[index_S27])
## m.scatter(x,y,3,marker='o',color='lightcoral')
## plt.text(x[-1], y[-1], ' S27', horizontalalignment='left', verticalalignment='center', fontsize=10, color='lightcoral', fontweight='bold')
x, y = m(stationLon[index_SESPB],stationLat[index_SESPB])
m.scatter(x,y,3,marker='o',color='r')
#plt.text(x[-1], y[-1], 'SESPB ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
#x, y = m(stationLon[index_SWSPB],stationLat[index_SWSPB])
#m.scatter(x,y,3,marker='o',color='r')
#plt.text(x[-1], y[-1], 'SWSPB ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
## x, y = m(stationLon[index_SS],stationLat[index_SS])
## m.scatter(x,y,3,marker='o',color='lightcoral')
## plt.text(x[0], y[0], 'SS ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='lightcoral', fontweight='bold')


M3 = m(-52.7, 49.5)
x, y = m(stationLon[index_S27],stationLat[index_S27])
m.scatter(x[0],y[0],60,marker='*',color='r', zorder=10)
plt.text(x[0], y[0], 'Stn-27  ', horizontalalignment='right', verticalalignment='center', fontsize=10, color='r', fontweight='bold')
m.scatter(M1[0],M1[1], 60, marker='*',color='m', zorder=10)
m.scatter(M2[0],M2[1], 60, marker='*',color='m', zorder=10)
m.scatter(M3[0],M3[1], 60, marker='*',color='m', zorder=10)

#x_glider, y_glider = m([glid1[0], glid2[0]], [glid1[1], glid2[1]])
#x_uctd, y_uctd = m([uctd1[0], uctd2[0], uctd3[0]], [uctd1[1], uctd2[1], uctd3[1]])
m.plot([glid1[0], glid2[0]], [glid1[1], glid2[1]], 'y', alpha=0.8, linewidth=5)   
m.plot([uctd1[0], uctd2[0], uctd3[0]], [uctd1[1], uctd2[1], uctd3[1]], 'k', alpha=1, linewidth=5)   


## PLot swot stuff

def createCircleAroundWithRadius(lat, lon, radius):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    latArray = []
    lonArray = []

    for brng in range(0,360):
        lat2, lon2 = getLocation(lat,lon,brng,radius)
        latArray.append(lat2)
        lonArray.append(lon2)
        
    return lonArray,latArray


def getLocation(lat1, lon1, brng, distance):
    lat1 = lat1 * math.pi/ 180.0
    lon1 = lon1 * math.pi / 180.0
    #earth radius
    R = 6378.1 #Km
    #R = ~ 3959 MilesR = 3959

    distance = distance/R
    
    brng = (brng / 90.0)* math.pi / 2

    lat2 = math.asin(math.sin(lat1) * math.cos(distance) + math.cos(lat1) * math.sin(distance) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(distance)*math.cos(lat1),math.cos(distance)-math.sin(lat1)*math.sin(lat2))
    
    lon2 = 180.0 * lon2/ math.pi
    lat2 = 180.0 * lat2/ math.pi

    return lat2, lon2

## def draw_screen_poly( lats, lons, m):
##     x, y = m( lons, lats )
##     xy = zip(x,y)
##     poly = Polygon(xy, facecolor='green', alpha=0.6)
##     plt.gca().add_patch(poly)
    

## # Plot swot path (comment if not wanted)
for i in range(0,len(swot_segment_lat)):
    x_swot, y_swot = m(swot_segment_lon[i], swot_segment_lat[i])
    m.plot(x_swot, y_swot, 'green', alpha=0.4, linewidth=60)   

    # For ellipse around each pt
    ## for j in range(0,len(swot_segment_lat[i])):
    ##     X,Y = createCircleAroundWithRadius(swot_segment_lon[i][j], swot_segment_lat[i][j], 60)
    ##     draw_screen_poly(X, Y, m )

## crossover_x = [-62.335, -61.11, -59.967, -61.11, -62.335]
## crossover_y = [61.6, 62.897, 61.6, 60.302, 61.6]
## x, y = m(crossover_x, crossover_y)
## #xy = zip(x,y)
## xy = np.array(list(zip(x,y)))
## poly = Polygon(xy, facecolor='green', alpha=0.4)
## plt.gca().add_patch(poly)
    
#### ---- Save Figure ---- ####
fig.set_size_inches(w=8, h=9)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


