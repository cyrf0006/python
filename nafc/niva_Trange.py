'''
To generate bottom climato:
import numpy as np
import azmp_utils as azu
dc = .1
lonLims = [-60, -45] # FC AZMP report region
latLims = [42, 56]
#lonLims = [-63, -45] # include 2H in above
#latLims = [42, 58]
lon_reg = np.arange(lonLims[0]+dc/2, lonLims[1]-dc/2, dc)
lat_reg = np.arange(latLims[0]+dc/2, latLims[1]-dc/2, dc)
azu.get_surfT_climato('/home/cyrf0006/data/dev_database/netCDF/*.nc', lon_reg, lat_reg, season='fall', year_lims=[1995, 2001], zlims=[60, 90], h5_outputfile='T60-90m_1995-2001_fall_0.10.h5') 


azu.get_cilT_climato('/home/cyrf0006/data/dev_database/netCDF/*.nc', lon_reg, lat_reg, season='fall', year_lims=[1995, 2018], zlims=[10, 500], h5_outputfile='TCIL_1995-2018_fall_0.10.h5') 


'''
import netCDF4
import h5py                                                                
import os
os.environ['PROJ_LIB'] = '/home/cyrf0006/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import cmocean
import cmocean.cm as cmo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cpf
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.ticker as mticker
import openpyxl, pprint
import shapefile 
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from bokeh.plotting import figure, save
import azmp_utils as azu


## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GEBCO_2014_1D.nc'
#lonLims = [-65, -40]
#latLims = [40, 60]
proj = 'merc'
decim_scale = 4
v = np.linspace(0, 3500, 36)
season='fall'
div_toplot = ['2J', '3K', '3L', '3N', '3O', '3Ps']
#div_toplot = ['2J', '3K', '3L', '3N', '3O']
period=0

## ---- Load Climato data ---- ##    
if period == 0:
    climato_file = '/home/cyrf0006/research/Niva/T60-90m_1995-2018_fall_0.10.h5'
elif period == 1:
    climato_file = '/home/cyrf0006/research/Niva/T60-90m_1995-2001_fall_0.10.h5'
elif period == 2:
    climato_file = '/home/cyrf0006/research/Niva/T60-90m_2002-2006_fall_0.10.h5'
elif period == 3:
    climato_file = '/home/cyrf0006/research/Niva/T60-90m_2007-2013_fall_0.10.h5'
elif period == 4:
    climato_file = '/home/cyrf0006/research/Niva/T60-90m_2014-2018_fall_0.10.h5'
    
h5f = h5py.File(climato_file, 'r')
Tsurf_climato = h5f['Tsurf'][:]
lon_reg = h5f['lon_reg'][:]
lat_reg = h5f['lat_reg'][:]
Zitp = h5f['Zitp'][:]
h5f.close()
lonLims = [lon_reg[0], lon_reg[-1]]
latLims = [lat_reg[0], lat_reg[-1]]
    
## ---- NAFO divisions ---- ##
nafo_div = azu.get_nafo_divisions()
       

## ---- Now plot ---- ##
print('--- Now plot ---')
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection=ccrs.Mercator())
ax.set_extent([lonLims[0], lonLims[1], latLims[0], latLims[1]], crs=ccrs.PlateCarree())
m=ax.gridlines(linewidth=0.5, color='black', draw_labels=True, alpha=0.5, zorder=1)
ax.add_feature(cpf.NaturalEarthFeature('physical', 'coastline', '50m', edgecolor='darkgoldenrod', alpha=1, linewidth=0.6, facecolor='darkgoldenrod'), zorder=5)
m.xlabels_top=False
m.ylabels_right=False
m.xlocator = mticker.FixedLocator([-60, -55, -50, -45])
m.ylocator = mticker.FixedLocator([40, 45, 50, 55, 60])
m.xformatter = LONGITUDE_FORMATTER
m.yformatter = LATITUDE_FORMATTER
m.ylabel_style = {'size': 7, 'color': 'black', 'weight':'bold'}
m.xlabel_style = {'size': 7, 'color': 'black', 'weight':'bold'}
lightdeep = cmocean.tools.lighten(cmo.deep, 0.5)
ls = np.linspace(0, 5500, 20)
levels = np.linspace(-2, 5, 15)
c = plt.contourf(lon_reg, lat_reg, Tsurf_climato, levels, transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r, extend='both', zorder=5)
c0 = plt.contour(lon_reg, lat_reg, Tsurf_climato, [0,], transform=ccrs.PlateCarree(), colors='k', zorder=5)
cc = plt.contour(lon_reg, lat_reg, -Zitp, [100, 300, 500, 1000, 2000, 3000, 4000], colors='silver', linewidths=1, transform=ccrs.PlateCarree(), zorder=10)
plt.clabel(cc, inline=True, fontsize=7, fmt='%i')
if period == 0:
    plt.title('Temperature (60-90m) - 1995-2018 average')
elif period == 1:
    plt.title('Temperature (60-90m) - 1995-2001 average')
elif period == 2:   
    plt.title('Temperature (60-90m) - 2002-2006 average')
elif period == 3:
    plt.title('Temperature (60-90m) - 2007-2013 average')
elif period == 4:    
    plt.title('Temperature (60-90m) - 2014-2018 average')
cax = fig.add_axes([0.16, 0.05, 0.7, 0.025])
cb = plt.colorbar(c, cax=cax, orientation='horizontal')
cb.set_label(r'$\rm T(^{\circ}C)$', fontsize=12, fontweight='normal')
   
# plot NAFO divisions
for div in div_toplot:
    ax.plot(nafo_div[div]['lon'], nafo_div[div]['lat'], 'dimgray', linewidth=2, zorder=20, transform=ccrs.PlateCarree())
    
# Save Figure
fig.set_size_inches(w=7, h=9)
if period == 0:
    outfile = 'Temp_60-90m_1995-2018_' + season + '.png'
elif period == 1:
    outfile = 'Temp_60-90m_1995-2001_' + season + '.png'
elif period == 2:
    outfile = 'Temp_60-90m_2002-2006_' + season + '.png'
elif period == 3:
    outfile = 'Temp_60-90m_2007-2013_' + season + '.png'
elif period == 4:
    outfile = 'Temp_60-90m_2014-2018_' + season + '.png'


fig.savefig(outfile, dpi=150)
os.system('convert -trim ' + outfile + ' ' + outfile)



