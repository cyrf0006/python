# data source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=71800&vid=676

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
#import datetime
os.environ['PROJ_LIB'] = '/home/cyrf0006/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap

# Some parameters
#years = [1945, 1969]
#years = [1999, 2014]
years = [1948, 1987]
#years = [1987, 2020]
months = [1, 12] # months to keep

# For map limits
lllon = -100.
urlon = 10.
lllat = 0.
urlat = 90.

# For study area
lat1 = 56.830218
lat2 = 37.982170
lon1 =  -40.986870
lon2 = -60.704367
lon1 =  360-40.986870
lon2 = 360-60.704367


#v = np.arange(990, 1030) # SLP values
v = np.arange(995, 1025) # SLP values

# Load SLP data from NOAA ESRL
#ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mnmean.nc')
ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mon.mean.nc')
#ds = xr.open_mfdataset('/home/cyrf0006/data/NOAA_ESRL/slp.201*.nc')

# Selection of a subset region
#ds = ds.where((ds.lon>=-120) & (ds.lon<=30), drop=True) # original one
#ds = ds.where((ds.lat>=0) & (ds.lat<=90), drop=True)

da = ds['slp']
#p = da.to_pandas() # deprecated
p = da.to_dataframe() # deprecated
# Restrict to 2020
p = p[(p.index.get_level_values('time').year<=2020)]

# Compute climatology
#p_clim = p[(p.items.month>=months[0]) & (p.items.year<=months[1])]
#df_clim = p_clim.mean(axis=0)
#df_clim = p.mean(axis=0)
# Compute climatology
df_clim = p.groupby(level=[1,2]).mean().unstack()

# select years
p_year = p[(p.index.get_level_values('time').year>=years[0]) & (p.index.get_level_values('time').year<=years[1])]

# average all years
df = p_year.unstack()
df = df.groupby(level=['lat']).mean() 

fig_name = 'SLP_map_' + np.str(years[0]) + '-' + np.str(years[1]) + '.png'
fig_name2 = 'SLP_map_' + np.str(years[0]) + '-' + np.str(years[1]) + '.svg'
print(fig_name)
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)

m = Basemap(projection='ortho',lon_0=-40,lat_0=30, resolution='l', llcrnrx=-4000000, llcrnry=-2000000, urcrnrx=5000000, urcrnry=7000000)
m.drawcoastlines()
m.fillcontinents(color='tan')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,60.))
#m.drawmapboundary(fill_color='aqua')
plt.title("Sea Level Pressure - " + np.str(years[0]) + '-' + np.str(years[1]))

#x,y = m(*np.meshgrid(df.columns.values,df.index.values))
x,y = m(*np.meshgrid(df.columns.droplevel(None) ,df.index))
c = m.contourf(x, y, df.values, v, cmap=plt.cm.inferno, extend='both');
ct = m.contour(x, y, df_clim.values, 10, colors='k');
cb = plt.colorbar(c)
cb.set_label('SLP (mb)')

# set limits 
## xmin, ymin = map(lllon, lllat)
## xmax, ymax = map(urlon, urlat)
## ax = plt.gca()
## ax.set_xlim([xmin, xmax])
## ax.set_ylim([ymin, ymax])

#### ---- Save Figure ---- ####
#plt.suptitle('Fall surveys', fontsize=16)
fig.set_size_inches(w=8, h=6)
#fig.tight_layout() 
fig.set_dpi(200)
fig.savefig(fig_name)
fig.savefig(fig_name2, format='svg')

plt.close('all')

#### ---- Anomaly ---- ####
anom = df - df_clim
fig_name = 'anom_SLP_' + np.str(years[0]) + '-' + np.str(years[1]) + '.png'
fig_name2 = 'anom_SLP_' + np.str(years[0]) + '-' + np.str(years[1]) + '.svg'
print(fig_name)
plt.clf()
fig2, ax = plt.subplots(nrows=1, ncols=1)

m = Basemap(projection='ortho',lon_0=-40,lat_0=40, resolution='l', llcrnrx=-4000000, llcrnry=-2000000, urcrnrx=5000000, urcrnry=7000000)
m.drawcoastlines()
m.fillcontinents(color='tan')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,60.))
#m.drawmapboundary(fill_color='aqua')
    
#x,y = m(*np.meshgrid(df.columns.values,df.index.values))
x,y = m(*np.meshgrid(df.columns.droplevel(None) ,df.index))
#c = m.contourf(x, y, anom.values, np.linspace(-2.2, 2.2, 12), cmap=plt.cm.seismic, extend='both');
c = m.contourf(x, y, anom.values, np.linspace(-1.8, 1.8, 10), cmap=plt.cm.seismic, extend='both');
ct = m.contour(x, y, df_clim.values, 10, colors='k');
cb = plt.colorbar(c)
cb.set_label('SLP anomaly (mb)')
xBox, yBox = m([lon2, lon1, lon1, lon2, lon2], [lat2, lat2, lat1, lat1, lat2])
m.plot(xBox, yBox, '--k', linewidth=2)
plt.text(8400000, 12800000, np.str(years[0]) + '-' + np.str(years[1]), fontsize=16, fontweight='bold')

## # set limits 
## xmin, ymin = m(lllon, lllat)
## xmax, ymax = m(urlon, urlat)
## ax1 = plt.gca()
## ax1.set_xlim([xmin, xmax])
## ax1.set_ylim([ymin, ymax])

#### ---- Save Figure ---- ####
#plt.suptitle('Fall surveys', fontsize=16)
fig2.set_size_inches(w=8, h=6)
fig2.savefig(fig_name, dpi=150)
fig2.savefig(fig_name2, format='svg')
os.system('convert -trim ' + fig_name + ' ' + fig_name)
plt.close()
    
## if dt_year == 5:    
##     os.system('montage anom*.png  -tile 3x5 -geometry +10+10  -background white  montage_anom.png')
## elif dt_year == 10:
##     os.system('montage anom*.png  -tile 2x4 -geometry +10+10  -background white  montage_anom.png')

## plt.close('all')
os.system('montage anom_SLP_1948-1987.png anom_SLP_1987-2020.png -tile 1x2 -geometry +10+10  -background white  SLP_anom_petrels.png')
