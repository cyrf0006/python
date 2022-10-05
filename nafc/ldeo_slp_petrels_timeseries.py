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
#years = [1945, 1969]
#years = [1948, 1987]
#years = [1987, 2020]
lat1 = 56.830218
lat2 = 37.982170
lon1 =  360-40.986870
lon2 = 360-60.704367

months = [1, 12] # months to keep
#v = np.arange(990, 1030) # SLP values
v = np.arange(995, 1025) # SLP values

# Load SLP data from NOAA ESRL
#ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mnmean.nc')
ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mon.mean.nc')
#ds = xr.open_mfdataset('/home/cyrf0006/data/NOAA_ESRL/slp.201*.nc')

# Selection of a subset region
ds = ds.where((ds.lon>=lon2) & (ds.lon<=lon1), drop=True) # original one
ds = ds.where((ds.lat>=lat2) & (ds.lat<=lat1), drop=True)

da = ds['slp']
#p = da.to_pandas() # deprecated
p = da.to_dataframe() # deprecated


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
print(fig_name)
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)

m = Basemap(projection='ortho',lon_0=-40,lat_0=40, resolution='l')
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

#### ---- Save Figure ---- ####
#plt.suptitle('Fall surveys', fontsize=16)
fig.set_size_inches(w=8, h=6)
#fig.tight_layout() 
fig.set_dpi(200)
fig.savefig(fig_name)


#### ---- Anomaly ---- ####
anom = df - df_clim
fig_name = 'anom_SLP_' + np.str(years[0]) + '-' + np.str(years[1]) + '.png'
print(fig_name)
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)

m = Basemap(projection='ortho',lon_0=-40,lat_0=40, resolution='l')
m.drawcoastlines()
m.fillcontinents(color='tan')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,60.))
#m.drawmapboundary(fill_color='aqua')
plt.title("SLP anomaly - " + np.str(years[0]) + '-' + np.str(years[1]), fontsize=20, fontweight='bold')

#x,y = m(*np.meshgrid(df.columns.values,df.index.values))
x,y = m(*np.meshgrid(df.columns.droplevel(None) ,df.index))
c = m.contourf(x, y, anom.values, np.linspace(-3, 3, 16), cmap=plt.cm.seismic, extend='both');
ct = m.contour(x, y, df_clim.values, 10, colors='k');
cb = plt.colorbar(c)
cb.set_label('SLP (mb)')

#### ---- Save Figure ---- ####
#plt.suptitle('Fall surveys', fontsize=16)
fig.set_size_inches(w=8, h=6)
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
plt.close()
    
## if dt_year == 5:    
##     os.system('montage anom*.png  -tile 3x5 -geometry +10+10  -background white  montage_anom.png')
## elif dt_year == 10:
##     os.system('montage anom*.png  -tile 2x4 -geometry +10+10  -background white  montage_anom.png')

## plt.close('all')
