'''
This is a test with xarray, see if I can manipulate a lot of netCDF files.
Try this script in /home/cyrf0006/research/AZMP_database
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import os
os.environ['PROJ_LIB'] = '/home/cyrf0006/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap

#import water_masses as wm

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)


ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/netCDF/2*.nc')

# Select a depth range
ds = ds.sel(level=ds['level']<500)

# Select Lake Melville
# Selection of a subset region
ds = ds.where((ds.longitude>-60.5) & (ds.longitude<-58.5), drop=True) # original one
ds = ds.where((ds.latitude>53.3) & (ds.latitude<54.1), drop=True)

# Sort by time dimension
ds = ds.sortby('time')
lat = ds.latitude.values
lon = ds.longitude.values

## ---- Plot map ---- ##
lonLims = [-61, -58] # ake Melville
latLims = [53, 54.2]
proj = 'merc'
fig_name = 'map_CTDs.png'
    
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc', llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='f')
levels = np.linspace(0, 15, 16)
x,y = m(lon, lat)
m.plot(x, y, '.r')
m.fillcontinents(color='tan');

# Monthly average
## ds = ds.sortby('time')
## ds_monthly = ds.resample(time="M").mean('time') 

# To Pandas Dataframe
da_temp = ds['temperature']
df_temp = da_temp.to_pandas()
da_sal = ds['salinity']
df_sal = da_sal.to_pandas()

sal = df_sal.values.reshape(1,df_sal.size)
temp = df_temp.values.reshape(1,df_temp.size)
year = np.tile(df_temp.index.year, (1,500))  

# HERE!!!!!
plt.scatter(sal, temp, c=year)

#Load winter 2019
df_temp = pd.read_pickle('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/temp_W2019.pkl')
df_sal = pd.read_pickle('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/sal_W2019.pkl')
sal1 = df_sal.values.reshape(1,df_sal.size)
temp1 = df_temp.values.reshape(1,df_temp.size)
temp1 = temp1[sal1>0]
sal1 = sal1[sal1>0]
year1 = np.repeat(2019, sal1.size)

plt.figure()
plt.scatter(sal, temp, c=year, year1)
plt.scatter(sal1, temp1, c=np.repeat(2019, sal1.size)) 

