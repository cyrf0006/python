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
import gsw
import water_masses as wm



font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)

## -----> For all year climatology

ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/201*.nc')

# Select a depth range
ds = ds.sel(level=ds['level']<30)

# Sort by time dimension
ds = ds.sortby('time')

ds = ds.sel(time=ds['time.year']>=2013)

# Find only BB Section [THIS IS SICK!]
ds = ds.isel(time=[i for i,item in enumerate(ds['comments'].values) if "BB-01" in item])

ds = ds.resample(time="M").mean('time') 


# To Pandas Dataframe
da_temp = ds['temperature']
df_temp = da_temp.to_pandas()
da_sal = ds['salinity']
df_sal = da_sal.to_pandas()

# Keep only surface
df_temp = df_temp.mean(axis=1)
df_sal = df_sal.mean(axis=1)
df_temp.dropna(inplace=True)
df_sal.dropna(inplace=True)

## Un-comment for fall only
#df_sal = df_sal[(df_sal.index.month>=5) & (df_sal.index.month>=9)]
#df_temp = df_temp[(df_temp.index.month>=5) & (df_temp.index.month>=9)]

Z = ds.level.values.mean()
SP = df_sal.values
PT = df_temp.values

SA = gsw.SA_from_SP(SP, Z, -50, 50)
CT = gsw.CT_from_pt(SA, PT)
RHO = gsw.rho(SA, CT, Z)

df_rho = pd.DataFrame(RHO)
df_rho.index = df_temp.index


# plot & save
fig = plt.figure(1)
plt.clf()
#df_rho.plot()
plt.plot(df_temp.index, RHO, 'ok')
plt.ylabel(r'$\rm \rho (kg m^{-3})$', fontsize=14, fontweight='bold')
plt.xlabel('time', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
fig.set_size_inches(w=8, h=8)
fig_name = 'BB01_density_2013-2018.png'
fig.set_dpi(300)
fig.savefig(fig_name)
