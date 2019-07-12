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

## -----> For survey transect plot
import azmp_sections_tools as az
az.standard_section_plot('/home/cyrf0006/data/dev_database/2015.nc', '20115', 'BB', 'sigma-t')


## -----> For all year climatology

ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/20*.nc')

# Select a depth range
ds = ds.sel(level=ds['level']<1000)

# Sort by time dimension
ds = ds.sortby('time')

ds = ds.sel(time=ds['time.year']>=2013)

# Monthly average (This takes a lot of time given the sorting above)
#ds_monthly = ds.resample('M', dim='time', how='mean') #deprecated

#ds_monthly = ds.resample(time="M").mean('time') 

# Find only BB Section [THIS IS SICK!]
ds = ds.isel(time=[i for i,item in enumerate(ds['comments'].values) if "BB-0" in item])


ds = ds.resample(time="M").mean('time') 


# To Pandas Dataframe
da_temp = ds['temperature']
df_temp = da_temp.to_pandas()
da_sal = ds['salinity']
df_sal = da_sal.to_pandas()

## Un-comment for fall only
df_sal = df_sal[(df_sal.index.month>=5) & (df_sal.index.month>=9)]
df_temp = df_temp[(df_temp.index.month>=5) & (df_temp.index.month>=9)]

sal_prof = df_sal.mean(axis=0)
temp_prof = df_temp.mean(axis=0)

Z = ds.level.values
SP = sal_prof.values
PT = temp_prof.values

SA = gsw.SA_from_SP(SP, Z, -50, 50)
CT = gsw.CT_from_pt(SA, PT)
RHO = gsw.rho(SA, CT, Z)



fig = plt.figure(1)
plt.clf()

plt.plot(RHO, Z)
plt.xlabel(r'$\rm \rho (kg m^{-3})$', fontsize=14, fontweight='bold')
plt.ylabel('Z (m)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
fig.set_size_inches(w=8, h=8)
fig_name = 'BB_section_rho.png'
fig.set_dpi(300)
fig.savefig(fig_name)

fig = plt.figure(2)
plt.clf()
# AX1 - T
ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
ax1.plot(CT, Z)
ax1.grid()
ax1.set_ylim(0, 300)
ax1.invert_yaxis()
ax1.set_xlabel(r'$\rm \Theta (^{\circ}C)$')
ax1.set_ylabel(r'Depth (m)')

# AX2 - S
ax2 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1)
ax2.plot(SA, Z)
ax2.grid()
ax2.set_ylim(0, 300)
ax2.set_yticklabels([])
ax2.invert_yaxis()
ax2.set_xlabel(r'$\rm S_A (g Kg^{-1})$')
#plt.title('BBline climatology (Oct-Dec. / 2015-2017)')
plt.title('BB-01 to BB-09 climatology (May-Sept. / 2013-2017)')

# AX3 - Rho
ax3 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1)
ax3.plot(RHO, Z)
ax3.grid()
ax3.set_ylim(0, 300)
ax3.set_xlim(1025, 1030)
ax3.set_yticklabels([])
ax3.invert_yaxis()
ax3.set_xlabel(r'$\rm \rho (Kg m^{-3})$')


fig.set_size_inches(w=12, h=8)
fig_name = 'BB_section_climato.png'
fig.set_dpi(300)
fig.savefig(fig_name)



