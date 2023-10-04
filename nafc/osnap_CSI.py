'''
Script to read OSNAP data prepared by Jonathan.Coyne@df-mpo.gc.ca

Fred's working folders: 
/home/cyrf0006/research/OSNAP

Data Folder:
/home/cyrf0006/data/OSNAP

At NAFC, Deployments 1-4 (PI: G. Han) are located in Funk Isl. deep.
Deployments 5- onward (PI: F. Cyr) are located on Seal Island Transet (CSI) and alignement with the main OSNAP array (C0)
Each iteration usually correspond to a turnover of both moorings.
Deployment_5 were deployed in 2020 / recovered in 2021
Deployment_6 were deployed in 2021 / recovered in 2022
[...]


Frederic.Cyr@dfo-mpo.gc.ca
October 2023

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import cmocean as cmo
import os

path = '/home/cyrf0006/data/OSNAP/Deployment_5/CSI1-01/'

# For CTD
ds = xr.open_mfdataset(path + 'SBE*QC.nc')

# For ADCP
ds_adcp = xr.open_mfdataset(path + 'V100*QC.nc', combine='nested', concat_dim=['INSTRDEPTH'])

## Test to access data from mf ADCP files
# U 75m
U75 = ds_adcp.sel(INSTRDEPTH=75.0, drop=True)['UCUR'].to_dataframe()
U75 = U75.unstack()
U75.columns = U75.columns.droplevel()
U75 = U75.dropna(how='all', axis=1)
U75 = U75.resample('1W').mean() # Weekly ave
# V 75m
V75 = ds_adcp.sel(INSTRDEPTH=75.0, drop=True)['VCUR'].to_dataframe()
V75 = V75.unstack()
V75.columns = V75.columns.droplevel()
V75 = V75.dropna(how='all', axis=1)
V75 = V75.resample('1W').mean() # Weekly ave

# U 175m
U175 = ds_adcp.sel(INSTRDEPTH=175.0, drop=True)['UCUR'].to_dataframe()
U175 = U175.unstack()
U175.columns = U175.columns.droplevel()
U175 = U175.dropna(how='all', axis=1)
U175 = U175.resample('1W').mean() # Weekly ave
# V 175m
V175 = ds_adcp.sel(INSTRDEPTH=175.0, drop=True)['VCUR'].to_dataframe()
V175 = V175.unstack()
V175.columns = V175.columns.droplevel()
V175 = V175.dropna(how='all', axis=1)
V175 = V175.resample('1W').mean() # Weekly ave

## Now plot
levels = np.linspace(-.5, .5, 11)
#U
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.contourf(U75.index, U75.columns, U75.T.values/100, levels, cmap = cmo.cm.balance, extend='both')
plt.contourf(U175.index, U175.columns, U175.T.values/100, levels, cmap = cmo.cm.balance, extend='both')
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.set_label(r'U ($\rm m\,s^{-1}$)', fontsize=12, fontweight='normal')
plt.ylabel('Depth (m)')
plt.tight_layout()
plt.gca().axes.get_xaxis().set_ticklabels([])
# save
fig.set_size_inches(w=6.5, h=4)
fig.set_dpi(300)
figname = 'OSNAP_CSI-1_U.png'
fig.savefig(figname)
os.system('convert -trim ' + figname + ' ' + figname)

#V
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.contourf(V75.index, V75.columns, V75.T.values/100, levels, cmap = cmo.cm.balance, extend='both')
plt.contourf(V175.index, V175.columns, V175.T.values/100, levels, cmap = cmo.cm.balance, extend='both')
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.set_label(r'V ($\rm m\,s^{-1}$)', fontsize=12, fontweight='normal')
plt.ylabel('Depth (m)')
plt.xticks(rotation=-45)
plt.tight_layout()
#save
fig.set_size_inches(w=6.5, h=4)
fig.set_dpi(300)
figname = 'OSNAP_CSI-1_V.png'
fig.savefig(figname)
os.system('convert -trim ' + figname + ' ' + figname)


os.system('montage OSNAP_CSI-1_U.png OSNAP_CSI-1_V.png -tile 1x2 -geometry +1+1  -background white  OSNAP_CSI-1_UV.png')
