'''
Test EOFs using XEOFS.
Example borrowed from here:

https://xeofs.readthedocs.io/en/latest/auto_examples/1eof/plot_eof-smode.html

Frederic.Cyr@dfo-mpo.gc.ca
October 2023

'''

# Load packages and data:
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import EqualEarth, PlateCarree
import numpy as np
from xeofs.models import EOF
import os

##  ---- Load and prepare the data ---- ##
ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-mld-rean-m_1698742553126.nc')

# Select Gulf of Lion
print('Restrict to Gulf of Lion')
# larger
ds = ds.where((ds['lon'] >= 3) & (ds['lon'] <= 9), drop=True) 
ds = ds.where((ds['lat'] >= 39) & (ds['lat'] <= 44.5), drop=True)
# narrower
#ds = ds.where((ds['lon'] >= 3.5) & (ds['lon'] <= 6.5), drop=True) 
#ds = ds.where((ds['lat'] >= 40.4) & (ds['lat'] <= 42.9), drop=True)

# Select only Jan-Feb.
ds = ds.where(ds['time.month'] <= 2, drop=True)

# "Winter" max values (Jan-Feb)
ds_max = ds.resample(time="As").max() # Winter max
ds_max = ds_max.mean({'lat', 'lon'}) # Spatial average
mld_max = ds_max.to_pandas()
mld_max.index = mld_max.index.year

# "Winter" mean values (Jan-Feb)
ds_mean = ds.resample(time="As").mean() # Winter mean
ds_mean = ds_mean.mean({'lat', 'lon'}) # Spatial average
mld_mean = ds_mean.to_pandas()
mld_mean.index = mld_mean.index.year


# plot
plt.close('all')
fig = plt.figure(3)
ax = mld_mean.plot(linewidth=2,legend=None, color='tab:blue')
plt.grid()
plt.ylabel('MLD mean (m)')
plt.xlabel(' ')
plt.title('Winter (Jan-Feb) mixed Layer depth in the Gulf of Lion')
ax.invert_yaxis()
ax2 = ax.twinx()
mld_max.plot(ax=ax2, linewidth=2,legend=None, color='tab:orange')
ax2.invert_yaxis()
plt.ylabel('MLD max (m)')
ax.legend(['MLD mean'], loc='lower left')
ax2.legend(['MLD max'], loc='lower right')
fig.set_size_inches(w=6,h=4)
fig_name = 'MLD_interannual.png'
plt.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

