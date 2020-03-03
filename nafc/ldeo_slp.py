# data source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=71800&vid=676

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
#import datetime
import os
os.environ['PROJ_LIB'] = '/home/cyrf0006/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap

# Some parameters
years = [2019, 2020] # years to loop
v = np.arange(990, 1030) # SLP values

# Load SLP data from NOAA ESRL
#ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mnmean.nc')
ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mon.mean.nc')
#ds = xr.open_mfdataset('/home/cyrf0006/data/NOAA_ESRL/slp.201*.nc')

# Selection of a subset region
#ds = ds.where((ds.lon>=-120) & (ds.lon<=30), drop=True) # original one
#ds = ds.where((ds.lat>=0) & (ds.lat<=90), drop=True)

# Weekly mean <-------- NOT FINISHED
#ds = ds.resample('1W', dim='time', how='mean')

da = ds['slp']
#p = da.to_pandas()
p = da.to_dataframe()

#longitude = ds.lon.values
#latitude = ds.lat.values
v = np.arange(990, 1030, 2)

# Compute climatology
df_clim = p.groupby(level=[1,2]).mean().unstack()

# loop on years
for year in years:
    
    p_year = p[p.index.get_level_values('time').year==year]  
    months = p_year.index.get_level_values('time').month.unique()
    #p_year = p[p.items.year==year]
    #months = p_year.items.month

    # loop on months
    for month in months:
        df = p_year[p_year.index.get_level_values('time').month==month].unstack().droplevel(level='time')
        fig_name = 'SLP_map_' + np.str(year) + '-' + np.str(month).zfill(2) + '.png'
        print(fig_name)
        ## ---- plot ---- ##
        plt.clf()
        fig, ax = plt.subplots(nrows=1, ncols=1)

        m = Basemap(projection='ortho',lon_0=-40,lat_0=40, resolution='l')
        m.drawcoastlines()
        m.fillcontinents(color='tan')
        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,120.,30.))
        m.drawmeridians(np.arange(0.,420.,60.))
        #m.drawmapboundary(fill_color='aqua')
        plt.title("Sea Level Pressure - " + np.str(year) + '-' + np.str(month).zfill(2))
        
        x,y = m(*np.meshgrid(df.columns.droplevel(None) ,df.index))
        c = m.contourf(x, y, df.values, v, cmap=plt.cm.inferno, extend='both');
        ct = m.contour(x, y, df_clim.values, 10, colors='k');
        cb = plt.colorbar(c)
        cb.set_label('SLP (mb)')

        #### ---- Save Figure ---- ####
        fig.set_size_inches(w=8, h=6)
        fig.savefig(fig_name, dpi=200)


