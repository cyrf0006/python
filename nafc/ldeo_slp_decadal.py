# data source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=71800&vid=676

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
#import datetime
from mpl_toolkits.basemap import Basemap

# Some parameters
dt_year = 10
years = np.arange(1950, 2020, dt_year)# years to loop
months = [1, 12] # months to keep
#v = np.arange(990, 1030) # SLP values
v = np.arange(995, 1025) # SLP values

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
p = da.to_pandas()

# Compute climatology
#p_clim = p[(p.items.month>=months[0]) & (p.items.year<=months[1])]
#df_clim = p_clim.mean(axis=0)
df_clim = p.mean(axis=0)

# Load NCAM data
df_ncam = pd.read_csv('/home/cyrf0006/data/NCAM/multispic_estimates.csv', index_col='year')
df_cod = df_ncam[df_ncam.species=='Cod']
df_had = df_ncam[df_ncam.species=='Haddock']
df_hak = df_ncam[df_ncam.species=='Hake']
df_pla = df_ncam[df_ncam.species=='Plaice']
df_red = df_ncam[df_ncam.species=='Redfish']
df_ska = df_ncam[df_ncam.species=='Skate']
df_wit = df_ncam[df_ncam.species=='Witch']
df_yel = df_ncam[df_ncam.species=='Yellowtail']

df_pe = pd.concat([df_cod.pe, df_had.pe, df_hak.pe, df_pla.pe, df_red.pe, df_ska.pe, df_wit.pe, df_yel.pe], axis=1).mean(axis=1)
df_B = pd.concat([df_cod.B, df_had.B, df_hak.B, df_pla.B, df_red.B, df_ska.B, df_wit.B, df_yel.B], axis=1).mean(axis=1)


# loop on years
for year in years:
    # select years
    p_year = p[(p.items.year>=year) & (p.items.year<=year+dt_year-1)]

    # select months
    #p_year = p[(p.items.month>=months[0]) & (p.items.year<=months[1])]
        
    # average all
    df = np.squeeze(p_year.mean(axis=0))

    
    fig_name = 'SLP_map_' + np.str(year) + '-' + np.str(year+dt_year-1) + '.png'
    print fig_name
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=1)

    m = Basemap(projection='ortho',lon_0=-40,lat_0=40, resolution='l')
    m.drawcoastlines()
    m.fillcontinents(color='tan')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    #m.drawmapboundary(fill_color='aqua')
    plt.title("Sea Level Pressure - " + np.str(year) + '-' + np.str(year+dt_year-1))

    x,y = m(*np.meshgrid(df.columns.values,df.index.values))
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
    fig_name = 'anom_SLP_' + np.str(year) + '-' + np.str(year+dt_year-1) + '.png'
    print fig_name
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=1)

    m = Basemap(projection='ortho',lon_0=-40,lat_0=40, resolution='l')
    m.drawcoastlines()
    m.fillcontinents(color='tan')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    #m.drawmapboundary(fill_color='aqua')
    plt.title("Sea Level Pressure anomaly- " + np.str(year) + '-' + np.str(year+dt_year-1))

    x,y = m(*np.meshgrid(df.columns.values,df.index.values))
    c = m.contourf(x, y, anom.values, np.linspace(-3, 3, 16), cmap=plt.cm.seismic, extend='both');
    ct = m.contour(x, y, df_clim.values, 10, colors='k');
    cb = plt.colorbar(c)
    cb.set_label('SLP (mb)')

    #### ---- Add ecosystem trends ---- ####
    trend = df_pe[(df_pe.index>=year) & (df_pe.index<year+dt_year)].mean()
    if trend>0:
        plt.annotate('+' + "{:.2f}".format(np.abs(trend)), xy=(.83, .9), xycoords='axes fraction', color='g', fontsize=22, fontweight='bold')
    elif trend<0:
        plt.annotate('-' + "{:.2f}".format(np.abs(trend)), xy=(.83, .9), xycoords='axes fraction', color='r', fontsize=22, fontweight='bold')

        
    #### ---- Save Figure ---- ####
    #plt.suptitle('Fall surveys', fontsize=16)
    fig.set_size_inches(w=8, h=6)
    fig.savefig(fig_name, dpi=150)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)

if dt_year == 5:    
    os.system('montage anom*.png  -tile 3x5 -geometry +10+10  -background white  montage_anom.png')
elif dt_year == 10:
    os.system('montage anom*.png  -tile 2x4 -geometry +10+10  -background white  montage_anom.png')

