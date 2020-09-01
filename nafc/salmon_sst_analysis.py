'''

Analysis of SSTs in LFA polygons. 
See Pickle files generated in lfa_sst_extract.py

This script assumes method=2

data source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html
https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=71800&vid=676
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import shapefile 
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import plotly.offline as py_off
from plotly.graph_objs import *
import cmocean
import os

## Load pickled SST per polygons
df_W = pd.read_pickle('sst_west_coast_method2.pkl')
df_S = pd.read_pickle('sst_south_coast_method2.pkl')
df_NE = pd.read_pickle('sst_northeast_coast_method2.pkl')
df_A = pd.read_pickle('sst_avalon_method2.pkl')
# Reset index (set time)
df_W.reset_index(inplace=True)
df_S.reset_index(inplace=True)
df_NE.reset_index(inplace=True)
df_A.reset_index(inplace=True)
df_W.set_index('time', inplace=True) 
df_S.set_index('time', inplace=True) 
df_NE.set_index('time', inplace=True)
df_A.set_index('time', inplace=True)

# Keep only 1982-2019
df_W = df_W[df_W.index.year>1981]
df_NE = df_NE[df_NE.index.year>1981]
df_S = df_S[df_S.index.year>1981]
df_A = df_A[df_A.index.year>1981]


## Weekly averages (get rid of lat/lon columns)
df_W_weekly = df_W.resample('W').mean().sst
df_NE_weekly = df_NE.resample('W').mean().sst
df_S_weekly = df_S.resample('W').mean().sst
df_A_weekly = df_A.resample('W').mean().sst
# dropna because weekly mean re-introduced them
df_W_weekly.dropna(inplace=True)
df_NE_weekly.dropna(inplace=True)
df_S_weekly.dropna(inplace=True)
df_A_weekly.dropna(inplace=True)

## Save .csv
df_W_weekly.to_csv('sst_weekly_westcoast.csv', float_format='%.3f')
df_NE_weekly.to_csv('sst_weekly_northeast.csv', float_format='%.3f')
df_S_weekly.to_csv('sst_weekly_southcoast.csv', float_format='%.3f')
df_A_weekly.to_csv('sst_weekly_avalon.csv', float_format='%.3f')

df_W_weekly.name='West_Coast'
df_NE_weekly.name='Northeast_Coast'
df_S_weekly.name='South_Coast'
df_A_weekly.name='Avalon'
df = pd.concat([df_W_weekly, df_NE_weekly, df_S_weekly, df_A_weekly], axis=1)

# Plot timeseries
regions = ['West_Coast', 'Avalon', 'Northeast_Coast', 'South_Coast']

for name in regions:

    fig = plt.figure(1)
    fig.clf()
    plt.plot(df[name])
    
    plt.ylabel(r'Weekly T($\rm ^{\circ}C$)')
    plt.title(name)
    #plt.ylim([-3, 3])
    plt.grid()
    # Save Figure
    fig.set_size_inches(w=7,h=4)
    fig_name = 'weekly_temp_' + name + '.png'
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)

   
