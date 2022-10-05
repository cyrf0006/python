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

YEAR_MAX = 2020


## Load pickled SST per polygons
df_W = pd.read_pickle('/home/cyrf0006/research/lobster/sst_west_coast_method2.pkl')
df_S = pd.read_pickle('/home/cyrf0006/research/lobster/sst_south_coast_method2.pkl')
df_NE = pd.read_pickle('/home/cyrf0006/research/lobster/sst_northeast_coast_method2.pkl')
df_A = pd.read_pickle('/home/cyrf0006/research/lobster/sst_avalon_method2.pkl')
# Reset index (set time)
df_W.reset_index(inplace=True)
df_S.reset_index(inplace=True)
df_NE.reset_index(inplace=True)
df_A.reset_index(inplace=True)
df_W.set_index('time', inplace=True) 
df_S.set_index('time', inplace=True) 
df_NE.set_index('time', inplace=True)
df_A.set_index('time', inplace=True)

# Keep only 1982-2020
df_W = df_W[(df_W.index.year>1981) & (df_W.index.year<=YEAR_MAX)]
df_NE = df_NE[(df_NE.index.year>1981) & (df_NE.index.year<=YEAR_MAX)]
df_S = df_S[(df_S.index.year>1981) & (df_S.index.year<=YEAR_MAX)]
df_A = df_A[(df_A.index.year>1981) & (df_A.index.year<=YEAR_MAX)]


## Daily averages (get rid of lat/lon columns)
df_W_daily = df_W.resample('D').mean().sst
df_NE_daily = df_NE.resample('D').mean().sst
df_S_daily = df_S.resample('D').mean().sst
df_A_daily = df_A.resample('D').mean().sst
# dropna because daily mean re-introduced them
df_W_daily.dropna(inplace=True)
df_NE_daily.dropna(inplace=True)
df_S_daily.dropna(inplace=True)
df_A_daily.dropna(inplace=True)


#### ---- Summer mean Temperature ---- ####
df_W_molt = df_W_daily[(df_W_daily.index.month>=6) & (df_W_daily.index.month<=9)]
df_NE_molt = df_NE_daily[(df_NE_daily.index.month>=6) & (df_NE_daily.index.month<=9)]
df_S_molt = df_S_daily[(df_S_daily.index.month>=6) & (df_S_daily.index.month<=9)]
df_A_molt = df_A_daily[(df_A_daily.index.month>=6) & (df_A_daily.index.month<=9)]
# annual average
df_W_molt = df_W_molt.resample('As').mean()
df_NE_molt = df_NE_molt.resample('As').mean()
df_S_molt = df_S_molt.resample('As').mean()
df_A_molt = df_A_molt.resample('As').mean()

# Concatenate
df_annual = pd.concat([df_W_molt, df_NE_molt, df_S_molt, df_A_molt], axis=1)
df_annual.columns = ['West_Coast', 'Northeast_Coast', 'South_Coast', 'Avalon']
df_annual.index = df_annual.index.year
df_annual.plot()

## Standardized Anomaly
clim_mean = df_annual[(df_annual.index>=1991) & (df_annual.index<=2020)].mean()
clim_std = df_annual[(df_annual.index>=1991) & (df_annual.index<=2020)].std()
anom_std_all = (df_annual-clim_mean) / clim_std
# Plot anomaly
regions = ['West_Coast', 'Avalon', 'Northeast_Coast', 'South_Coast']

for name in regions:
    anom_std = anom_std_all[name]
    df1 = anom_std[anom_std>0]
    df2 = anom_std[anom_std<0]
    fig = plt.figure(1)
    fig.clf()
    width = .7
    p1 = plt.bar(df1.index, np.squeeze(df1.values), width, alpha=0.8, color='indianred', zorder=10)
    p2 = plt.bar(df2.index, np.squeeze(df2.values), width, bottom=0, alpha=0.8, color='steelblue', zorder=10)
    plt.text(1981, 2.4, r'$\rm \overline{X}_{1991-2020}=$' + str(np.round(clim_mean[name], 1)) + r'$\rm~\pm~$' + str(np.round(clim_std[name], 1)) + r'$\rm ^{\circ}C$' )
    plt.ylabel('Standardized anomaly')
    plt.title(name)
    plt.ylim([-3, 3])
    plt.grid()
    # Save Figure
    fig.set_size_inches(w=7,h=4)
    fig_name = 'lfa_sst_mean_anomaly_' + name + '.png'
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)
    del df1, df2, anom_std

del  clim_mean, clim_std, anom_std_all 
        
#### ---- Days above 12---- ####
# keep only above 12
df_W_daily = df_W_daily[df_W_daily>=12]
df_NE_daily = df_NE_daily[df_NE_daily>=12]
df_S_daily = df_S_daily[df_S_daily>=12]
df_A_daily = df_A_daily[df_A_daily>=12]
# Year counts
days_above_W = df_W_daily.resample('As').count()
days_above_NE = df_NE_daily.resample('As').count()
days_above_S = df_S_daily.resample('As').count()
days_above_A = df_A_daily.resample('As').count()

days_above = pd.concat([days_above_W, days_above_NE, days_above_S, days_above_A], axis=1)
days_above.columns = ['West_Coast', 'Northeast_Coast', 'South_Coast', 'Avalon']
days_above.index = days_above.index.year
#days_above.plot()

## Standardized Anomaly
clim_mean = days_above[(days_above.index>=1991) & (days_above.index<=2020)].mean()
clim_std = days_above[(days_above.index>=1991) & (days_above.index<=2020)].std()
anom_std_all = (days_above-clim_mean) / clim_std
# Plot anomaly
regions = ['West_Coast', 'Avalon', 'Northeast_Coast', 'South_Coast']

# Absolute Numbers
for name in regions:
    da12 = days_above[name]
    fig = plt.figure(1)
    fig.clf()
    width = .7
    p1 = plt.bar(da12.index, np.squeeze(da12.values), width, alpha=0.8, color='steelblue', zorder=10)
    plt.ylabel('Days >12')
    plt.title(name)
    #plt.ylim([-3, 3])
    plt.xlim([1980, 2021])
    plt.grid()
    # Save Figure
    fig.set_size_inches(w=7,h=4)
    fig_name = 'lfa_days_above12_' + name + '.png'
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)
    del da12

del  clim_mean, clim_std, anom_std_all
    
#### ---- Days above 15---- ####
# Reload
## Daily averages (get rid of lat/lon columns)
df_W_daily = df_W.resample('D').mean().sst
df_NE_daily = df_NE.resample('D').mean().sst
df_S_daily = df_S.resample('D').mean().sst
df_A_daily = df_A.resample('D').mean().sst
# dropna because daily mean re-introduced them
df_W_daily.dropna(inplace=True)
df_NE_daily.dropna(inplace=True)
df_S_daily.dropna(inplace=True)
df_A_daily.dropna(inplace=True)
# keep only above 15
df_W_daily = df_W_daily[df_W_daily>=15]
df_NE_daily = df_NE_daily[df_NE_daily>=15]
df_S_daily = df_S_daily[df_S_daily>=15]
df_A_daily = df_A_daily[df_A_daily>=15]
# Year counts
days_above_W = df_W_daily.resample('As').count()
days_above_NE = df_NE_daily.resample('As').count()
days_above_S = df_S_daily.resample('As').count()
days_above_A = df_A_daily.resample('As').count()

days_above = pd.concat([days_above_W, days_above_NE, days_above_S, days_above_A], axis=1)
days_above.columns = ['West_Coast', 'Northeast_Coast', 'South_Coast', 'Avalon']
days_above.index = days_above.index.year
#days_above.plot.bar()
#plt.grid()
#plt.title('number of days above 15 degC')

## Standardized Anomaly
clim_mean = days_above[(days_above.index>=1991) & (days_above.index<=2020)].mean()
clim_std = days_above[(days_above.index>=1991) & (days_above.index<=2020)].std()
anom_std_all = (days_above-clim_mean) / clim_std
# Plot anomaly
regions = ['West_Coast', 'Avalon', 'Northeast_Coast', 'South_Coast']

# Absolute Numbers
for name in regions:
    da15 = days_above[name]
    fig = plt.figure(1)
    fig.clf()
    width = .7
    p1 = plt.bar(da15.index, np.squeeze(da15.values), width, alpha=0.8, color='steelblue', zorder=10)
    plt.ylabel('Days >15')
    plt.title(name)
    #plt.ylim([-3, 3])
    plt.xlim([1980, 2021])
    plt.grid()
    # Save Figure
    fig.set_size_inches(w=7,h=4)
    fig_name = 'lfa_days_above15_' + name + '.png'
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)
    del da15

del  clim_mean, clim_std, anom_std_all


keyboard

#### ---- Days above 12 / below 18 ---- ####
# Reload
## Daily averages (get rid of lat/lon columns)
df_W_daily = df_W.resample('D').mean().sst
df_NE_daily = df_NE.resample('D').mean().sst
df_S_daily = df_S.resample('D').mean().sst
df_A_daily = df_A.resample('D').mean().sst
# dropna because daily mean re-introduced them
df_W_daily.dropna(inplace=True)
df_NE_daily.dropna(inplace=True)
df_S_daily.dropna(inplace=True)
df_A_daily.dropna(inplace=True)
# keep only above 12
df_W_daily = df_W_daily[(df_W_daily>=12) & (df_W_daily<=18)]
df_NE_daily = df_NE_daily[(df_NE_daily>=12) & (df_W_daily<=18)]
df_S_daily = df_S_daily[(df_S_daily>=12) & (df_W_daily<=18)]
df_A_daily = df_A_daily[(df_A_daily>=12) & (df_W_daily<=18)]
# Year counts
days_above_W = df_W_daily.resample('As').count()
days_above_NE = df_NE_daily.resample('As').count()
days_above_S = df_S_daily.resample('As').count()
days_above_A = df_A_daily.resample('As').count()

days_above = pd.concat([days_above_W, days_above_NE, days_above_S, days_above_A], axis=1)
days_above.columns = ['West_Coast', 'Northeast_Coast', 'South_Coast', 'Avalon']
days_above.index = days_above.index.year
#days_above.plot()
#plt.grid()
#plt.title('number of days between 12-18 degC')

## Standardized Anomaly
clim_mean = days_above[(days_above.index>=1991) & (days_above.index<=2020)].mean()
clim_std = days_above[(days_above.index>=1991) & (days_above.index<=2020)].std()
anom_std_all = (days_above-clim_mean) / clim_std
# Plot anomaly
regions = ['West_Coast', 'Avalon', 'Northeast_Coast', 'South_Coast']

for name in regions:
    anom_std = anom_std_all[name]
    df1 = anom_std[anom_std>0]
    df2 = anom_std[anom_std<0]
    fig = plt.figure(1)
    fig.clf()
    width = .7
    p1 = plt.bar(df1.index, np.squeeze(df1.values), width, alpha=0.8, color='indianred', zorder=10)
    p2 = plt.bar(df2.index, np.squeeze(df2.values), width, bottom=0, alpha=0.8, color='steelblue', zorder=10)
    plt.text(1981, 2.4, r'$\rm \overline{X}_{1991-2020}=$' + str(np.int(np.round(clim_mean[name]))) + r'$\rm~\pm~$' + str(np.int(np.round(clim_std[name]))) + r' days' )
    plt.ylabel('Standardized anomaly')
    plt.title(name)
    plt.ylim([-3, 3])
    plt.grid()
    # Save Figure
    fig.set_size_inches(w=7,h=4)
    fig_name = 'lfa_days_range12-18_anomaly_' + name + '.png'
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)
    del df1, df2, anom_std

del  clim_mean, clim_std, anom_std_all 
    
    
#### ---- Week of max temp.---- ####
df_W_weekly = df_W.resample('W').mean().sst
df_NE_weekly = df_NE.resample('W').mean().sst
df_S_weekly = df_S.resample('W').mean().sst
df_A_weekly = df_A.resample('W').mean().sst
# restrict years
df_W_weekly = df_W_weekly[df_W_weekly.index.year<=YEAR_MAX]
df_NE_weekly = df_NE_weekly[df_NE_weekly.index.year<=YEAR_MAX]
df_S_weekly = df_S_weekly[df_S_weekly.index.year<=YEAR_MAX]
df_A_weekly = df_A_weekly[df_A_weekly.index.year<=YEAR_MAX]


# find max weekly average temperature
weekly_max_W = df_W_weekly.resample('As').max()
weekly_max_NE = df_NE_weekly.resample('As').max()
weekly_max_S = df_S_weekly.resample('As').max()
weekly_max_A = df_A_weekly.resample('As').max()

weekly_max = pd.concat([weekly_max_W, weekly_max_NE, weekly_max_S, weekly_max_A], axis=1)
weekly_max.columns = ['West_Coast', 'Northeast_Coast', 'South_Coast', 'Avalon']
weekly_max.index = weekly_max.index.year
#weekly_max.plot.bar()
#plt.grid()
#plt.title('Maximum weekly average temperature')

## Standardized Anomaly
clim_mean = weekly_max[(weekly_max.index>=1991) & (weekly_max.index<=2020)].mean()
clim_std = weekly_max[(weekly_max.index>=1991) & (weekly_max.index<=2020)].std()
anom_std_all = (weekly_max-clim_mean) / clim_std
# Plot anomaly
regions = ['West_Coast', 'Avalon', 'Northeast_Coast', 'South_Coast']
region_name_EN = ['West Coast', 'Avalon', 'Northeast Coast', 'South Coast']
region_name_FR = ['Côte Ouest', 'Avalon', 'Côte Nord-Ouest', 'Côte Sud']

for idx, name in enumerate(regions):
    anom_std = anom_std_all[name]
    df1 = anom_std[anom_std>0]
    df2 = anom_std[anom_std<0]
    fig = plt.figure(1)
    fig.clf()
    width = .7
    p1 = plt.bar(df1.index, np.squeeze(df1.values), width, alpha=0.8, color='indianred', zorder=10)
    p2 = plt.bar(df2.index, np.squeeze(df2.values), width, bottom=0, alpha=0.8, color='steelblue', zorder=10)
    plt.text(1981, 2.4, r'$\rm \overline{X}_{1991-2020}=$' + str(np.round(clim_mean[name], 1)) + r'$\rm~\pm~$' + str(np.round(clim_std[name], 1)) + r'$\rm ^{\circ}C$' )
    plt.ylabel('Normalized anomaly')
    plt.title(region_name_EN[idx])
    plt.ylim([-3, 3])
    plt.grid()
    # Save Figure
    fig.set_size_inches(w=7,h=4)
    fig_name = 'lfa_weekly_max_anomaly_' + name + '.png'
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)

    # Save French Figure
    plt.ylabel('Anomalie normalisée')
    plt.title(region_name_FR[idx])
    fig_name = 'lfa_weekly_max_anomaly_' + name + '_FR.png'
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)   
    
    del df1, df2, anom_std

del  clim_mean, clim_std, anom_std_all 


#### ---- WOY of max temp.---- ####
df_W_weekly = df_W.resample('W').mean().sst
df_NE_weekly = df_NE.resample('W').mean().sst
df_S_weekly = df_S.resample('W').mean().sst
df_A_weekly = df_A.resample('W').mean().sst

# find week of max average temperature
weekly_stack_W = df_W_weekly.groupby([(df_W_weekly.index.year), (df_W_weekly.index.weekofyear)]).mean()
weekly_stack_NE = df_NE_weekly.groupby([(df_W_weekly.index.year), (df_W_weekly.index.weekofyear)]).mean()
weekly_stack_S = df_S_weekly.groupby([(df_W_weekly.index.year), (df_W_weekly.index.weekofyear)]).mean()
weekly_stack_A = df_A_weekly.groupby([(df_W_weekly.index.year), (df_W_weekly.index.weekofyear)]).mean()
weekly_W = weekly_stack_W.unstack()
weekly_NE = weekly_stack_NE.unstack()
weekly_S = weekly_stack_S.unstack()
weekly_A = weekly_stack_A.unstack()
woy_max_W = weekly_W.idxmax(axis=1)  
woy_max_NE = weekly_NE.idxmax(axis=1)  
woy_max_S = weekly_S.idxmax(axis=1)  
woy_max_A = weekly_A.idxmax(axis=1)  

woy_max = pd.concat([woy_max_W, woy_max_NE, woy_max_S, woy_max_A], axis=1)
woy_max.columns = ['West_Coast', 'Northeast_Coast', 'South_Coast', 'Avalon']


## Standardized Anomaly
clim_mean = woy_max[(woy_max.index>=1991) & (woy_max.index<=2020)].mean()
clim_std = woy_max[(woy_max.index>=1991) & (woy_max.index<=2020)].std()
anom_std_all = (woy_max-clim_mean) / clim_std
# Plot anomaly
regions = ['West_Coast', 'Avalon', 'Northeast_Coast', 'South_Coast']

for name in regions:
    anom_std = anom_std_all[name]
    df1 = anom_std[anom_std>0]
    df2 = anom_std[anom_std<0]
    fig = plt.figure(1)
    fig.clf()
    width = .7
    p1 = plt.bar(df1.index, np.squeeze(df1.values), width, alpha=0.8, color='indianred', zorder=10)
    p2 = plt.bar(df2.index, np.squeeze(df2.values), width, bottom=0, alpha=0.8, color='steelblue', zorder=10)
    plt.ylabel('Standardized anomaly')
    plt.title(name)
    plt.ylim([-3, 3])
    plt.grid()
    # Save Figure
    fig.set_size_inches(w=7,h=4)
    fig_name = 'lfa_woy_max_anomaly_' + name + '.png'
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)
    del df1, df2, anom_std

del  clim_mean, clim_std, anom_std_all


os.system('montage lfa_days_above12_anomaly_West_Coast.png lfa_days_above12_anomaly_Northeast_Coast.png lfa_days_above12_anomaly_South_Coast.png lfa_days_above12_anomaly_Avalon.png -tile 2x2 -geometry +20+25  -background white  lfa_days_above12.png')

os.system('montage lfa_days_above15_anomaly_West_Coast.png lfa_days_above15_anomaly_Northeast_Coast.png lfa_days_above15_anomaly_South_Coast.png lfa_days_above15_anomaly_Avalon.png -tile 2x2 -geometry +20+25  -background white  lfa_days_above15.png')

os.system('montage lfa_days_range12-18_anomaly_West_Coast.png  lfa_days_range12-18_anomaly_Northeast_Coast.png lfa_days_range12-18_anomaly_South_Coast.png lfa_days_range12-18_anomaly_Avalon.png -tile 2x2 -geometry +20+25  -background white  lfa_days_range12-18.png')

os.system('montage lfa_weekly_max_anomaly_West_Coast.png lfa_weekly_max_anomaly_Northeast_Coast.png lfa_weekly_max_anomaly_South_Coast.png lfa_weekly_max_anomaly_Avalon.png -tile 2x2 -geometry +20+25  -background white  lfa_weekly_max.png')

os.system('montage lfa_weekly_max_anomaly_West_Coast_FR.png lfa_weekly_max_anomaly_Northeast_Coast_FR.png lfa_weekly_max_anomaly_South_Coast_FR.png lfa_weekly_max_anomaly_Avalon_FR.png -tile 2x2 -geometry +20+25  -background white  lfa_weekly_max_FR.png')

os.system('montage lfa_woy_max_anomaly_West_Coast.png lfa_woy_max_anomaly_Northeast_Coast.png lfa_woy_max_anomaly_South_Coast.png lfa_woy_max_anomaly_Avalon.png -tile 2x2 -geometry +20+25  -background white  lfa_woy_max.png')

os.system('montage lfa_sst_mean_anomaly_West_Coast.png lfa_sst_mean_anomaly_Northeast_Coast.png lfa_sst_mean_anomaly_South_Coast.png lfa_sst_mean_anomaly_Avalon.png -tile 2x2 -geometry +20+25  -background white  lfa_sst_mean.png')


