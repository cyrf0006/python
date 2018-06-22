'''
This is a test script to read the entire timeseries, and extract timeseries of average temperature based on NAFO division.
Eventually, I would make this a function where parameters can be selected.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import water_masses as wm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapefile 
import os

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)

## ---- Some info on what we want ---- ##



## ---- get NAFO divisions ---- ##
myshp = open('/home/cyrf0006/AZMP/utils/NAFO_divisions/Divisions/Divisions.shp', 'rb')
mydbf = open('/home/cyrf0006/AZMP/utils/NAFO_divisions/Divisions/Divisions.dbf', 'rb')
r = shapefile.Reader(shp=myshp, dbf=mydbf)
records = r.records()
shapes = r.shapes()

# Fill dictionary with NAFO divisions
nafo_divisions = {}
for idx, rec in enumerate(records):
    if rec[-1] == '':
        continue
    else:
        nafo_divisions[rec[-1]] = np.array(shapes[idx].points)


## ---- Get temperature data ---- ##
if os.path.exists("./df_summer_yearly.pkl"):
    print 'Load existing pickled file!'
    df_fall_yearly = pd.read_pickle('df_fall_yearly.pkl')
    df_spring_yearly = pd.read_pickle('df_spring_yearly.pkl')
    df_summer_yearly = pd.read_pickle('df_summer_yearly.pkl')

else:
    print 'Extract data from entire NAFC database (may take some time...)'
    # This is a dataset
    ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/*.nc') 

    # Select only years after 1970
    ds = ds.sel(time=ds['time.year']>=1970)

    # Select a depth range
    ds = ds.sel(level=ds['level']<700)
    ds = ds.sel(level=ds['level']>0)

    # Selection of a subset region
    ds = ds.where((ds.longitude>-60) & (ds.longitude<-40), drop=True)
    ds = ds.where((ds.latitude>38) & (ds.latitude<58), drop=True)

    # Sort time dimension (this takes time to display!!)
    #ds = ds.isel(time=np.argsort(ds.time))

    # Selection according to season (xarray seasons)
    ## ds_spring = ds.sel(time=ds['time.season']=='MAM')
    ## ds_summer = ds.sel(time=ds['time.season']=='JJA')
    ## ds_fall = ds.sel(time=ds['time.season']=='SON')

    # Selection according to season (DFO sampling seasons)
    ds_spring = ds.sel(time=((ds['time.month']>=4)) & ((ds['time.month']<=6)))
    ds_summer = ds.sel(time=((ds['time.month']>=7)) & ((ds['time.month']<=9)))
    ds_fall = ds.sel(time=((ds['time.month']>=10)) & ((ds['time.month']<=12)))
    
    # Temperature to Pandas Dataframe
    da_spring = ds_spring['temperature']
    df_spring = da_spring.to_pandas()
    da_fall = ds_fall['temperature']
    df_fall = da_fall.to_pandas()
    da_summer = ds_summer['temperature']
    df_summer = da_summer.to_pandas()
    
    # coordinates to array and dataframe
    coord_fall = ds_fall[['latitude', 'longitude']] # For map
    df_coord_fall = coord_fall.to_dataframe()
    coord_spring = ds_spring[['latitude', 'longitude']]
    df_coord_spring = coord_spring.to_dataframe()
    coord_summer = ds_summer[['latitude', 'longitude']]
    df_coord_summer = coord_summer.to_dataframe()
    
    lat_spring = np.array(ds_spring['latitude']) # to get indices in NAFO div.
    lon_spring = np.array(ds_spring['longitude'])
    lat_fall = np.array(ds_fall['latitude'])
    lon_fall = np.array(ds_fall['longitude'])
    lat_summer = np.array(ds_summer['latitude'])
    lon_summer = np.array(ds_summer['longitude'])

    # Find casts in polygon
    polygon3K = Polygon(nafo_divisions['3K'])
    polygon3L = Polygon(nafo_divisions['3L'])
    polygon3N = Polygon(nafo_divisions['3N'])
    polygon3O = Polygon(nafo_divisions['3O'])
    polygon3Ps = Polygon(nafo_divisions['3Ps'])
    polygon2J = Polygon(nafo_divisions['2J'])

    # Find indices
    spring_idx = []
    for idx, profile in enumerate(np.array(lat_spring)):
        point = Point(lon_spring[idx], lat_spring[idx])
        if polygon3L.contains(point) | polygon3N.contains(point) | polygon3O.contains(point) | polygon3Ps.contains(point):
            spring_idx.append(idx)

    fall_idx = []
    for idx, profile in enumerate(np.array(lat_fall)):
        point = Point(lon_fall[idx], lat_fall[idx])
        if polygon2J.contains(point) | polygon3K.contains(point) | polygon3L.contains(point) | polygon3N.contains(point) | polygon3O.contains(point) | polygon3Ps.contains(point):
            fall_idx.append(idx)

    summer_idx = []
    for idx, profile in enumerate(np.array(lat_summer)):
        point = Point(lon_summer[idx], lat_summer[idx])
        if polygon2J.contains(point) | polygon3K.contains(point) | polygon3L.contains(point) | polygon3N.contains(point) | polygon3O.contains(point) | polygon3Ps.contains(point):
            summer_idx.append(idx)

            
    # reduce range (overwrite preceeding dataframe)
    df_spring = df_spring.iloc[spring_idx]
    df_fall = df_fall.iloc[fall_idx]
    df_summer = df_summer.iloc[summer_idx]
    df_coord_spring = df_coord_spring.iloc[spring_idx]
    df_coord_spring.to_pickle('spring_coords.pkl')
    df_coord_fall = df_coord_fall.iloc[fall_idx]
    df_coord_fall.to_pickle('fall_coords.pkl')
    df_coord_summer = df_coord_summer.iloc[summer_idx]
    df_coord_summer.to_pickle('summer_coords.pkl')

    # Month average (monthly average before seasonal average reduces variance)
    df_spring_count = df_spring.resample('M').count()
    df_fall_count = df_fall.resample('M').count()
    df_summer_count = df_summer.resample('M').count()

    df_spring = df_spring.resample('M').mean()
    df_fall = df_fall.resample('M').mean()
    df_summer = df_summer.resample('M').mean()

    # Year average
    df_spring_yearly = df_spring.resample('As').mean()
    df_fall_yearly = df_fall.resample('As').mean()
    df_summer_yearly = df_summer.resample('As').mean()

    df_spring_yearly.to_pickle('df_spring_yearly.pkl')
    df_fall_yearly.to_pickle('df_fall_yearly.pkl')
    df_summer_yearly.to_pickle('df_summer_yearly.pkl')

# !!!!!!!!! CHECK SUMMER TO MAKE SURE TEMP IS HIGH !!!!!!!!!!!!!!!

## Tcontour-seasonal
fig = plt.figure(3)
plt.clf()
v = np.arange(-2,10)
plt.contourf(df_spring_yearly.index, df_spring_yearly.columns, df_spring_yearly.T, 20, cmap=plt.cm.RdBu_r)  
plt.xlim([pd.Timestamp('1970-01-01'), pd.Timestamp('2017-12-01')])
plt.gca().invert_yaxis()
plt.ylabel('Depth (m)', fontsize=14, fontweight='bold')
plt.xlabel('years', fontsize=14, fontweight='bold')
cb = plt.colorbar()
plt.title(r'$\rm T(^{\circ}C)$ - Spring', fontsize=14, fontweight='bold')
fig.set_size_inches(w=12,h=9)
fig_name = 'T_spring_1970-2016.png'
fig.savefig(fig_name)

fig = plt.figure(4)
plt.clf()
v = np.arange(-2,10)
plt.contourf(df_fall_yearly.index, df_fall_yearly.columns, df_fall_yearly.T, 20, cmap=plt.cm.RdBu_r)  
plt.xlim([pd.Timestamp('1970-01-01'), pd.Timestamp('2017-12-01')])
plt.gca().invert_yaxis()
plt.ylabel('Depth (m)', fontsize=14, fontweight='bold')
plt.xlabel('years', fontsize=14, fontweight='bold')
cb = plt.colorbar()
plt.title(r'$\rm T(^{\circ}C)$ - Fall', fontsize=14, fontweight='bold')
fig.set_size_inches(w=12,h=9)
fig_name = 'T_fall_1970-2016.png'
fig.savefig(fig_name)


## --- temperature plots --- ##
# Subplots
xticks = [pd.Timestamp('1980-1-1'), pd.Timestamp('1985-1-1'),
            pd.Timestamp('1990-1-1'), pd.Timestamp('1995-1-1'),
            pd.Timestamp('2000-1-1'), pd.Timestamp('2005-1-1'),
            pd.Timestamp('2010-1-1'), pd.Timestamp('2015-1-1')]

fig, axes = plt.subplots(nrows=2, ncols=1)

ax = axes.flat[0]
ax.plot(df_spring_yearly.index, df_spring_yearly.iloc[:,(df_spring_yearly.columns>=50) & (df_spring_yearly.columns<=200)].mean(axis=1), 'k', linewidth=4)
ax.plot(df_spring_yearly.index, df_spring_yearly.iloc[:,(df_spring_yearly.columns>=300) & (df_spring_yearly.columns<=500)].mean(axis=1), 'grey', linewidth=4)

df_spring_top = df_spring_yearly.iloc[:,(df_spring_yearly.columns>=50) & (df_spring_yearly.columns<=200)].mean(axis=1)
df_spring_bot = df_spring_yearly.iloc[:,(df_spring_yearly.columns>=300) & (df_spring_yearly.columns<=500)].mean(axis=1)
ax.plot(df_spring_top.index, df_spring_top.rolling(window=5, center=True).mean(), '--k')
ax.plot(df_spring_bot.index, df_spring_bot.rolling(window=5, center=True).mean(), color='grey', linestyle='--')
ax.legend(['50-200m', '300-500m'], fontsize=15)
ax.set_ylabel(r'$\rm T_{mean}$ ($^{\circ}$C)', fontsize=15, fontweight='bold')
ax.set_xlim([pd.Timestamp('1980-01-01'), pd.Timestamp('2015-01-01')])
ax.set_ylim([-1, 7])
ax.set_xticks(xticks)
ax.xaxis.label.set_visible(False)
ax.tick_params(labelbottom='off', top='on')
#ax.text(pd.Timestamp('1980-01-01'), -.5, '   Spring')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.set_title(' Spring (3LNOPs)', fontsize=18, loc="left", fontweight='bold')

ax = axes.flat[1]
ax.plot(df_fall_yearly.index, df_fall_yearly.iloc[:,(df_fall_yearly.columns>=50) & (df_fall_yearly.columns<=200)].mean(axis=1), 'k', linewidth=4)
ax.plot(df_fall_yearly.index, df_fall_yearly.iloc[:,(df_fall_yearly.columns>=300) & (df_fall_yearly.columns<=500)].mean(axis=1), 'grey', linewidth=4)

df_fall_top = df_fall_yearly.iloc[:,(df_fall_yearly.columns>=50) & (df_fall_yearly.columns<=200)].mean(axis=1)
df_fall_bot = df_fall_yearly.iloc[:,(df_fall_yearly.columns>=300) & (df_fall_yearly.columns<=500)].mean(axis=1)
ax.plot(df_fall_top.index, df_fall_top.rolling(window=5, center=True).mean(), '--k')
ax.plot(df_fall_bot.index, df_fall_bot.rolling(window=5, center=True).mean(), color='grey', linestyle='--')
ax.set_ylabel(r'$\rm T_{mean}$ ($^{\circ}$C)', fontsize=15, fontweight='bold')
ax.set_xlabel('Year', fontsize=15, fontweight='bold')
ax.set_xlim([pd.Timestamp('1980-01-01'), pd.Timestamp('2015-01-01')])
ax.set_ylim([-1, 7])
ax.set_xticks(xticks)
ax.tick_params(labelbottom='on', top='on')
#ax.text(pd.Timestamp('1980-01-01'), -.5, '   Fall')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.set_title(' Fall (2J-3KLNO)', fontsize=18, loc="left", fontweight='bold')


fig.set_size_inches(w=12,h=12)
fig_name = 'T_timeseries.png'
fig.savefig(fig_name)

keyboard

## --- temperature plots --- ##
fig = plt.figure(1)
plt.clf()
plt.plot(df_spring_yearly.index, df_spring_yearly.iloc[:,(df_spring_yearly.columns>=50) & (df_spring_yearly.columns<=200)].mean(axis=1), 'k', linewidth=3)
plt.plot(df_spring_yearly.index, df_spring_yearly.iloc[:,(df_spring_yearly.columns>=300) & (df_spring_yearly.columns<=500)].mean(axis=1), 'grey', linewidth=3)
plt.legend(['50-200m', '300-500m'], fontsize=15)
plt.ylabel(r'$\rm T_{mean}$ ($^{\circ}$C)', fontsize=15, fontweight='bold')
plt.xlabel('Year', fontsize=15, fontweight='bold')
plt.xlim([pd.Timestamp('1975-01-01'), pd.Timestamp('2017-12-01')])
plt.ylim([-1, 8])
plt.title('Spring temperature 3LNOPs', fontsize=15, fontweight='bold')
plt.grid('on')
fig.set_size_inches(w=12,h=9)
fig_name = 'T_spring_timeseries.png'
fig.savefig(fig_name)


fig = plt.figure(2)
plt.clf()
plt.plot(df_fall_yearly.index, df_fall_yearly.iloc[:,(df_fall_yearly.columns>=50) & (df_fall_yearly.columns<=200)].mean(axis=1))
plt.plot(df_fall_yearly.index, df_fall_yearly.iloc[:,(df_fall_yearly.columns>=300) & (df_fall_yearly.columns<=500)].mean(axis=1))
plt.legend(['50-200m', '300-500m'], fontsize=15)
plt.ylabel(r'$T_{mean}$ ($^{\circ}$C)', fontsize=15, fontweight='bold')
plt.xlabel('Year', fontsize=15, fontweight='bold')
plt.xlim([pd.Timestamp('1975-01-01'), pd.Timestamp('2017-12-01')])
plt.ylim([-1, 8])
plt.title('Fall temperature 3LNOPs', fontsize=15, fontweight='bold')
plt.grid('on')
fig.set_size_inches(w=12,h=9)
fig_name = 'T_fall_timeseries.png'
fig.savefig(fig_name)



keyboard


# Some numbers for David:
dd = df_spring_top-df_spring_bot
print 'spring 1980-2015:'
print dd[(dd.index.year>=1980) & (dd.index.year<=2015)].mean()
print dd[(dd.index.year>=1980) & (dd.index.year<=2015)].std()
print ' '
print 'spring 1980-1995:'
print dd[(dd.index.year>=1980) & (dd.index.year<=1995)].mean()
print dd[(dd.index.year>=1980) & (dd.index.year<=1995)].std()
print ' '

print 'spring 1996-2015:'
print dd[(dd.index.year>=1996) & (dd.index.year<=2015)].mean()
print dd[(dd.index.year>=1996) & (dd.index.year<=2015)].std()
print ' '

dd = df_fall_top-df_fall_bot
print 'fall 1980-2015:'
print dd[(dd.index.year>=1980) & (dd.index.year<=2015)].mean()
print dd[(dd.index.year>=1980) & (dd.index.year<=2015)].std()
print ' '

print 'fall 1980-1995:'
print dd[(dd.index.year>=1980) & (dd.index.year<=1995)].mean()
print dd[(dd.index.year>=1980) & (dd.index.year<=1995)].std()
print ' '

print 'fall 1996-2015:'
print dd[(dd.index.year>=1996) & (dd.index.year<=2015)].mean()
print dd[(dd.index.year>=1996) & (dd.index.year<=2015)].std()
print ' '






spring_series = df_spring_count[97][df_spring_count[97]!=0]
fall_series = df_fall_count[97][df_fall_count[97]!=0]


fig = plt.figure(5)
width = 40      # the width of the bars
plt.clf()
plt.bar(spring_series.index, spring_series, width)
plt.bar(fall_series.index, fall_series, width)
plt.legend(['Spring', 'Fall'], fontsize=15)
plt.title('Number of cast per season')
plt.show()
