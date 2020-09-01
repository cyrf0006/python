# data source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=71800&vid=676

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

# Some parameters
#lonLims = [-66.5, -47] # Maximum MPA zone
#latLims = [40, 62]
lonLims = [-61, -47] # LFAs
latLims = [43, 53]

lonLims = [-61, -51] # LFAs truncated
latLims = [46, 53]
METHOD=2

## --------- Get SSTs -------- ####
# Load ESA SST
#ds = xr.open_dataset('/home/cyrf0006/data/SST_ESACCI/ESACCI-GLO-SST-L4-REP-OBS-SST_1570728723070.nc')
# Load NOAA SST
#ds = xr.open_mfdataset('/home/cyrf0006/data/NOAA_SST/sst.day.mean*.nc', combine='by_coords')
print('load mfdataset ...')
ds = xr.open_mfdataset('/media/cyrf0006/Seagate Backup Plus Drive/SST_NOAA/sst.day.mean.*.nc', combine='by_coords')
print(' -> Done!')

# update longitude (WOW!)
print('Subset data ...')
ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
# Selection of a subset region
ds = ds.where((ds.lon>lonLims[0]) & (ds.lon<lonLims[1]), drop=True) # original one
ds = ds.where((ds.lat>latLims[0]) & (ds.lat<latLims[1]), drop=True)
print(' -> Done!')

# to DataFrame
print('to DataFrame...')
df = ds.to_dataframe()
print(' -> Done!')

## --------- GetShapefiles -------- ####
myshp = open('/home/cyrf0006/research/PeopleStuff/CoughlanStuff/LobsterFishingAreas.shp', 'rb')
mydbf = open('/home/cyrf0006/research/PeopleStuff/CoughlanStuff/LobsterFishingAreas.dbf', 'rb')
r = shapefile.Reader(shp=myshp, dbf=mydbf)
records = r.records()
shapes = r.shapes()
# Fill dictionary with NAFO divisions
lobster_area = {}
for idx, rec in enumerate(records):
    if rec[-1] == '':
        continue
    else:
        lobster_area[rec[-1]] = np.array(shapes[idx].points)

#lfa = ['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13A', '13B', '14A', '14B', '14C']                        
polygon3 = Polygon(lobster_area['3'])
polygon4 = Polygon(lobster_area['4'])
polygon5 = Polygon(lobster_area['5'])
polygon6 = Polygon(lobster_area['6'])
polygon7 = Polygon(lobster_area['7'])
polygon8 = Polygon(lobster_area['8'])
polygon9 = Polygon(lobster_area['9'])
polygon10 = Polygon(lobster_area['10'])
polygon11 = Polygon(lobster_area['11'])
polygon12 = Polygon(lobster_area['12'])
polygon13A = Polygon(lobster_area['13A'])
polygon13B = Polygon(lobster_area['13B'])
polygon14A = Polygon(lobster_area['14A'])
polygon14B = Polygon(lobster_area['14B'])
polygon14C = Polygon(lobster_area['14C'])


#### --------- Extract SST in polygons -------- ####

print('Select polygons...')

if METHOD == 1:
# METHOD 1 - Works but long
    # swap level to column dataframe
    df_col = df.swaplevel(i='time', j='lat')    
    df_col.dropna(inplace=True)

    # Transfer index to datetime
    df_col = df_col.reset_index()
    df_col.set_index('time', inplace=True)
    df_col.index = pd.to_datetime(df_col.index)

    # drop winter
    #df_col = df_col[(df_col.index.month>=4) & (df_col.index.month<=10)]

    # Keep only sst>12degC
    df_col = df_col[df_col.sst>=8]

    # Find data in polygon
    idx_W = []
    idx_S = []
    idx_NE = []
    idx_A = []

    for i,sst in enumerate(df_col.sst.values):
        point = Point(df_col.lon[i], df_col.lat[i])

        if i%100000 == 0:
            print( str(i) + ' / ' + str(len(df_col)))

        if polygon3.contains(point) | polygon4.contains(point) | polygon5.contains(point) | polygon6.contains(point):
            idx_NE.append(i)
        elif polygon7.contains(point) | polygon8.contains(point) | polygon9.contains(point) | polygon10.contains(point):
            idx_A.append(i)
        elif polygon11.contains(point) | polygon12.contains(point):
            idx_S.append(i)
        elif polygon13A.contains(point) | polygon13B.contains(point) | polygon14A.contains(point) | polygon14B.contains(point) | polygon14C.contains(point):
            idx_W.append(i)
    print(' -> Done!')

    
    # New DataFrame per region            
    df_W = df_col.iloc[idx_W]
    df_S = df_col.iloc[idx_S]
    df_NE = df_col.iloc[idx_NE]
    df_A = df_col.iloc[idx_A]
    df_W.to_pickle('sst_west_coast_method1.pkl')
    df_NE.to_pickle('sst_northeast_coast_method1.pkl')
    df_S.to_pickle('sst_south_coast_method1.pkl')
    df_A.to_pickle('sst_avalon_method1.pkl')

    
elif METHOD == 2:
    # METHOD 2 - Faster (I hope!)
    df.dropna(inplace=True)    
    # remove time from index
    df_col = df.reset_index(level=2) 

    df_col.time = pd.to_datetime(df_col.time)    
    
    # Find data in polygon
    idx_W = []
    idx_S = []
    idx_NE = []
    idx_A = []

    for i,coords in enumerate(df_col.index):
        lat, lon  = np.array(coords) 
        point = Point(lon, lat)
        if i%100000 == 0:
            print( str(i) + ' / ' + str(len(df_col)))

        if polygon3.contains(point) | polygon4.contains(point) | polygon5.contains(point) | polygon6.contains(point):
            idx_NE.append(i)
        elif polygon7.contains(point) | polygon8.contains(point) | polygon9.contains(point) | polygon10.contains(point):
            idx_A.append(i)
        elif polygon11.contains(point) | polygon12.contains(point):
            idx_S.append(i)
        elif polygon13A.contains(point) | polygon13B.contains(point) | polygon14A.contains(point) | polygon14B.contains(point) | polygon14C.contains(point):
            idx_W.append(i)
            
    print(' -> Done!')


    df_W = df_col.iloc[idx_W]
    df_S = df_col.iloc[idx_S]
    df_NE = df_col.iloc[idx_NE]
    df_A = df_col.iloc[idx_A]
    df_W.to_pickle('sst_west_coast_method2.pkl')
    df_NE.to_pickle('sst_northeast_coast_method2.pkl')
    df_S.to_pickle('sst_south_coast_method2.pkl')
    df_A.to_pickle('sst_avalon_method2.pkl')    

    
#### --------- Stats per polygons -------- ####       


if METHOD == 1:
    df_W = pd.read_pickle('sst08_west_coast.pkl')
    df_S = pd.read_pickle('sst08_south_coast.pkl')
    df_NE = pd.read_pickle('sst08_northeast_coast.pkl')
    df_A = pd.read_pickle('sst08_avalon.pkl')

    # Daily averages
    df_W_daily = df_W.resample('D').mean().sst
    df_NE_daily = df_S.resample('D').mean().sst
    df_S_daily = df_NE.resample('D').mean().sst
    df_A_daily = df_A.resample('D').mean().sst

    # dropna because daily mean re-introduced them
    df_W_daily.dropna(inplace=True)
    df_NE_daily.dropna(inplace=True)
    df_S_daily.dropna(inplace=True)
    df_A_daily.dropna(inplace=True)

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

    # plot
    ax = days_above_W.plot()
    days_above_NE.plot(ax=ax)
    days_above_S.plot(ax=ax)
    days_above_A.plot(ax=ax)
    plt.legend(['West Coast', 'NE Coast', 'South Coast', 'Avalon'])

elif METHOD == 2:

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

    # Daily averages
    df_W_daily = df_W.resample('D').mean().sst
    df_NE_daily = df_S.resample('D').mean().sst
    df_S_daily = df_NE.resample('D').mean().sst
    df_A_daily = df_A.resample('D').mean().sst

    # dropna because daily mean re-introduced them
    df_W_daily.dropna(inplace=True)
    df_NE_daily.dropna(inplace=True)
    df_S_daily.dropna(inplace=True)
    df_A_daily.dropna(inplace=True)

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

    # plot
    ax = days_above_W.plot()
    days_above_NE.plot(ax=ax)
    days_above_S.plot(ax=ax)
    days_above_A.plot(ax=ax)
    plt.legend(['West Coast', 'NE Coast', 'South Coast', 'Avalon'])
