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
import azmp_utils as azu 

# Some parameters
lonLims = [-62, -47] # 2J3KLNO
latLims = [45, 57]

## ---- Load NAFO divisions ---- ##
nafo_div = azu.get_nafo_divisions()
# Polygons
polygon2J = Polygon(zip(nafo_div['2J']['lon'], nafo_div['2J']['lat']))
polygon3K = Polygon(zip(nafo_div['3Kx']['lon'], nafo_div['3Kx']['lat']))
polygon3L = Polygon(zip(nafo_div['3L']['lon'], nafo_div['3L']['lat']))

## ---- get NL shelf ---- ##
NLshelf = azu.get_NLshelf('/home/cyrf0006/github/AZMP-NL/data/NLshelf_definition.npy')
polygonShelf = Polygon(NLshelf)

## --------- Get SSTs -------- ####
# Load ESA SST
#ds = xr.open_dataset('/home/cyrf0006/data/SST_ESACCI/ESACCI-GLO-SST-L4-REP-OBS-SST_1570728723070.nc')
# Load NOAA SST
#ds = xr.open_mfdataset('/home/cyrf0006/data/NOAA_SST/sst.day.mean*.nc', combine='by_coords')
print('load mfdataset ...')
#ds = xr.open_mfdataset('/media/cyrf0006/Seagate Backup Plus Drive/SST_NOAA/sst.day.mean.*.nc', combine='by_coords')
ds = xr.open_mfdataset('/media/cyrf0006/DevClone/data_orwell/SST_NOAA/sst.day.mean.*.nc', combine='by_coords')
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


# METHOD 2 - Faster (I hope!)
df.dropna(inplace=True)    
# remove time from index
df_col = df.reset_index(level=2) 

df_col.time = pd.to_datetime(df_col.time)    

# Find data in polygon
idx_2J = []
idx_3K = []
idx_3L = []


for i,coords in enumerate(df_col.index):
    lat, lon  = np.array(coords) 
    point = Point(lon, lat)
    if i%100000 == 0:
        print( str(i) + ' / ' + str(len(df_col)))

    if (polygon2J.contains(point)) & (polygonShelf.contains(point)) :
        idx_2J.append(i)
    elif (polygon3K.contains(point)) & (polygonShelf.contains(point)):
        idx_3K.append(i)
    elif (polygon3L.contains(point)) & (polygonShelf.contains(point)):
        idx_3L.append(i)
        
print(' -> Done!')


df_2J = df_col.iloc[idx_2J]
df_3K = df_col.iloc[idx_3K]
df_3L = df_col.iloc[idx_3L]
df_2J.to_pickle('sst_2J_method2.pkl')
df_3K.to_pickle('sst_3K_method2.pkl')
df_3L.to_pickle('sst_3L_method2.pkl')

## **see cap_sst_analysis.py for the rest...
