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
lonLims = [4, 18] # Khanh boxes
latLims = [58, 71]

# Box 1
lat1 = [58.23, 59.23]
lon1 = [4, 5]
x1 = lon1[0]
x2 = lon1[1]
y1 = lat1[0]
y2 = lat1[1]
polygon1 = Polygon([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
# Box2
lat2 = [69.78, 70.5]
lon2 = [17.5, 18]
xx1 = lon2[0]
xx2 = lon2[1]
yy1 = lat2[0]
yy2 = lat2[1]
polygon2 = Polygon([[xx1, yy1], [xx1, yy2], [xx2, yy2], [xx2, yy1]])


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
idx_p1 = []
idx_p2 = []

for i,coords in enumerate(df_col.index):
    lat, lon  = np.array(coords) 
    point = Point(lon, lat)
    if i%100000 == 0:
        print( str(i) + ' / ' + str(len(df_col)))

    if polygon1.contains(point):
        idx_p1.append(i)
    elif polygon2.contains(point):
        idx_p2.append(i)

print(' -> Done!')


df_p1 = df_col.iloc[idx_p1]
df_p2 = df_col.iloc[idx_p2]
df_p1.to_pickle('sst_p1_method2.pkl')
df_p2.to_pickle('sst_p2_method2.pkl')
    
#### --------- Stats per polygons -------- ####       
df_p1 = pd.read_pickle('sst_p1_method2.pkl')
df_p2 = pd.read_pickle('sst_p2_method2.pkl')

# Reset index (set time)
df_p1.reset_index(inplace=True)
df_p2.reset_index(inplace=True)
df_p1.set_index('time', inplace=True) 
df_p2.set_index('time', inplace=True) 

# Daily averages
df_p1_daily = df_p1.resample('D').mean().sst
df_p2_daily = df_p2.resample('D').mean().sst
# dropna because daily mean re-introduced them
df_p1_daily.dropna(inplace=True)
df_p2_daily.dropna(inplace=True)
df_p1_daily.to_csv('SST_Khanh_polygon1_daily.csv', float_format='%.2f')
df_p2_daily.to_csv('SST_Khanh_polygon2_daily.csv', float_format='%.2f')

# Monthly averages
df_p1_monthly = df_p1.resample('M').mean().sst
df_p2_monthly = df_p2.resample('M').mean().sst
df_p1_monthly.dropna(inplace=True)
df_p2_monthly.dropna(inplace=True)
df_p1_monthly.to_csv('SST_Khanh_polygon1_monthly.csv', float_format='%.2f')
df_p2_monthly.to_csv('SST_Khanh_polygon2_monthly.csv', float_format='%.2f')

# plot
ax = df_p1_monthly.plot()
df_p2_monthly.plot(ax=ax)
plt.legend(['poly1', 'poly2'])
