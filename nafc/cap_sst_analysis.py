'''
NOAA SST analysis for Capelin projet.
Pickle extraction is made through cap_sst_extract.py

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import xarray as xr
#from netCDF4 import Dataset
#import shapefile 
#from shapely.geometry import Point
#from shapely.geometry.polygon import Polygon
#import plotly.offline as py_off
#from plotly.graph_objs import *
import cmocean
#import azmp_utils as azu

#### --------- Stats per polygons -------- ####       
df_2J = pd.read_pickle('sst_2J_method2.pkl')
df_3K = pd.read_pickle('sst_3K_method2.pkl')
df_3L = pd.read_pickle('sst_3L_method2.pkl')

# Reset index (set time)
df_2J.reset_index(inplace=True)
df_3K.reset_index(inplace=True)
df_3L.reset_index(inplace=True)
df_2J.set_index('time', inplace=True) 
df_3K.set_index('time', inplace=True) 
df_3L.set_index('time', inplace=True) 

# Daily averages
df_2J_daily = df_2J.resample('D').mean().sst
df_3K_daily = df_3K.resample('D').mean().sst
df_3L_daily = df_3L.resample('D').mean().sst
# dropna because daily mean re-introduced them
df_2J_daily.dropna(inplace=True)
df_3K_daily.dropna(inplace=True)
df_3L_daily.dropna(inplace=True)
df_2J_daily.to_csv('SST_2J_daily.csv', float_format='%.2f')
df_3K_daily.to_csv('SST_3K_daily.csv', float_format='%.2f')
df_3L_daily.to_csv('SST_3L_daily.csv', float_format='%.2f')

# Monthly averages
df_2J_monthly = df_2J.resample('M').mean().sst
df_3K_monthly = df_3K.resample('M').mean().sst
df_3L_monthly = df_3L.resample('M').mean().sst
df_2J_monthly.dropna(inplace=True)
df_3K_monthly.dropna(inplace=True)
df_3L_monthly.dropna(inplace=True)
df_2J_monthly.to_csv('SST_2J_monthly.csv', float_format='%.2f')
df_3K_monthly.to_csv('SST_3K_monthly.csv', float_format='%.2f')
df_3L_monthly.to_csv('SST_3L_monthly.csv', float_format='%.2f')

# Feb - June averages
df_2J_FebJune = df_2J_monthly[(df_2J_monthly.index.month>=2) & (df_2J_monthly.index.month<=6)]
df_3K_FebJune = df_3K_monthly[(df_3K_monthly.index.month>=2) & (df_3K_monthly.index.month<=6)]
df_3L_FebJune = df_3L_monthly[(df_3L_monthly.index.month>=2) & (df_3L_monthly.index.month<=6)]
df_2J_FebJune = df_2J_FebJune.resample('As').mean()
df_3K_FebJune = df_3K_FebJune.resample('As').mean()
df_3L_FebJune = df_3L_FebJune.resample('As').mean()
df_2J_FebJune.index = df_2J_FebJune.index.year
df_3K_FebJune.index = df_3K_FebJune.index.year
df_3L_FebJune.index = df_3L_FebJune.index.year
df_2J_FebJune.to_csv('SST_2J_FebJune.csv', float_format='%.2f')
df_3K_FebJune.to_csv('SST_3K_FebJune.csv', float_format='%.2f')
df_3L_FebJune.to_csv('SST_3L_FebJune.csv', float_format='%.2f')

# plot
ax = df_2J_FebJune.plot()
df_3K_FebJune.plot(ax=ax)
df_3L_FebJune.plot(ax=ax)
plt.legend(['2J', '3K', '3L'])
plt.title('Feb.-June SST')
plt.ylabel(r'$\rm T_{spring} (^{\circ}C)$')
plt.xlabel('Year')

