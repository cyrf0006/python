import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime


ds = xr.open_dataset('cmems_mod_glo_bgc_my_0.25_P1M-m_1664910381648.nc')

# Selection of a subset region
ds = ds.where((ds.longitude>=-65) & (ds.longitude<=-47), drop=True) # original one
ds = ds.where((ds.latitude>=47) & (ds.latitude<=65), drop=True)

# Select depth range
ds = ds.where((ds.depth<=100), drop=True) # original one

# Annual mean
#ds = ds.resample('As', dim='time', how='mean')

# mean depth and space
ds = ds.mean(dim=['depth', 'latitude', 'longitude']

# To pandas
pp = ds['nppv'].to_dataframe()
phy = ds['phyc'].to_dataframe()
pp = ds['nppv'].to_dataframe()

pp_annual = pp.resample('As').mean()
pp_annual.index = pp_annual.index.year
pp_annual.to_csv('cmes_PP.csv')
