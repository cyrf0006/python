'''
CIL area

This script is still in progress...


'''
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np

df = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/2018_update/Newf_SeaIce_Timeseries.csv', header=12)
df.set_index(['Decimal Year'], inplace=True)
df.index = pd.to_datetime(df.index, format='%Y')


clim_year = [1981, 2010]

clim = df[(df.index.year>=clim_year[0]) & (df.index.year<=clim_year[1])].mean()
std = df[(df.index.year>=clim_year[0]) & (df.index.year<=clim_year[1])].std()
anom = df - clim
anom.index = anom.index.year
std_anom = (df - clim)/std
std_anom.index = std_anom.index.year
