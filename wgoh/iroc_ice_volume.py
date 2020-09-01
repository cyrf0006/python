'''
A Script to calculate ice volume on Newfoundland and Labrador shelves for ICES-WGOH IROC
This method was only use since year 2019 on.
Before, the ice cover provided by Ingrid Peterson to Eugene Colbourne was used.

There is an equivalent for maximum ice volume.

'''
# A first test to read Excel nutrient file and export to Pandas.

# Check in:
#  /home/cyrf0006/research/AZMP_database/biochem

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from scipy import stats
import os

# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)

clim_year = [1981, 2010]
years = [1969, 2019]
width = 0.7


## ----  Load data a make index ---- ##
# Peter's volume
df_vol = pd.read_csv('/home/cyrf0006/data/seaIce_IML/IceVolumeSeasonalAverage.dat', delimiter=r"\s+", error_bad_lines=False, names=['Year', 'GSL', 'GSL_days', 'LAB', 'LAB_days', 'NFLD', 'NFLD_days', 'NL', 'NL_days'])
df_vol = df_vol.set_index('Year', drop=True)

df = pd.concat([df_vol.LAB, df_vol.NFLD], axis=1, keys=['Lab', 'Nfl'])
df = df[(df.index>=years[0]) & (df.index<=years[1])]

df.to_csv('iroc_seaice.csv', sep=',', float_format='%.1f')



## ---- plot volumes ---- ##
ind = df.index
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, df.Lab.values, width, label='Labrador shelf', color='steelblue')
rects2 = ax.bar(ind + width/2, df.Nfl.values, width, label='Newfoundland shelf', color='lightblue')
ax.legend()
ax.yaxis.grid() # horizontal lines
ax.xaxis.grid() # horizontal lines
plt.ylabel(r'Ice volume ($\rm km^3$)')

## ---- std anom ---- ##
df_clim = df[(df.index>=clim_year[0]) & (df.index<=clim_year[1])]
df_std_anom = (df - df_clim.mean()) / df_clim.std()
