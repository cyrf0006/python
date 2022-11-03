# -*- coding: utf-8 -*-
'''
Station 27 analysis for manuscript on Wu revisited

Check also wu_stn27_clim.py for generation of s27_TSclim_bloom_3LNO.png

Frederic.Cyr@dfo-mpo.gc.ca - 2021-2022

Run in '/home/cyrf0006/research/Wu_revisited'

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import xarray as xr
import datetime
import os
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
import cmocean
import gsw

# Adjust fontsize/weight
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

## ---- Some custom parameters ---- ##
#year_clim = [1981, 2010]
year_clim = [1991, 2020]
current_year = 2020
XLIM = [datetime.date(1945, 1, 1), datetime.date(2020, 12, 31)]
french_months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
months = mdates.MonthLocator()  # every month
month_fmt = mdates.DateFormatter('%b')
Vsig = np.linspace(21,27, 13)  
V26 = np.array(26)  
REGION = '3LNO' 

# Load bloom timing
bloom = pd.read_csv('MeanTiming.csv')
bloom.set_index('Region', inplace=True)
doy = bloom.loc[REGION]
init = pd.to_datetime(1900 * 1000 + doy.meanIniation, format='%Y%j')
init_low = init - timedelta(days=doy['sdIni'])
init_high = init + timedelta(doy['sdIni'])

## 1. ---- Monthly clim ---- ##
weekly_climT = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_temperature_weekly_clim.pkl')
weekly_climS = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_salinity_weekly_clim.pkl')
# add 53rd week
weekly_climT.loc[pd.to_datetime('1901-01-07')] = weekly_climT.loc[pd.to_datetime('1900-01-07')]
weekly_climS.loc[pd.to_datetime('1901-01-07')] = weekly_climS.loc[pd.to_datetime('1900-01-07')]
# Convert to DOY
doy_index = weekly_climS.index.dayofyear.values
doy_index[-1] = doy_index[-1]+365
weekly_climT.index = doy_index
weekly_climS.index = doy_index
# Calculate density
SA = gsw.SA_from_SP(weekly_climS, weekly_climS.columns, -52, 47)
CT = gsw.CT_from_t(SA, weekly_climT, weekly_climT.columns)
SIG = gsw.sigma0(SA, CT)
weekly_climSIG = pd.DataFrame(SIG, index=weekly_climT.index, columns=weekly_climT.columns)

# plot twiny T-S
fig, ax = plt.subplots(nrows=1, ncols=1)
weekly_climT.iloc[:,4:10].mean(axis=1).plot(ax=ax, color='indianred')
weekly_climT.iloc[:,149:-1].mean(axis=1).plot(ax=ax, color='indianred', linestyle='--')
YLABT = ax.set_ylabel(r'$\rm T(^{\circ}C)$', fontsize=16, fontweight='normal')
ax2 = ax.twinx()
weekly_climS.iloc[:,4:10].mean(axis=1).plot(ax=ax2, color='steelblue')
weekly_climS.iloc[:,149:-1].mean(axis=1).plot(ax=ax2, color='steelblue', linestyle='--')
YLABS = ax2.set_ylabel(r'S', fontsize=14, fontweight='normal')
ax2.set_xlabel(r'DOY', fontsize=14, fontweight='normal')
# Load bloom timing
bloomt = pd.read_csv('MeanTimingMax.csv')
bloomt.set_index('Region', inplace=True)
doy = bloomt.loc[REGION]
init = pd.to_datetime(1900 * 1000 + doy.meanIniation, format='%Y%j')
init_low = init - timedelta(days=doy['sdIni'])
init_high = init + timedelta(doy['sdIni'])
ax.fill_between([init_low.dayofyear, init_high.dayofyear], [-2, -2], [15, 15], facecolor='green', interpolate=True , alpha=.3)
max = pd.to_datetime(1900 * 1000 + doy.meanMax, format='%Y%j')
max_low = max - timedelta(days=doy['sdMax'])
max_high = max + timedelta(doy['sdMax'])
ax.fill_between([max_low.dayofyear, max_high.dayofyear], [-2, -2], [15, 15], facecolor='red', interpolate=True , alpha=.3)
ax.set_ylim([-2, 15])
ax.legend([r'$\rm T_{5-10m}$', r'$\rm T_{150-bottom}$'])
ax2.legend([r'$\rm S_{5-10m}$', r'$\rm S_{150-bottom}$'])
ax2.text(10, 33.1, '  c', fontsize=20, fontweight='bold', horizontalalignment='left')

YLABS.set_position([636, .5])
YLABT.set_position([50, .5])


# Save Figure
fig.set_size_inches(w=8, h=6)
outfile_clim = 's27_T-S_clim.png'
fig.savefig(outfile_clim, dpi=200)
os.system('convert -trim ' + outfile_clim + ' ' + outfile_clim)

# add white border
os.system('convert -bordercolor white -border 20 ' + outfile_clim + ' ' + outfile_clim)

# montage of 3 subplots
os.system('montage  s27_TSclim_bloom_3LNO.png s27_T-S_clim.png -tile 1x2 -geometry +10+10  -background white  S27_wu.png')
