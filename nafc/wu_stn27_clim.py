# -*- coding: utf-8 -*-
'''
Station 27 analysis for manuscript on Wu revisited

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

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

## ---- Some custom parameters ---- ##
#year_clim = [1981, 2010]
year_clim = [1991, 2020]
current_year = 2020
XLIM = [datetime.date(1945, 1, 1), datetime.date(2020, 12, 31)]
#french_months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
months = mdates.MonthLocator()  # every month
month_fmt = mdates.DateFormatter('%b')
Vsig = np.linspace(21,27, 13)  
V26 = np.array(26)  
REGION = '3LNO' 

# Load bloom initiation timing
bloom = pd.read_csv('MeanTiming.csv')
bloom.set_index('Region', inplace=True)
doy = bloom.loc[REGION]
init = pd.to_datetime(1900 * 1000 + doy.meanIniation, format='%Y%j')
init_low = init - timedelta(days=doy['sdIni'])
init_high = init + timedelta(doy['sdIni'])
# Load bloom max timing
bloomt = pd.read_csv('MeanTimingMax.csv')
bloomt.set_index('Region', inplace=True)
doy_max = bloomt.loc[REGION]
max = pd.to_datetime(1900 * 1000 + doy_max.meanMax, format='%Y%j')
max_low = max - timedelta(days=doy_max['sdMax'])
max_high = max + timedelta(doy_max['sdMax'])

## 1. ---- Monthly clim ---- ##
weekly_climT = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_temperature_weekly_clim.pkl')
weekly_climS = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_salinity_weekly_clim.pkl')
# add 53rd week
weekly_climT.loc[pd.to_datetime('1901-01-07')] = weekly_climT.loc[pd.to_datetime('1900-01-07')]
weekly_climS.loc[pd.to_datetime('1901-01-07')] = weekly_climS.loc[pd.to_datetime('1900-01-07')]
# Calculate density
SA = gsw.SA_from_SP(weekly_climS, weekly_climS.columns, -52, 47)
CT = gsw.CT_from_t(SA, weekly_climT, weekly_climT.columns)
SIG = gsw.sigma0(SA, CT)
weekly_climSIG = pd.DataFrame(SIG, index=weekly_climT.index, columns=weekly_climT.columns)

# plot T
fig, ax = plt.subplots(nrows=1, ncols=1)
CMAP = cmocean.tools.lighten(cmocean.cm.thermal, .9)
V = np.arange(-2, 14, 1)
c = plt.contourf(weekly_climT.index, weekly_climT.columns, weekly_climT.values.T, V, extend='max', cmap=CMAP)
cc = plt.contour(weekly_climSIG.index, weekly_climSIG.columns, weekly_climSIG.values.T, Vsig, colors='dimgray')
cc26 = plt.contour(weekly_climSIG.index, weekly_climSIG.columns, weekly_climSIG.values.T, [26,], colors='dimgray', linewidths=3)
plt.clabel(cc, inline=1, fontsize=10, colors='dimgray', fmt='%1.1f')
plt.ylim([0, 175])
plt.ylabel('Depth (m)', fontsize=14)
ax.invert_yaxis()
# add bloom timing
plt.fill_between([init_low, init_high], [0, 0], [175, 175], facecolor='white', interpolate=True , alpha=.5)
plt.fill_between([max_low, max_high], [0, 0], [175, 175], facecolor='white', interpolate=True , alpha=.5)
# format date ticks
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
# format the ticks
#ax.xaxis.set_major_locator(months)
#ax.xaxis.set_major_formatter(month_fmt)
cax = fig.add_axes([0.91, .15, 0.01, 0.7])
cb = plt.colorbar(c, cax=cax, orientation='vertical')
ccc = ax.contour(weekly_climT.index, weekly_climT.columns, weekly_climT.T, [0,], colors='k', linewidths=3)
cb.set_label(r'$\rm T(^{\circ}C)$', fontsize=14, fontweight='normal')
ax.xaxis.label.set_visible(False)
ax.text(weekly_climS.index[0], 15, '  a', fontsize=20, fontweight='bold', horizontalalignment='left', color='w')
# Save Figure
fig.set_size_inches(w=8, h=6)
outfile_clim = 's27_temp_clim.png'
fig.savefig(outfile_clim, dpi=200)
os.system('convert -trim ' + outfile_clim + ' ' + outfile_clim)


# plot S
fig, ax = plt.subplots(nrows=1, ncols=1)
V = np.arange(30, 33.5, .25)
CMAP = cmocean.cm.haline
c = plt.contourf(weekly_climS.index, weekly_climS.columns, weekly_climS.values.T, V, extend='both', cmap=CMAP)
cc = plt.contour(weekly_climSIG.index, weekly_climSIG.columns, weekly_climSIG.values.T, Vsig, colors='dimgray')
cc26 = plt.contour(weekly_climSIG.index, weekly_climSIG.columns, weekly_climSIG.values.T, [26,], colors='dimgray', linewidths=3)
plt.clabel(cc, inline=1, fontsize=10, colors='dimgray', fmt='%1.1f')
plt.ylim([0, 175])
plt.ylabel('Depth (m)', fontsize=14)
plt.ylim([0, 175])
ax.invert_yaxis()
# add bloom timing
plt.fill_between([init_low, init_high], [0, 0], [175, 175], facecolor='white', interpolate=True , alpha=.5)
plt.fill_between([max_low, max_high], [0, 0], [175, 175], facecolor='white', interpolate=True , alpha=.5)
# format date ticks
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
# format the ticks
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(month_fmt)
cax = fig.add_axes([0.91, .15, 0.01, 0.7])
cb = plt.colorbar(c, cax=cax, orientation='vertical')
ccc = ax.contour(weekly_climS.index, weekly_climS.columns, weekly_climS.T, [0,], colors='k', linewidths=3)
cb.set_label(r'S', fontsize=14, fontweight='normal')
ax.text(weekly_climS.index[0], 15, '  b', fontsize=20, fontweight='bold', horizontalalignment='left', color='w')
# Save Figure
fig.set_size_inches(w=8, h=6)
outfile_clim = 's27_sal_clim.png'
fig.savefig(outfile_clim, dpi=200)
os.system('convert -trim ' + outfile_clim + ' ' + outfile_clim)

# Convert to subplot
os.system('montage  s27_temp_clim.png s27_sal_clim.png -tile 1x2 -geometry +10+10  -background white  s27_TSclim_bloom_' + REGION + '.png')
os.system('rm s27_temp_clim.png  s27_sal_clim.png')

