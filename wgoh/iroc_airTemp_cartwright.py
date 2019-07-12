'''
AZMP reporting - Air temperature from Colbourne's Excel sheets
(script ran in /home/cyrf0006/AZMP/annual_meetings/2019)

Using data from (see ~/research/PeopleStuff/ColbourneStuff):
AZMP_AIR_TEMP_COMPOSITE_2018.xlsx
Fred built:
AZMP_AIR_TEMP_COMPOSITE_BONAVISTA.xlsx
AZMP_AIR_TEMP_COMPOSITE_CARTWRIGHT.xlsx
AZMP_AIR_TEMP_COMPOSITE_IQALUIT.xlsx
AZMP_AIR_TEMP_COMPOSITE_NUUK.xlsx
AZMP_AIR_TEMP_COMPOSITE_STJOHNS.xlsx
that are loaded and plotted here. 

Ideally, I would use directly data from EC  Homogenized Temperature: ftp://ccrp.tor.ec.gc.ca/pub/AHCCD/Homog_monthly_mean_temp.zip

but since some data are delayed or unavailable for NL stations (NUUK is in Greenland, Bonavista N/A and Cartwright stops in 2015), Eugne used to got them from :
http://climate.weather.gc.ca/prods_servs/cdn_climate_summary_e.html
http://climate.weather.gc.ca/prods_servs/cdn_climate_summary_report_e.html?intYear=2018&intMonth=2&prov=NL&dataFormat=csv&btnSubmit=Download+data

and update the Excel files.

Eventually, I could find a way to update directly from server (see azmp_airTemp.py).

Frederic.Cyr@dfo-mpo.gc.ca - February 2019

'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gsw
from seawater import extras as swx
import matplotlib.dates as mdates
from scipy.interpolate import griddata
import os

# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)

clim_year = [1981, 2010]
current_year = 2018

## ----  Prepare the data ---- ##
# load from Excel sheets
df = pd.read_excel('/home/cyrf0006/research/PeopleStuff/ColbourneStuff/AZMP_AIR_TEMP_COMPOSITE_CARTWRIGHT.xlsx', header=0)

# Rename columns
col_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df.columns = col_names

# Stack months under Years (pretty cool!)
df = df.stack() 

# Transform to a series with values based the 15th of each month (had to convert years to string)
df.index = pd.to_datetime('15-' + df.index.get_level_values(1) + '-' + df.index.get_level_values(0).values.astype(np.str))

## ---- Annual anomalies ---- ##
df_annual = df.resample('As').mean()
#df_annual = df_annual[df_annual.index.year>=1950]
clim = df_annual[(df_annual.index.year>=clim_year[0]) & (df_annual.index.year<=clim_year[1])].mean()
std = df_annual[(df_annual.index.year>=clim_year[0]) & (df_annual.index.year<=clim_year[1])].std()
anom_annual = df_annual - clim
anom_annual.index = anom_annual.index.year
std_anom_annual = (df_annual - clim)/std
std_anom_annual.index = std_anom_annual.index.year

# Save
anom_annual.to_csv('iroq_airT_Cartwright.csv', sep=',', float_format='%.2f')

# Compare plots
df_orig = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/2018_update/Newf_Cartwright_Air_Timeseries.csv', header=11)
df_old = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/2018_update/Newf_Cartwright_Air_Timeseries.csv', header=11)
df_orig.set_index(['Decimal Year'], inplace=True)
df_old.set_index(['Decimal Year'], inplace=True)

# Compare plots
ax = df_orig['Temperature Anomaly \xb0C'].plot()
df_old['Temperature Anomaly \xb0C'].plot(ax=ax)
ax.set_ylabel(r'$\rm Temperature anomaly (^{\circ}C)$', fontWeight = 'bold')
plt.legend(['New way', 'Old way'])
plt.savefig('compare_AirTemp.png', dpi=150)
plt.close()
