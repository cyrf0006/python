, #wget https://www.ncdc.noaa.gov/teleconnections/nao/data.csv# check in : /home/cyrf0006/AZMP/annual_meetings/2019

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
import pandas as pd
import os
from sys import version_info

## load upwelling
df_u = pd.read_csv('upwells_BristolsHope.csv')
df_u.startDate = pd.to_datetime(df_u.startDate)
df_u.index = df_u.startDate
df_u['doy'] = df_u.index.dayofyear   
df_u = df_u.doy
df_u.index = df_u.index.year

## load spawning
df_s = pd.read_excel('Peak spawning Bellevue and Bryants Cove.xlsx') 
df_s.set_index('Year', inplace=True) 
df_s = df_s[' Bryants'] 
df_s.replace('NS', np.nan, inplace=True)
df_s.dropna(inplace=True)

ax = df_s.plot(style='.')
for i, yy in enumerate(df_u.index):
    plt.plot([yy, yy], [df_u.iloc[i].doy, df_u.iloc[i].doy+df_u.iloc[i].duration], color='steelblue')
df_s.plot(ax=ax, style='o')

# Save csv
df_JJ.index = df_JJ.index.year
df_JJ.to_csv('NAO_June-July.csv', float_format='%.2f')
df_JJ.rolling(5).mean().to_csv('NAO_June-July_5yr_rolling_mean.csv', float_format='%.2f')
