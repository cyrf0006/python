, #wget https://www.ncdc.noaa.gov/teleconnections/nao/data.csv# check in : /home/cyrf0006/AZMP/annual_meetings/2019

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
import pandas as pd
import os
from sys import version_info

# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)

# Download and save up-to-date  NAO index from NOAA (data.csv) if needed
url = 'https://www.ncdc.noaa.gov/teleconnections/nao/data.csv'
nao_file = '/home/cyrf0006/data/AZMP/indices/data.csv'
if os.path.exists(nao_file):
    py3 = version_info[0] > 2 #creates boolean value for test that Python major version > 2        
    response_isnt_good = True
    while response_isnt_good:
        if py3:
            response = input('Do you what to update '  + nao_file + '? [y/n]')
        else:
            response = raw_input('Do you what to update '  + nao_file + '? [y/n]')
        
        if response == 'y':
            import urllib3
            http = urllib3.PoolManager()
            r = http.request('GET', url)
            open('/home/cyrf0006/data/AZMP/indices/data.csv', 'wb').write(r.data)
            response_isnt_good = False
        elif response == 'n':
            response_isnt_good = False
        else:
            print(' -> Please answer "y" or "n"')
            
# Reload using pandas
df = pd.read_csv(nao_file, header=1)

# Set index
df = df.set_index('Date')
df.index = pd.to_datetime(df.index, format='%Y%m')

## ---- summer NAO ---- ####
df_JJA = df[(df.index.month>=6) & (df.index.month<=8)]
df_JJ = df[(df.index.month>=6) & (df.index.month<=7)]
df_JJA = df_JJA.resample('As').mean()
df_JJ = df_JJ.resample('As').mean()
df_may = df[(df.index.month==5)]
df_june = df[(df.index.month==6)]
df_july = df[(df.index.month==7)]
df_aug = df[(df.index.month==8)]
df_JJA.index = df_JJA.index.year
df_may.index = df_may.index.year
df_june.index = df_june.index.year
df_july.index = df_july.index.year
df_aug.index = df_aug.index.year

ax = df_may.rolling(5).mean().plot()
df_june.rolling(5).mean().plot(ax=ax)
df_july.rolling(5).mean().plot(ax=ax)
df_aug.rolling(5).mean().plot(ax=ax)
plt.grid()
plt.legend(['NAO May', 'NAO June', 'NAO July', 'NAO August'])



ax = df_JJA.rolling(5).mean().plot()
df_JJ.rolling(5).mean().plot(ax=ax)
plt.grid()
plt.legend(['NAO JJA', 'NAO JJ']) 



ax = df_JJ.plot(color='k', linewidth=.5, zorder=10)
ax.fill_between([df_JJ.index[0], df_JJ.index[-1]], 0, [2, 2], color='steelblue', alpha=.4)
ax.fill_between([df_JJ.index[0], df_JJ.index[-1]], 0, [-2, -2], color='indianred', alpha=.4)
df_JJ.rolling(5).mean().plot(ax=ax, color='dimgray', linewidth=3, zorder=10)
plt.grid()
plt.title('NAO June-July')
plt.ylim([-1.5, 1.5])
plt.xlabel(None)
ax.get_legend().remove()
plt.set_size_inches(w=15, h=10)
plt.savefig('NAO_JJ.png', dpi=200)

# Save csv
df_JJ.index = df_JJ.index.year
df_JJ.to_csv('NAO_June-July.csv', float_format='%.2f')
df_JJ.rolling(5).mean().to_csv('NAO_June-July_5yr_rolling_mean.csv', float_format='%.2f')
