'''
Merge wind data files downloaded from Environment Canada and merge them into a regular timeseries

Downloaded with: ~/github/shellscripts/wget_ECwinds.sh 10113 1980 2018
data in: /home/cyrf0006/data/EC/wind_argentia
process in: /home/cyrf0006/research/EC_process/wind_data

Link example: 
http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=10113&Year=1987&Month=1&Day=1&timeframe=1&submit=Download+Data

Frederic.Cyr@dfo-mpo.gc.ca

May 2019

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
import os
import subprocess

# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)


## ---- Editable parameters ---- ##
station_id = '10113'
path_data = '/home/cyrf0006/data/EC/wind_argentia/'


os.system('ls -1 ' + path_data + station_id + '*.csv > mylist.list')


## ----  Load data in a loop ---- ##
filelist = np.genfromtxt('mylist.list', dtype=str)

dfs = []
#df_labels = []
for fname in filelist:

    # find header size
    p = subprocess.Popen(['grep', '-n', 'Date', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    hdr = np.int(out.decode().split(':')[0])

    df = pd.read_csv(fname, delimiter=',', header=hdr-3, skip_blank_lines=True)

    
    df = df.rename(columns={'Date/Time':'Date'})
    # Set index (need first to swap (a,b) by (7,15))
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)

    df.drop(['Year', 'Month', 'Day', 'Time'], axis=1, inplace=True)
    
    dfs.append(df)
    #df_labels.append(box)

df_all = pd.concat(dfs, axis=0)   

# Save to .csv
df_all.to_csv('meteo_Argentia_hourly_1990-2018.csv')

# plot results
fig = plt.figure(1)
df_all['Wind Spd (km/h)'].resample('Q').mean().plot()
plt.ylabel('Wind Spd (km/h)')
plt.grid()
fig.set_size_inches(w=12,h=9)
fig_name = 'wind_argentia.png'
fig.savefig(fig_name)

