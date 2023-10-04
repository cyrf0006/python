'''
PLot Cartwright air temperature used for regional IROC presentation in the WGOH meeting

Frederic.Cyr@dfo-mpo.gc.ca - March 2023


'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
from scipy.interpolate import griddata

# Adjust fontsize/weight
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

   
clim_year = [1991, 2020]
current_year = 2022

## ---- Monthly anomalies for current year ---- ##
air_monthly = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/airTemp/airT_monthly.pkl')
df = air_monthly[['Cartwright']]
df_clim_period = df[(df.index.year>=clim_year[0]) & (df.index.year<=clim_year[1])]
df_monthly_stack = df_clim_period.groupby([(df_clim_period.index.year),(df_clim_period.index.month)]).mean()
df_monthly_clim = df_monthly_stack.groupby(level=1).mean()
df_monthly_std = df_monthly_stack.groupby(level=1).std()

df_current_year = df[df.index.year==current_year]
year_index = df_current_year.index # backup index
df_current_year.index=df_monthly_std.index # reset index
anom = df_current_year - df_monthly_clim
std_anom = (df_current_year - df_monthly_clim)/df_monthly_std
#std_anom.index = year_index.month # replace index
std_anom.index = year_index.strftime('%b') # replace index (by text)
anom.index = year_index.strftime('%b') # replace index (by text)


## ---- Annual anomalies (load fromm climate index) ---- ##
air = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/airTemp/airT_std_anom.pkl')
air.index.name='Year'
air = air[['Cartwright']]
air = air.loc[air.index>=1930]


## ---- Plots ---- ##

## Monthly Barplot
ax = anom.plot(kind='bar', color='slategray', zorder=10)
plt.grid('on')
ax.set_ylabel(r'[$^{\circ}$C]')
ax.set_title(np.str(current_year) + ' Air temperature anomalies')
plt.ylim([-4, 4])
fig = ax.get_figure()
fig.set_size_inches(w=9,h=6)
fig_name = 'Cartwright_' + str(current_year) + '.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## Annual Barplot
df1 = air[air>0]
df2 = air[air<0]

fig = plt.figure(1)
fig.clf()
width = .9
p1 = plt.bar(df1.index, np.squeeze(df1.values), width, alpha=0.8, color='indianred', zorder=10)
p2 = plt.bar(df2.index, np.squeeze(df2.values), width, bottom=0, alpha=0.8, color='steelblue', zorder=10)
plt.ylabel('Standardized anomaly')
plt.title('Cartwright - Annual air temperature')
ticks = plt.gca().xaxis.get_ticklocs()
plt.fill_between([ticks[0]-1, ticks[-1]+1], [-.5, -.5], [.5, .5], facecolor='gray', alpha=.2)
plt.xlim([1932, 2023])
plt.ylim([-3.2, 3.2])
plt.grid()
# Save Figure
fig.set_size_inches(w=15,h=9)
fig_name = 'Cartwright_annual_stdanom.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
