'''
Script written to process Alexis' river runoff for Gilbert Bay

data source: https://wateroffice.ec.gc.ca/download/index_e.html?results_type=historical (Date-Data Format (CSV with missing days))
-> https://wateroffice.ec.gc.ca/download/report_e.html?dt=dd&df=ddf&md=1&ext=csv

I downloaded and renamed to /home/cyrf0006/data/GilbertBay/alexis_runoff.csv

Frederic.Cyr@dfo-mpo.gc.ca

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
import pandas as pd
import os


# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)

## ---- prepare data ---- ##
# Cannot download directly becasaue I receive the content of the html file rather than the data.
data_file = '/home/cyrf0006/data/GilbertBay/alexis_runoff.csv'
df = pd.read_csv(data_file, header=1)

# Set index
df = df.set_index('Date')
df.index = pd.to_datetime(df.index, format='%Y/%m/%d')
df_disch = df[df.PARAM==1]
df_level = df[df.PARAM==2]

discharge = df_disch.Value
Wlevel = df_level.Value


## ---- basic plots  ---- ##
discharge.resample('M').mean().plot()
plt.title('Monthly discharge - Alexis River')
plt.ylabel(r'$\rm Q (m^3 s^{-1})$')
fig_name = 'Alexis_discharge.png'
plt.savefig(fig_name, dpi=300)

Wlevel.resample('M').mean().plot()
plt.title('Monthly water level - Alexis River')
plt.ylabel(r'$\rm \eta (m)$')
fig_name = 'Alexis_water_level.png'
plt.savefig(fig_name, dpi=300)


## ---- Climatologies  ---- ##
clim_year = [1981, 2010]
monthly_discharge = discharge.resample('M').mean()

# Summer mean
dseries = monthly_discharge[(monthly_discharge.index.month>=5) & (monthly_discharge.index.month<=7)]
dseries =  dseries.resample('As').mean()
clim = dseries[(dseries.index.year>=clim_year[0]) & (dseries.index.year<=clim_year[1])].mean()
std = dseries[(dseries.index.year>=clim_year[0]) & (dseries.index.year<=clim_year[1])].std()
std_anom = (dseries - clim)/std
std_anom.index = std_anom.index.year

fig = plt.figure(4)
fig.clf()
sign=std_anom>0
width = .7
n = 5 # xtick every n years
ax = std_anom.plot(kind='bar', color=sign.map({True: 'indianred', False: 'steelblue'}), width = width)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.ylabel('Mean Standardized Anomaly', weight='bold', fontsize=14)
plt.title(u'Summer (May-July) average discharge', weight='bold', fontsize=14)
plt.grid()
plt.ylim([-3,3])
fig.set_size_inches(w=15,h=7)
fig_name = 'Alexis_summer_discharge.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# freshet intensity
dseries = monthly_discharge.resample('As').max()
clim = dseries[(dseries.index.year>=clim_year[0]) & (dseries.index.year<=clim_year[1])].mean()
std = dseries[(dseries.index.year>=clim_year[0]) & (dseries.index.year<=clim_year[1])].std()
std_anom = (dseries - clim)/std
std_anom.index = std_anom.index.year

fig = plt.figure(4)
fig.clf()
sign=std_anom>0
width = .7
n = 5 # xtick every n years
ax = std_anom.plot(kind='bar', color=sign.map({True: 'indianred', False: 'steelblue'}), width = width)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.ylabel('Mean Standardized Anomaly', weight='bold', fontsize=14)
plt.title(u'Freshet monthly max', weight='bold', fontsize=14)
plt.grid()
plt.ylim([-3,3])
fig.set_size_inches(w=15,h=7)
fig_name = 'Alexis_freshet_max.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


# HERE!!!
## ---- annual curve ---- ##
#discharge['woy'] = pd.Series(discharge.index.weekofyear, index=discharge.index)
WOY = pd.Series(discharge.index.weekofyear, index=discharge.index)
df_dis = pd.concat([discharge, WOY], axis=1, keys=['discharge', 'woy'])

weekly_clim = df_dis.groupby('woy').mean()
weekly_std = df_dis.groupby('woy').std()
df_2017 = df_dis[df_dis.index.year>=2017]
weekly_2017 = df_2017.groupby('woy').mean()

fig = plt.figure(4)
fig.clf()
plt.plot(weekly_clim.index, weekly_clim.values, linewidth=3)
plt.plot(weekly_2017.index, weekly_2017.values, linewidth=3)
plt.fill_between(weekly_clim.index, np.squeeze(weekly_clim.values+weekly_std.values),np.squeeze(weekly_clim.values-weekly_std.values), facecolor='steelblue', interpolate=True , alpha=.3)
plt.ylabel(r'$\rm Q (m^3 s^{-1})$')
plt.xlabel('Week of the year')
plt.title('Alexis river 2017 discharge')
plt.xlim([0,53])
#plt.ylim([-2,18])
plt.grid()
plt.legend(['1978-2017 average', '2017'])
fig.set_size_inches(w=15,h=9)
fig_name = 'Alexis_2017_discharge.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
