'''
This is a test with xarray, see if I can manipulate a lot of netCDF files.
Try this script in /home/cyrf0006/research/AZMP_database
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import os

#import water_masses as wm

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)


## ---- Attempt to speed up ---- ##
## if os.path.exists("./ds_avalon.nc"):
##     print('Loading pickled file historical_temp_1948-2017.pkl')
##     #    df_temp = pd.read_pickle('historical_temp_1948-2017.pkl') # 
##     #    df_sal = pd.read_pickle('historical_sal_1948-2017.pkl') #
##     ds = xr.open_dataset('ds_avalon.nc')
    
## else:
##     print('historical_temp_1948-2017.pkl doesnt exist, will generate')   
##     #This is a dataset
##     ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/*.nc')

##     # Select a depth range
##     ds = ds.sel(level=ds['level']<500)
##     ds = ds.sel(level=ds['level']>10)

##     # Selection of a subset region
##     ds = ds.where((ds.longitude>-55) & (ds.longitude<-50), drop=True)
##     #ds = ds.where((ds.latitude>50) & (ds.latitude<55), drop=True)
##     ds = ds.where((ds.latitude>45) & (ds.latitude<50), drop=True)

##     # Sort by time dimension
##     ds = ds.sortby('time')
##     ds.to_netcdf('ds_avalon.nc')
##     print(' -> Done!')

ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/*.nc')

# Select a depth range
ds = ds.sel(level=ds['level']<500)
ds = ds.sel(level=ds['level']>10)

# Selection of a subset region
ds = ds.where((ds.longitude>-55) & (ds.longitude<-50), drop=True)
#ds = ds.where((ds.latitude>50) & (ds.latitude<55), drop=True)
ds = ds.where((ds.latitude>45) & (ds.latitude<50), drop=True)

# Sort by time dimension
ds = ds.sortby('time')

# Monthly average (This takes a lot of time given the sorting above)
#ds_monthly = ds.resample('M', dim='time', how='mean') #deprecated
ds_monthly = ds.resample(time="M").mean('time') 


# To Pandas Dataframe
da_temp = ds_monthly['temperature']
df_temp = da_temp.to_pandas()
da_sal = ds_monthly['salinity']
df_sal = da_sal.to_pandas()

## # Pickle data
## df_temp.to_pickle('historical_temp_1948-2017.pkl') # 
## df_sal.to_pickle('historical_sal_1948-2017.pkl') # 
## print(' -> Done!')


# Compute climatology
df_temp_may = df_temp.loc[df_temp.index.month==5]
df_temp_june = df_temp.loc[df_temp.index.month==6]
df_temp_july = df_temp.loc[df_temp.index.month==7]
df_concat = pd.concat((df_temp_may, df_temp_june, df_temp_july))
df_all = df_concat.resample('A').mean()
df_all.to_pickle('temp_summer_1948-2017.pkl') # 


## --- CIL core --- ## 
fig = plt.figure(1)
plt.clf()
plt.plot(df_temp_may.index, df_temp_may.min(axis=1), '-')
plt.plot(df_temp_june.index, df_temp_june.min(axis=1), '-')
plt.plot(df_temp_july.index, df_temp_july.min(axis=1), '-')
plt.plot(df_all.index, df_all.min(axis=1).rolling(5,center=True).mean(), 'k-', linewidth=3)

plt.legend(['May', 'June', 'July', '5y moving ave'], fontsize=15)
plt.ylabel(r'$T_{min}$ in monthly mean profile ($^{\circ}$C)', fontsize=15, fontweight='bold')
plt.xlabel('Year', fontsize=15, fontweight='bold')
plt.title('CIL core temperature', fontsize=15, fontweight='bold')
plt.xlim([pd.Timestamp('1948-01-01'), pd.Timestamp('2017-01-01')])
plt.ylim([-2, 1])
plt.xticks(pd.date_range('1950-01-01', periods=7, freq='10Y'))
plt.grid('on')


fig.set_size_inches(w=9,h=6)
fig_name = 'CIL_core_1948-2017.png'
fig.set_dpi(300)
fig.savefig(fig_name)




## --- No. of cast per year --- ##
years = np.arange(1912, 2017)
time_series = ds.time.to_pandas()
month05 = time_series.loc[time_series.index.month==5]
month06 = time_series.loc[time_series.index.month==6]
month07 = time_series.loc[time_series.index.month==7]
year_count_05 = np.zeros(years.shape)
year_count_06 = np.zeros(years.shape)
year_count_07 = np.zeros(years.shape)
year_count = np.zeros(years.shape)

for idx, year in enumerate(years):
    year_count_05[idx] = np.size(np.where(month05.index.year==year))
    year_count_06[idx] = np.size(np.where(month06.index.year==year))
    year_count_07[idx] = np.size(np.where(month07.index.year==year))
    year_count[idx] = np.size(np.where(time_series.index.year==year))

fig = plt.figure(2)
plt.clf()
plt.bar(years, year_count, align='center')
plt.ylabel('Number of stations', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.annotate('1992;\nground fish\nmoratorium', xy=(1991, 2000), xytext=(1940, 2000),
            arrowprops=dict(facecolor='black', width=1, shrink=0.05), fontsize=14
            )
plt.annotate('1999;\nAZMP', xy=(1999, 750), xytext=(1999, 1800),
            arrowprops=dict(facecolor='black', width=1, shrink=0.05), fontsize=14
            )

fig.set_size_inches(w=9,h=6)
fig_name = 'no_profile_histo.png'
fig.set_dpi(300)
fig.savefig(fig_name)


# T-S seasonal timeseries
df_temp_season = df_temp.resample('3M').mean()
df_sal_season = df_sal.resample('3M').mean()
df_sal_year = df_sal.resample('A').mean()

T = np.array(df_temp_season)
TT = np.reshape(T, T.size)
S = np.array(df_sal_season)

Tbin = np.arange(-2,10)
Slist = []
for idx in np.arange(0, T.shape[0]):
    TVec = T[idx,:]
    SVec = T[idx,:]  
    digitized = np.digitize(TVec, Tbin) #<- this is awesome! 
    Slist.append([SVec[digitized == i].mean() for i in range(0, len(Tbin))])


plt.figure(2)
plt.clf()
#plt.contourf(df_temp.index, Tbin, np.array(Slist).T, 20)  
plt.pcolormesh(df_temp_season.index, Tbin, np.array(Slist).T, vmin=-2, vmax=10)  

## Tcontour-seasonal
fig = plt.figure(3)
plt.clf()
v = np.arange(-2,10)
plt.contourf(df_temp_season.index, df_temp_season.columns, df_temp_season.T, v, cmap=plt.cm.RdBu_r)  
plt.xlim([pd.Timestamp('1948-01-01'), pd.Timestamp('2017-01-01')])
plt.xticks(pd.date_range('1950-01-01', periods=7, freq='10Y'))
plt.gca().invert_yaxis()
plt.ylabel('Depth (m)', fontsize=14, fontweight='bold')
plt.xlabel('years', fontsize=14, fontweight='bold')
cb = plt.colorbar()
plt.title(r'$\rm T(^{\circ}C)$', fontsize=14, fontweight='bold')
fig.set_size_inches(w=12,h=9)
fig_name = 'Tcontours_1950-2016.png'
fig.savefig(fig_name)


## Scontour-seasonal
fig = plt.figure(4)
plt.clf()
v = np.linspace(30, 35, 20)
plt.contourf(df_sal_season.index, df_sal_season.columns, df_sal_season.T, v, cmap=plt.cm.RdBu_r)  
plt.xlim([pd.Timestamp('1950-01-01'), pd.Timestamp('2017-12-01')])
plt.gca().invert_yaxis()
plt.ylabel('Depth (m)', fontsize=14, fontweight='bold')
plt.xlabel('years', fontsize=14, fontweight='bold')
cb = plt.colorbar()
plt.title(r'$\rm S$', fontsize=14, fontweight='bold')
fig.set_size_inches(w=12,h=9)
fig_name = 'Scontours_1950-2016.png'
fig.savefig(fig_name)



## Salinity timeseries
fig = plt.figure(5)
plt.clf()
plt.plot(df_sal_season.index, df_sal_season.iloc[:,df_sal_season.columns<200].mean(axis=1), '-')
#plt.plot(df_sal_year.index, df_sal_year.iloc[:,df_sal_year.columns<200].mean(axis=1), '-r')
A = df_sal_season.iloc[:,df_sal_season.columns<200].mean(axis=1)
#A = df_sal_season.iloc[:,(df_sal_season.columns>75) & (df_sal_season.columns<150)].mean(axis=1)
plt.plot(df_sal_season.index, A.rolling(20, center=True).mean(), 'r')
df_sal_season.to_pickle('salinity_1948-2017.pkl')

## plt.plot(df_sal_season.index, df_sal_season.iloc[:,((df_sal_season.columns>150)&(df_sal_season.columns<300))].mean(axis=1), '-')
## #plt.plot(df_sal_year.index, df_sal_year.iloc[:,((df_sal_season.columns>200)&(df_sal_season.columns<300))].mean(axis=1), '-r')
## B =  df_sal_season.iloc[:,((df_sal_season.columns>150)&(df_sal_season.columns<300))].mean(axis=1)
## plt.plot(df_sal_season.index, B.rolling(24).mean(), 'r')


plt.ylabel(r'$\rm <S>_{0-200m}$', fontsize=15, fontweight='bold')
plt.xlabel('Year', fontsize=15, fontweight='bold')
#plt.title(r'$<S>_{0-300m}$', fontsize=15, fontweight='bold')
plt.xlim([pd.Timestamp('1948-01-01'), pd.Timestamp('2017-01-01')])
plt.xticks(pd.date_range('1950-01-01', periods=7, freq='10Y'))
plt.ylim([31.75, 33.50])
plt.grid('on')

fig.set_size_inches(w=9,h=6)
fig_name = 'sal_0-200m_1948-2017.png'
fig.set_dpi(300)
fig.savefig(fig_name)




## T and S

fig = plt.figure(6)
plt.clf()
ax0 = plt.subplot(1,1,1)
ax0.plot(df_all.index, df_all.min(axis=1).rolling(5,center=True).mean(), linewidth=3)
ax0.set_ylabel(r'$T_{min}$ in monthly mean profile ($^{\circ}$C)', fontsize=15, fontweight='bold')
ax0.set_xlabel('Year', fontsize=15, fontweight='bold')
ax0.set_xlim(pd.Timestamp('1948-01-01'), pd.Timestamp('2017-01-01'))
ax0.set_ylim(-1.8, .2)
#ax0.set_xticks([pd.date_range('1950-01-01', periods=7, freq='10Y')])
ax0.grid('on')
ax0.tick_params('y', colors='k')
ax0.set_xticks([pd.date_range('1950-01-01', periods=7, freq='10Y')])
plt.xticks(pd.date_range('1950-01-01', periods=7, freq='10Y'))



ax01 = ax0.twinx() # WAVES
ax01.plot(df_sal_season.index, A.rolling(20, center=True).mean(), 'r', linewidth=3)
ax01.set_xlim([pd.Timestamp('1948-01-01'), pd.Timestamp('2017-01-01')])
#ax01.tick_params(labelbottom='off')
ax01.set_ylabel(r'$\rm <S>_{0-200m}$', color='r')
ax01.set_ylim(32.2, 33)
ax01.xaxis.label.set_visible(False)
ax01.tick_params('y', colors='r')
plt.xticks(pd.date_range('1950-01-01', periods=7, freq='10Y'))

fig.set_size_inches(w=9,h=6)
fig_name = 'T-S_1948-2017.png'
fig.set_dpi(300)
fig.savefig(fig_name)

keyboard





