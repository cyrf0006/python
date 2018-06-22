
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates

### --- T-S --- ###
#df_sal_season = pd.read_pickle('salinity_1948-2017.pkl')
#meanS = df_sal_season.iloc[:,df_sal_season.columns<300].mean(axis=1)
df_all = pd.read_pickle('temp_summer_1948-2017.pkl')
CIL_annual = df_all.min(axis=1)
CIL_annual = CIL_annual[(CIL_annual.index.year>1950) & (CIL_annual.index.year<2018)]

### --- Nutrients --- ###

### --- NAO index --- ###
# Load data
nao_file = '/home/cyrf0006/data/AZMP/indices/data.csv'
df = pd.read_csv(nao_file, header=1)

# Set index
df = df.set_index('Date')
df.index = pd.to_datetime(df.index, format='%Y%m')

# Select only DJF
df_winter = df[(df.index.month==12) | (df.index.month==1) | (df.index.month==2)]

# Start Dec-1950
df_winter = df_winter[df_winter.index>pd.to_datetime('1950-10-01')]
df_winter.resample('3M').mean()

# Average 3 consecutive values (DJF average); We loose index.
df_winter = df_winter.groupby(np.arange(len(df_winter))//3).mean()

# Reset index using years only
df_winter.index = pd.unique(df.index.year)[1:,]
df_winter.index = pd.to_datetime(df_winter.index, format='%Y')
df_winter = df_winter[(df_winter.index.year>1950) & (df_winter.index.year<2018)]


### --- AOO index --- ###
# Load data
aoo_file = '/home/cyrf0006/data/AZMP/indices/AOO.xlsx'
df_aoo = pd.read_excel(aoo_file)

# Set index
df_aoo = df_aoo.set_index('Year')
df_aoo.index = pd.to_datetime(df_aoo.index)
df_aoo = df_aoo[(df_aoo.index.year>1950) & (df_aoo.index.year<2018)]


### --- AMO index --- ###
# Load data
amo_file = '/home/cyrf0006/data/AZMP/indices/amon.us.data'
#amo_file = '/home/cyrf0006/data/AZMP/indices/AMO_index.txt'names=['ID','CODE']
#col_names = ['Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
col_names = ['Year','01','02','03','04','05','06','07','08','09','10','11','12']
#df_amo = pd.read_csv(amo_file, header=1, delimiter=r"\s+", index_col=[0])
df_amo = pd.read_csv(amo_file, header=1, delimiter=r"\s+", names=col_names)
df_amo = df_amo.drop(df_amo.tail(4).index) # drop NOAA footer
#df_amo = df_amo.rename(columns={ df_amo.columns[0]: "Year" })

df_amo = df_amo.set_index('Year')
#df_amo.index = pd.to_datetime(df_amo.index)

# Stack months under Years (pretty cool!)
df_amo = df_amo.stack() 

# Transform to a series with values based the 15th of each month
df_amo.index = pd.to_datetime({'year':df_amo.index.get_level_values(0), 'month':df_amo.index.get_level_values(1), 'day':15})
df_amo = df_amo[(df_amo.index.year>1950) & (df_amo.index.year<2018)]

#df_amo = df_amo[df_amo>-10] # remove -99.99 data
df_amo = pd.to_numeric(df_amo) # default dtype is object
df_amo_yearly = df_amo.resample('A').mean() 



### --- plot --- ### (Physics)
CIL_anom = (CIL_annual-CIL_annual.mean())/CIL_annual.std()
AMO_anom = (df_amo_yearly-df_amo_yearly.mean())/df_amo_yearly.std()
NAO_anom = np.squeeze((df_winter-df_winter.mean())/df_winter.std())
AOO_anom = np.squeeze((df_aoo-df_aoo.mean())/df_aoo.std())

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(111)

#plt.scatter(df_amo_yearly.values, df_winter.values, c=CIL_annual.values, alpha=1, s=500, cmap=plt.cm.RdBu_r)
plt.scatter(AMO_anom.values, NAO_anom.values, c=CIL_anom.values, alpha=1, s=500, cmap=plt.cm.RdBu_r)

# Major ticks &  minor ticks
major_ticks = np.arange(-2, 3, 1)
minor_ticks = np.array([0])
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')
ax.grid(which='minor', alpha=1)
ax.grid(which='major', alpha=0.2)

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xlabel('AMO')
plt.ylabel('NAO')
plt.colorbar()

fig.set_size_inches(w=9,h=8)
fig_name = 'index_plane_CIL.png'
fig.set_dpi(300)
fig.savefig(fig_name)


## With AOO
fig = plt.figure(2)
fig.clf()
ax = fig.add_subplot(111)

#plt.scatter(df_amo_yearly.values, df_winter.values, c=CIL_annual.values, alpha=1, s=500, cmap=plt.cm.RdBu_r)
plt.scatter(AOO_anom.values, NAO_anom[NAO_anom.index.year<=2015].values, c=CIL_anom[CIL_anom.index.year<=2015].values, alpha=1, s=500, cmap=plt.cm.Blues_r)

# Major ticks &  minor ticks
major_ticks = np.arange(-2, 3, 1)
minor_ticks = np.array([0])
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')
ax.grid(which='minor', alpha=1)
ax.grid(which='major', alpha=0.2)

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xlabel('AOO')
plt.ylabel('NAO')
plt.colorbar()

fig.set_size_inches(w=9,h=8)
fig_name = 'index_plane_CIL_AOO.png'
fig.set_dpi(300)
fig.savefig(fig_name)




