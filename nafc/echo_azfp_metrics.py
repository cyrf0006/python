'''
Here we use generated NetCDF files to classify multi-frequence stuff

'''

### ****** PLot figure using Sv_clean *********
# But use MVBS for classification

# Open entire dataset    
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt     
import numpy as np
import matplotlib.pyplot as plt     
import cmocean
import os
import matplotlib.dates as mdates

# Attempt to load all files at the same time
#path = '/home/cyrf0006/data_orwell/AZFP_all/55139'
#path2 = '/home/cyrf0006/data_orwell/AZFP_all/55140'

dataset_path = '/media/cyrf0006/DevClone/data_orwell/AZFP_all'

## ---- Get MVBS ---- ##
ds = xr.open_mfdataset(os.path.join(dataset_path, '55139/*_MVBS.nc'), combine='by_coords')
ds2 = xr.open_mfdataset(os.path.join(dataset_path, '55140/*_MVBS.nc'), combine='by_coords')
# Select a depth range
ds = ds.sel(range_bin=ds['range_bin']<=166)
ds = ds.sel(range_bin=ds['range_bin']>=5)
ds2 = ds2.sel(range_bin=ds2['range_bin']<=100)
ds2 = ds2.sel(range_bin=ds2['range_bin']>=5)

# Select a time range & flag bad data
time_dpl1 = [pd.to_datetime('2017-07-10 12:00'), pd.to_datetime('2017-07-28 14:30')]
time_dpl2 = [pd.to_datetime('2017-12-16 16:00'), pd.to_datetime('2018-08-02 22:00')]
time_dpl3 = [pd.to_datetime('2018-12-05 12:00'), pd.to_datetime('2019-07-11 20:30')]
# Explicit loading in order to flag bad data (long... try to find a better way maybe)
dpl1 = ds.sel(ping_time=slice(time_dpl1[0], time_dpl1[1])).load()
dpl1['MVBS'][:,:,11]=np.nan
dpl1['MVBS'][:,:,1]=np.nan
dpl1['MVBS'][:,-1,:]=np.nan
dpl2 = ds.sel(ping_time=slice(time_dpl2[0], time_dpl2[1])).load()
dpl2['MVBS'][:,:,11]=np.nan
dpl2['MVBS'][:,:,12]=np.nan
dpl2['MVBS'][:,:,1]=np.nan
dpl2['MVBS'][:,-1,:]=np.nan
dpl3 = ds.sel(ping_time=slice(time_dpl3[0], time_dpl3[1])).load()
dpl3['MVBS'][:,:,10]=np.nan
dpl3['MVBS'][:,:,1]=np.nan
ds = xr.concat((dpl1, dpl2, dpl3), 'ping_time')
dpl1 = ds2.sel(ping_time=slice(time_dpl1[0], time_dpl1[1])).load()
dpl1['MVBS'][:,-1,:]=np.nan
dpl2 = ds2.sel(ping_time=slice(time_dpl2[0], time_dpl2[1])).load()
dpl2['MVBS'][:,-1,:]=np.nan
dpl3 = ds2.sel(ping_time=slice(time_dpl3[0], time_dpl3[1])).load()
dpl3['MVBS'][:,-1,:]=np.nan
ds2 = xr.concat((dpl1, dpl2, dpl3), 'ping_time')


# Select frequency
ds_38 = ds.sel(frequency=ds['frequency']==38000.0).squeeze()
ds_67 = ds.sel(frequency=ds['frequency']==67000.0).squeeze()
ds_125 = ds.sel(frequency=ds['frequency']==125000.0).squeeze()
ds_200 = ds.sel(frequency=ds['frequency']==200000.0).squeeze()
ds_200_2 = ds2.sel(frequency=ds2['frequency']==200000.0).squeeze()
ds_333 = ds2.sel(frequency=ds2['frequency']==333000.0).squeeze()
ds_455 = ds2.sel(frequency=ds2['frequency']==455000.0).squeeze()

da_38 = ds_38['MVBS']
mvbs_38 = da_38.to_pandas()
da_67 = ds_67['MVBS']
mvbs_67 = da_67.to_pandas()
da_125 = ds_125['MVBS']
mvbs_125 = da_125.to_pandas()
#da_200 = ds_200['MVBS']
#mvbs_200 = da_200.to_pandas()
da_200 = ds_200_2['MVBS']
mvbs_200 = da_200.to_pandas()
da_333 = ds_333['MVBS']
mvbs_333 = da_333.to_pandas()
da_455 = ds_455['MVBS']
mvbs_455 = da_455.to_pandas()

## ---- plot MVBS Figure ---- ##
XLIM = [pd.to_datetime('2017-07-10 12:00:00'), pd.to_datetime('2019-07-23 16:00:00')]
YLIM = [0, 162]
fig = plt.figure()
# ax1 - 38 kHz
ax = plt.subplot2grid((6, 1), (0, 0))
c = plt.pcolormesh(mvbs_38.index, 166-mvbs_38.columns, mvbs_38.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax.set_ylim([0, 176])
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.xaxis.label.set_visible(False)
ax.tick_params(labelbottom='off')
ax.text(pd.to_datetime('2017-08-15'), 150, '38 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.set_xticklabels([])

# ax2 - 67 kHz
ax2 = plt.subplot2grid((6, 1), (1, 0))
c = plt.pcolormesh(mvbs_67.index, 166-mvbs_67.columns, mvbs_67.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax2.set_ylim([0, 176])
ax2.set_xlim(XLIM)
ax2.set_ylim(YLIM)
#ax2.set_ylabel('Depth (m)', fontWeight = 'bold')
ax2.invert_yaxis()
plt.colorbar(c)
ax2.xaxis.label.set_visible(False)
ax2.tick_params(labelbottom='off')
ax2.text(pd.to_datetime('2017-08-15'), 150, '67 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.set_xticklabels([])

# ax3 - 125 kHz
ax3 = plt.subplot2grid((6, 1), (2, 0))
c = plt.pcolormesh(mvbs_125.index, 166-mvbs_125.columns, mvbs_125.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax3.set_ylim([0, 176])
ax3.set_xlim(XLIM)
ax3.set_ylim(YLIM)
#ax3.set_ylabel('Depth (m)', fontWeight = 'bold')
ax3.invert_yaxis()
plt.colorbar(c)
ax3.xaxis.label.set_visible(False)
ax3.tick_params(labelbottom='off')
ax3.text(pd.to_datetime('2017-08-15'), 150, '125 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax3.set_xticklabels([])

# ax4 - 200 kHz
ax4 = plt.subplot2grid((6, 1), (3, 0))
c = plt.pcolormesh(mvbs_200.index, 100-mvbs_200.columns, mvbs_200.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax4.set_ylim([0, 176])
ax4.set_xlim(XLIM)
ax4.set_ylim(YLIM)
#ax4.set_ylabel('Depth (m)', fontWeight = 'bold')
ax4.invert_yaxis()
plt.colorbar(c)
ax4.xaxis.label.set_visible(False)
ax4.tick_params(labelbottom='off')
ax4.text(pd.to_datetime('2017-08-15'), 150, '200 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax4.set_xticklabels([])

# ax5 - 333 kHz
ax5 = plt.subplot2grid((6, 1), (4, 0))
c = plt.pcolormesh(mvbs_333.index, 100-mvbs_333.columns, mvbs_333.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax5.set_ylim([0, 176])
ax5.set_xlim(XLIM)
ax5.set_ylim(YLIM)
#ax5.set_ylabel('Depth (m)', fontWeight = 'bold')
ax5.invert_yaxis()
plt.colorbar(c)
ax5.xaxis.label.set_visible(False)
ax5.tick_params(labelbottom='off')
ax5.text(pd.to_datetime('2017-08-15'), 150, '333 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax5.set_xticklabels([])

# ax6 - 455 kHz
ax6 = plt.subplot2grid((6, 1), (5, 0))
c = plt.pcolormesh(mvbs_455.index, 100-mvbs_455.columns, mvbs_455.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax6.set_ylim([0, 176])
ax6.set_xlim(XLIM)
ax6.set_ylim(YLIM)
#ax6.set_ylabel('Depth (m)', fontWeight = 'bold')
ax6.invert_yaxis()
plt.colorbar(c)
ax6.text(pd.to_datetime('2017-08-15'), 150, '455 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

fig.set_size_inches(w=8.5,h=11)
fig_name = 'AZFP_MVBS_all.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## Urmy metrics
import echometrics.echometrics as urmy
echo38 = urmy.Echogram(mvbs_38.values.T, 166-mvbs_38.columns.values, np.arange(0,len(mvbs_38.index)), threshold=[-120,0])
echo67 = urmy.Echogram(mvbs_67.values.T, 166-mvbs_67.columns.values, np.arange(0,len(mvbs_67.index)), threshold=[-120,0])
echo125 = urmy.Echogram(mvbs_125.values.T, 166-mvbs_125.columns.values, np.arange(0,len(mvbs_125.index)), threshold=[-120,0])
echo200 = urmy.Echogram(mvbs_200.values.T, 100-mvbs_200.columns.values, np.arange(0,len(mvbs_200.index)), threshold=[-120,0])
echo333 = urmy.Echogram(mvbs_333.values.T, 100-mvbs_333.columns.values, np.arange(0,len(mvbs_333.index)), threshold=[-120,0])
echo455 = urmy.Echogram(mvbs_455.values.T, 100-mvbs_455.columns.values, np.arange(0,len(mvbs_455.index)), threshold=[-120,0])

Sv38echo = urmy.sv_avg(echo38)
Sv38_top = urmy.depth_integral(echo38,range=[0,50])
cm38 = urmy.center_of_mass(echo38)   
I38 = urmy.inertia(echo38)   
IA38 = urmy.aggregation_index(echo38)   
EA38 = urmy.equivalent_area(echo38)   
Pocc38 = urmy.proportion_occupied(echo38)
#Nl = urmy.nlayers(echo38)
df_urmy38 = pd.DataFrame({'Sv':Sv38echo, 'Sv_top':Sv38_top, 'CM':cm38, 'I':I38,'IA':IA38, 'EA':EA38, 'Pocc':Pocc38})
df_urmy38.index = mvbs_38.index


Sv67echo = urmy.sv_avg(echo67)
Sv67_top = urmy.depth_integral(echo67,range=[0,50])
cm67 = urmy.center_of_mass(echo67)   
I67 = urmy.inertia(echo67)   
IA67 = urmy.aggregation_index(echo67)   
EA67 = urmy.equivalent_area(echo67)   
Pocc67 = urmy.proportion_occupied(echo67)
df_urmy67 = pd.DataFrame({'Sv':Sv67echo, 'Sv_top':Sv67_top, 'CM':cm67, 'I':I67,'IA':IA67, 'EA':EA67, 'Pocc':Pocc67})
df_urmy67.index = mvbs_67.index

Sv125echo = urmy.sv_avg(echo125)
Sv125_top = urmy.depth_integral(echo125,range=[0,50])
cm125 = urmy.center_of_mass(echo125)   
I125 = urmy.inertia(echo125)   
IA125 = urmy.aggregation_index(echo125)   
EA125 = urmy.equivalent_area(echo125)   
Pocc125 = urmy.proportion_occupied(echo125)
df_urmy125 = pd.DataFrame({'Sv':Sv125echo, 'Sv_top':Sv125_top, 'CM':cm125, 'I':I125,'IA':IA125, 'EA':EA125, 'Pocc':Pocc125})
df_urmy125.index = mvbs_125.index

Sv200echo = urmy.sv_avg(echo200)
Sv200_top = urmy.depth_integral(echo200,range=[0,50])
cm200 = urmy.center_of_mass(echo200)   
I200 = urmy.inertia(echo200)   
IA200 = urmy.aggregation_index(echo200)   
EA200 = urmy.equivalent_area(echo200)   
Pocc200 = urmy.proportion_occupied(echo200)
df_urmy200 = pd.DataFrame({'Sv':Sv200echo, 'Sv_top':Sv200_top, 'CM':cm200, 'I':I200,'IA':IA200, 'EA':EA200, 'Pocc':Pocc200})
df_urmy200.index = mvbs_200.index

Sv333echo = urmy.sv_avg(echo333)
Sv333_top = urmy.depth_integral(echo333,range=[0,50])
cm333 = urmy.center_of_mass(echo333)   
I333 = urmy.inertia(echo333)   
IA333 = urmy.aggregation_index(echo333)   
EA333 = urmy.equivalent_area(echo333)   
Pocc333 = urmy.proportion_occupied(echo333)
df_urmy333 = pd.DataFrame({'Sv':Sv333echo, 'Sv_top':Sv333_top, 'CM':cm333, 'I':I333,'IA':IA333, 'EA':EA333, 'Pocc':Pocc333})
df_urmy333.index = mvbs_333.index

Sv455echo = urmy.sv_avg(echo455)
Sv455_top = urmy.depth_integral(echo455,range=[0,50])
cm455 = urmy.center_of_mass(echo455)   
I455 = urmy.inertia(echo455)   
IA455 = urmy.aggregation_index(echo455)   
EA455 = urmy.equivalent_area(echo455)   
Pocc455 = urmy.proportion_occupied(echo455)
df_urmy455 = pd.DataFrame({'Sv':Sv455echo, 'Sv_top':Sv455_top, 'CM':cm455, 'I':I455,'IA':IA455, 'EA':EA455, 'Pocc':Pocc455})
df_urmy455.index = mvbs_455.index

df38 = df_urmy38.resample('1D').mean()         
df67 = df_urmy67.resample('1D').mean()         
df125 = df_urmy125.resample('1D').mean()         
df200 = df_urmy200.resample('1D').mean()         
df333 = df_urmy333.resample('1D').mean()         
df455 = df_urmy455.resample('1D').mean()         


# Sv top 50m
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(df38.index,df38.Sv_top) 
ax.plot(df67.index,df67.Sv_top) 
ax.plot(df125.index,df125.Sv_top) 
ax.plot(df200.index,df200.Sv_top) 
ax.plot(df333.index,df333.Sv_top) 
ax.plot(df455.index,df455.Sv_top) 
ax.invert_yaxis()
ax.legend(['38kHz', '67kHz', '125kHz', '200kHz', '333kHz', '455kHz'])
plt.title('Sv 0-50m')

# Center of mass
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(df38.index,df38.CM) 
ax.plot(df67.index,df67.CM) 
ax.plot(df125.index,df125.CM) 
ax.plot(df200.index,df200.CM) 
ax.plot(df333.index,df333.CM) 
ax.plot(df455.index,df455.CM) 
ax.invert_yaxis()
ax.legend(['38kHz', '67kHz', '125kHz', '200kHz', '333kHz', '455kHz'])
plt.title('Center of mass')

# Aggregation Index
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(df38.index,df38.IA) 
ax.plot(df67.index,df67.IA) 
ax.plot(df125.index,df125.IA) 
ax.plot(df200.index,df200.IA) 
ax.plot(df333.index,df333.IA) 
ax.plot(df455.index,df455.IA) 
ax.invert_yaxis()
ax.legend(['38kHz', '67kHz', '125kHz', '200kHz', '333kHz', '455kHz'])
plt.title('Aggregation Index')

keyboard

# Spring
dfSv_spring = df38.Sv_top[(df38.index.month>=3) & (df38.index.month<=5)]
dfSv_spring.append(df38.Sv_top[(df67.index.month>=3) & (df67.index.month<=5)])  

df38.Sv_
df38[(df38.index.month>=3) & (df38.index.month<=5)] 

keyboard

