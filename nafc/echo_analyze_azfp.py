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

path = '/media/cyrf0006/ExternalDrive/STN27_AZFP_01/55139'
path2 = '/media/cyrf0006/ExternalDrive/STN27_AZFP_01/55140'
path = '/home/cyrf0006/data_orwell/S27_AZFP/STN27_AZFP_01/55139'
path2 = '/home/cyrf0006/data_orwell/S27_AZFP/STN27_AZFP_01/55140'

# Attempt to load all files at the same time
path = '/home/cyrf0006/data_orwell/AZFP_all/55139'
path2 = '/home/cyrf0006/data_orwell/AZFP_all/55140'


## ---- Get Sv_clean ---- ##
ds = xr.open_mfdataset(path + '/*_Sv_clean.nc')
ds2 = xr.open_mfdataset(path2 + '/*_Sv_clean.nc')
# Select a depth range
ds = ds.sel(range_bin=ds['range_bin']<=166)
ds = ds.sel(range_bin=ds['range_bin']>=5)
ds2 = ds2.sel(range_bin=ds2['range_bin']<=100)
ds2 = ds2.sel(range_bin=ds2['range_bin']>=5)
# Select frequency
ds_38 = ds.sel(frequency=ds['frequency']==38000.0).squeeze()
ds_67 = ds.sel(frequency=ds['frequency']==67000.0).squeeze()
ds_125 = ds.sel(frequency=ds['frequency']==125000.0).squeeze()
ds_200 = ds.sel(frequency=ds['frequency']==200000.0).squeeze()
ds_200_2 = ds2.sel(frequency=ds2['frequency']==200000.0).squeeze()
ds_333 = ds2.sel(frequency=ds2['frequency']==333000.0).squeeze()
ds_455 = ds2.sel(frequency=ds2['frequency']==455000.0).squeeze()

# Convert to Pandas
da_38 = ds_38['Sv_clean']
df_38 = da_38.to_pandas()
da_67 = ds_67['Sv_clean']
df_67 = da_67.to_pandas()
da_125 = ds_125['Sv_clean']
df_125 = da_125.to_pandas()
#da_200 = ds_200['Sv_clean']
#df_200 = da_200.to_pandas()
da_200 = ds_200_2['Sv_clean']
df_200 = da_200.to_pandas()
da_333 = ds_333['Sv_clean']
df_333 = da_333.to_pandas()
da_455 = ds_455['Sv_clean']
df_455 = da_455.to_pandas()

## ---- Get MVBS ---- ##
ds = xr.open_mfdataset(path + '/*_MVBS.nc')
ds2 = xr.open_mfdataset(path2 + '/*_MVBS.nc')
# Select a depth range
ds = ds.sel(range_bin=ds['range_bin']<=166)
ds = ds.sel(range_bin=ds['range_bin']>=5)
ds2 = ds2.sel(range_bin=ds2['range_bin']<=100)
ds2 = ds2.sel(range_bin=ds2['range_bin']>=5)
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

## ---- plot Figure ---- ##
XLIM = [pd.to_datetime('2017-07-10 12:00:00'), pd.to_datetime('2017-07-28 14:30:00')]

fig = plt.figure()
# ax1 - 38 kHz
ax = plt.subplot2grid((6, 1), (0, 0))
c = plt.pcolormesh(df_38.index, 166-df_38.columns, df_38.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax.set_ylim([0, 176])
ax.set_xlim(XLIM)
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.xaxis.label.set_visible(False)
ax.tick_params(labelbottom='off')
ax.text(XLIM[0]+pd.Timedelta(np.timedelta64(6, 'h')), 165, '38 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.set_xticklabels([])

# ax2 - 67 kHz
ax2 = plt.subplot2grid((6, 1), (1, 0))
c = plt.pcolormesh(df_67.index, 166-df_67.columns, df_67.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax2.set_ylim([0, 176])
ax2.set_xlim(XLIM)
#ax2.set_ylabel('Depth (m)', fontWeight = 'bold')
ax2.invert_yaxis()
plt.colorbar(c)
ax2.xaxis.label.set_visible(False)
ax2.tick_params(labelbottom='off')
ax2.text(XLIM[0]+pd.Timedelta(np.timedelta64(6, 'h')), 165, '67 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax2.set_xticklabels([])

# ax3 - 125 kHz
ax3 = plt.subplot2grid((6, 1), (2, 0))
c = plt.pcolormesh(df_125.index, 166-df_125.columns, df_125.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax3.set_ylim([0, 176])
ax3.set_xlim(XLIM)
#ax3.set_ylabel('Depth (m)', fontWeight = 'bold')
ax3.invert_yaxis()
plt.colorbar(c)
ax3.xaxis.label.set_visible(False)
ax3.tick_params(labelbottom='off')
ax3.text(XLIM[0]+pd.Timedelta(np.timedelta64(6, 'h')), 165, '125 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax3.set_xticklabels([])

# ax4 - 200 kHz
ax4 = plt.subplot2grid((6, 1), (3, 0))
c = plt.pcolormesh(df_200.index, 100-df_200.columns, df_200.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax4.set_ylim([0, 176])
ax4.set_xlim(XLIM)
#ax4.set_ylabel('Depth (m)', fontWeight = 'bold')
ax4.invert_yaxis()
plt.colorbar(c)
ax4.xaxis.label.set_visible(False)
ax4.tick_params(labelbottom='off')
ax4.text(XLIM[0]+pd.Timedelta(np.timedelta64(6, 'h')), 165, '200 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax4.set_xticklabels([])

# ax5 - 333 kHz
ax5 = plt.subplot2grid((6, 1), (4, 0))
c = plt.pcolormesh(df_333.index, 100-df_333.columns, df_333.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax5.set_ylim([0, 176])
ax5.set_xlim(XLIM)
#ax5.set_ylabel('Depth (m)', fontWeight = 'bold')
ax5.invert_yaxis()
plt.colorbar(c)
ax5.xaxis.label.set_visible(False)
ax5.tick_params(labelbottom='off')
ax5.text(XLIM[0]+pd.Timedelta(np.timedelta64(6, 'h')), 165, '333 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax5.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax5.set_xticklabels([])

# ax6 - 455 kHz
ax6 = plt.subplot2grid((6, 1), (5, 0))
c = plt.pcolormesh(df_455.index, 100-df_455.columns, df_455.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax6.set_ylim([0, 176])
ax6.set_xlim(XLIM)
#ax6.set_ylabel('Depth (m)', fontWeight = 'bold')
ax6.invert_yaxis()
plt.colorbar(c)
ax6.text(XLIM[0]+pd.Timedelta(np.timedelta64(6, 'h')), 165, '455 kHz', horizontalalignment='left', verticalalignment='bottom', fontsize=13, color='k', backgroundcolor='w')
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%D'))
ax6.xaxis.set_major_locator(mdates.DayLocator(interval=3))

fig.set_size_inches(w=8.5,h=11)
fig_name = 'AZFP_Sv_clean.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

keyboard
# Do do:
# - add other frequencies (and keep same vertical axis)
# - Probably need to average in linear domain here...

## ---- An average day plot ---- ##
# This is sick! - Make an average one-minute day
aDay_38 = df_38.groupby([df_38.index.hour, df_38.index.minute]).mean()   
aDay_67 = df_67.groupby([df_67.index.hour, df_67.index.minute]).mean()   
aDay_125 = df_125.groupby([df_125.index.hour, df_125.index.minute]).mean()
aDay_200 = df_200.groupby([df_200.index.hour, df_200.index.minute]).mean()
aDay_index = pd.to_datetime(aDay_38.index.map(lambda x: ''.join((str(x[0]), ':',str(x[1]))))  , format='%H:%M')  
aDay_38.index = aDay_index
aDay_67.index = aDay_index
aDay_125.index = aDay_index
aDay_200.index = aDay_index

# cheating the index
prev_day = aDay_38[aDay_index.hour>11]
next_day = aDay_38[aDay_index.hour<=11]
new_index = prev_day.index.append(next_day.index+pd.DateOffset(1))

aDay_38 = prev_day.append(next_day)
aDay_38.index = new_index
prev_day = aDay_67[aDay_index.hour>11]
next_day = aDay_67[aDay_index.hour<=11]
aDay_67 = prev_day.append(next_day)
aDay_67.index = new_index
prev_day = aDay_125[aDay_index.hour>11]
next_day = aDay_125[aDay_index.hour<=11]
aDay_125 = prev_day.append(next_day)
aDay_125.index = new_index
prev_day = aDay_200[aDay_index.hour>11]
next_day = aDay_200[aDay_index.hour<=11]
aDay_200 = prev_day.append(next_day)
aDay_200.index = new_index
## # Plot the result
## plt.pcolormesh(A.index, 166-A.columns, A.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal) 
## plt.gca().invert_yaxis() 
## plt.colorbar() 
## plt.show()     

## # cheating the index
## prev_day = A[A.index.hour>11]
## next_day = A[A.index.hour<=11]
## A = prev_day.append(next_day)
## new_index = prev_day.index.append(next_day.index.pd.DateOffset(1))
## A.index = new_index
## # Plot the result
## plt.pcolormesh(aDay_38.index, 166-aDay_38.columns, aDay_38.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal) 
## plt.gca().invert_yaxis() 
## plt.colorbar() 
## plt.show()     

fig = plt.figure()
# ax1 - 38 kHz
ax = plt.subplot2grid((4, 1), (0, 0))
c = ax.pcolormesh(aDay_38.index, 166-aDay_38.columns, aDay_38.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
#ax.set_ylim([0, 400])
#ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title('38 kHz')
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.set_xticklabels([])

# ax2 - 67 kHz
ax2 = plt.subplot2grid((4, 1), (1, 0))
c = ax2.pcolormesh(aDay_67.index, 166-aDay_67.columns, aDay_67.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax2.set_ylabel('Depth (m)', fontWeight = 'bold')
ax2.invert_yaxis()
plt.colorbar(c)
ax2.set_title('67 kHz')
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.set_xticklabels([])

# ax3 - 125 kHz
ax3 = plt.subplot2grid((4, 1), (2, 0))
c = ax3.pcolormesh(aDay_125.index, 166-aDay_125.columns, aDay_125.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax3.set_ylabel('Depth (m)', fontWeight = 'bold')
ax3.invert_yaxis()
plt.colorbar(c)
ax3.set_title('125 kHz')
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax3.set_xticklabels([])

# ax4 - 200 kHz
ax4 = plt.subplot2grid((4, 1), (3, 0))
c = ax4.pcolormesh(aDay_200.index, 166-aDay_200.columns, aDay_200.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
ax4.set_ylabel('Depth (m)', fontWeight = 'bold')
ax4.invert_yaxis()
plt.colorbar(c)
ax4.set_title('200 kHz')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                 
fig.set_size_inches(w=8.5,h=11)
fig_name = 'AZFP_typycal_day.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


# Do do:


df_top = df_38.loc[:,5:50] 
A = df_top.groupby(df_top.index.hour).mean()

B = df_38.groupby(df_38.index.hour).mean() 






## DPL 2
path = '/media/cyrf0006/ExternalDrive/STN27-AZFP-02/55139'
ds = xr.open_mfdataset(path + '/1807*_Sv_clean.nc')
#ds = xr.open_mfdataset(path + '/1808*_MVBS.nc')
# Select a depth range
ds = ds.sel(range_bin=ds['range_bin']<=166)
ds = ds.sel(range_bin=ds['range_bin']>=5)
# Select frequency
ds_38 = ds.sel(frequency=ds['frequency']==38000.0).squeeze()
#ds_67 = ds.sel(frequency=ds['frequency']==67000.0).squeeze()
#ds_120 = ds.sel(frequency=ds['frequency']==120000.0).squeeze()
#ds_200 = ds.sel(frequency=ds['frequency']==200000.0).squeeze()

da = ds_38['Sv_clean']
#da = ds_38['MVBS']
df_38 = da.to_pandas()
#da = ds_67['Sv_clean']
#df_67 = da.to_pandas()


plt.pcolormesh(df_38.index, 166-df_38.columns, df_38.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()


