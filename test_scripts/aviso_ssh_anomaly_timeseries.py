'''
Explore Aviso product netcdf file (SSH).
Example derived from http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html

'''

import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
import datetime # Python standard library datetime module
from netCDF4 import Dataset,netcdftime,num2date # http://unidata.github.io/netcdf4-python/
import pandas as pd
import os.path
import csv
import re

# ANomaly box size
box_size = 1 #degree
model_resolution = 1/12.0
box_size_pixel = np.int(box_size/model_resolution)
## map stuff

# Zoom SPM shelf
## lat_min = 45
## lat_max = 48
## lon_min = -59
## lon_max = -52

# Close zoom SPM
lat_min = 46.5
lat_max = 47.5
lon_min = -57
lon_max = -55.5

# AZFP front obs
pt_lat = 47.1
pt_lon = -56.3


## ------- Aviso ------- ##
# load file
nc_f = '/home/cyrf0006/Data/Copernicus/hourly/global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh_1511459141562.nc'
nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file

# Extract data from NetCDF file
lats = nc_fid.variables['latitude'][:]  # extract/copy the data
lons = nc_fid.variables['longitude'][:]
time = nc_fid.variables['time'][:]
zos_tmp = nc_fid.variables['zos'][:]  # shape is time, lat, lon as shown above
zos = np.reshape(zos_tmp, (len(time), len(lats), len(lons)))

# Reduce data according to Region params
idx_lon = np.where(np.logical_and(lons>=lon_min-2, lons<=lon_max+2))
idx_lat = np.where(np.logical_and(lats>=lat_min-2, lats<=lat_max+2))
lons = lons[idx_lon]
lats = lats[idx_lat]

# Caldendar time
time_unit = nc_fid.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"
try :
    t_cal = nc_fid.variables['time'].calendar
except AttributeError : # Attribute doesn't exist
    t_cal = u"gregorian" # or standard

tvalue = num2date(time,units = time_unit,calendar = t_cal)
str_time = [i.strftime("%Y-%m-%d %H:%M") for i in tvalue] # to display dates as string

# Find pts in 
lon_idx_pt = (np.abs(lons-pt_lon)).argmin()
lat_idx_pt = (np.abs(lats-pt_lat)).argmin()

pickle_name = 'ssh_anom_AZMPfront.pkl'

# Load or generate anomaly timeseries
if os.path.isfile(pickle_name):
    print 'Open existing pickeled file'
    df = pd.read_pickle(pickle_name)
else:

    # plot ZOS in timeframe
    anom_timeseries = []
    for time_idx in np.arange(0,len(time)):
        print str_time[time_idx]
    
        # raw SSH
        ssh = np.squeeze(zos[time_idx,:,:])
        ssh = ssh[np.ix_(np.squeeze(idx_lat), np.squeeze(idx_lon))]
    
        ssh_anom = np.empty(np.shape(ssh))
        ssh_anom[:] = np.NAN
        
        # Compute anomaly
        for i in np.arange(box_size_pixel/2,len(lats)-box_size_pixel/2):
            for j in np.arange(box_size_pixel/2,len(lons)-box_size_pixel/2):
                ssh_anom[i,j] = ssh[i,j] -  np.nanmean(ssh[i-box_size_pixel/2:i+box_size_pixel/2,j-box_size_pixel/2:j+box_size_pixel/2])
    
        anom_timeseries.append(ssh_anom[lat_idx_pt, lon_idx_pt])
    
    df = pd.DataFrame(anom_timeseries, index=pd.DatetimeIndex(str_time))
    df.to_pickle(pickle_name)


#### ---- Load mooring data ---- #
#mooring_file = './data/_miq_17155_14_09_2017_18_17_03.txt' # ~37m
#mooring_file = './data/_miq_17156_14_09_2017_18_14_52.txt' # ~30m
#mooring_file = './data/_miq_17159_14_09_2017_18_22_20.txt' # top
#mooring_file = './data/_miq_17157_14_09_2017_18_12_38.txt' # middle
mooring_file = './data/_Miquelon_60m_17114_14_09_2017_18_38_21.txt'
temp = []
pres = []
moo_time = []
with open(mooring_file, 'r') as td:
        for idx, line in enumerate(td):
            if idx < 20:
                continue # ignore line
            else: # read data
                line = re.sub('\n','', line)
                line = line.strip()
                #line = re.sub(' +',' ', line)
                moo_time.append(line.split('\t')[1])
                temp.append(line.split('\t')[3])
                pres.append(line.split('\t')[5])
                
# to dataFrame       
XLIM = [pd.to_datetime('20170831110000'), pd.to_datetime('20170912170000')]
df_temp = pd.DataFrame(temp, index=pd.to_datetime(moo_time, format='%d/%m/%Y %H:%M:%S'), dtype=float)
df_pres = pd.DataFrame(pres, index=pd.to_datetime(moo_time, format='%d/%m/%Y %H:%M:%S'), dtype=float)
df_pres = df_pres[XLIM[0]:XLIM[1]]
df_temp = df_temp[XLIM[0]:XLIM[1]]

df_temp = df_temp.resample('60Min').mean()
df_pres = df_pres.resample('60Min').mean()
fs_temp = 1.0/3600

    
#### ---- timeseries compa ---- ####
plt.figure(1)
#ax0 = plt.axes()
ax0 = plt.subplot(211)
ax0.plot(df.index, df*100, '#1f77b4')
ax0.set_xlim(XLIM[0], XLIM[1])
ax0.set_ylabel(r'$\rm \eta\prime (cm)$', color='#1f77b4', fontweight='bold')
ax0.tick_params('y', colors='#1f77b4')
plt.grid()
ax0.text(XLIM[1], 3, 'a  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k', fontweight='bold')

ax01 = ax0.twinx()
ax01.plot(df_temp.index, df_temp, '#ff7f0e')
ax01.set_xlim(XLIM[0], XLIM[1])
ax01.set_ylabel(r'$\rm T (^{\circ}C)$', color='#ff7f0e', fontweight='bold')
ax01.tick_params('y', colors='#ff7f0e')



#### ---- Welch method ---- ####
from scipy import signal
N = seriesize = np.size(df.index)
fs = 1.0/3600
f, Pxx = signal.welch(np.squeeze(np.array(df)), fs, nperseg=N/2)

N = np.size(df_temp.index)
f_temp, Pxx_temp = signal.welch(np.squeeze(np.array(df_temp)), fs_temp, nperseg=N)


sdiurnal_x = [1.0/(12*60*60), 1.0/(12*60*60)]
sdiurnal_y = [np.min(Pxx), np.max(Pxx)]
diurnal_x = [1.0/(25*60*60), 1.0/(25*60*60)]
diurnal_y = [np.min(Pxx), np.max(Pxx)]
twodays_x = [1.0/(48*60*60), 1.0/(48*60*60)]
twodays_y = [np.min(Pxx), np.max(Pxx)]
h28_x = [1.0/(29*60*60), 1.0/(29*60*60)]
h28_y = [np.min(Pxx), np.max(Pxx)]

ax = plt.subplot(212)
plt.loglog(f, Pxx, '#1f77b4')
plt.loglog(f_temp, Pxx_temp/1e4, '#ff7f0e')
plt.plot(diurnal_x,diurnal_y, '--k')
plt.plot(twodays_x,twodays_y, '--k')
plt.plot(sdiurnal_x,sdiurnal_y, '--k')
plt.plot(h28_x,h28_y, '--k')
ax.set_xlim([1e-6, 1e-4])
ax.set_ylim([1e-3, 2e2])
ax.text(1e-4, 1e2, 'b  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k', fontweight='bold')

#plt.loglog(2*np.pi/(60.0*f_Whigh), Pxx_den_Whigh)
#plt.xlim([4e0, 3e1])
plt.ylabel('PSD [V**2/Hz]', fontweight='bold')
plt.xlabel('f [Hz]', fontweight='bold')
ax.grid('on')
plt.text(sdiurnal_x[0],sdiurnal_y[1] , '12h', fontweight='bold')
plt.text(diurnal_x[0],diurnal_y[1] , '25h', fontweight='bold')
plt.text(h28_x[0],h28_y[1] , '29h', horizontalalignment='left', fontweight='bold')
plt.text(twodays_x[0], twodays_y[1] , '48h', horizontalalignment='left', fontweight='bold')

plt.show()

#fig.savefig(fig_name)
