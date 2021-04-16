'''
This is a test with xarray, see if I can manipulate a lot of netCDF files.
Try this script in /home/cyrf0006/research/AZMP_database
'''

## ---- Convert the data ---- ##

import adcpy.TRDIstuff.TRDIpd0tonetcdf as ad
# in '/media/cyrf0006/Seagate Backup Plus Drive/VMP_dataprocessing/FISHDIP2020'
ad.convert_pd0_to_netcdf('FishDip_iceCamp_20200203T192010.pd0', 'ADCP_FishDipiceCamp01.nc', [0, 136659], '24452', 'CF', 1)
ad.convert_pd0_to_netcdf('FishDip_iceCamp_20200206T025432.pd0', 'ADCP_FishDipiceCamp02.nc', [0, 118400], '24452', 'CF', 1)

## ---- First plots ---- ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import os
ds1 = xr.open_dataset('/media/cyrf0006/Seagate Backup Plus Drive/VMP_dataprocessing/FISHDIP2020/ADCP_FishDipiceCamp01.nc')           
ds2 = xr.open_dataset('/media/cyrf0006/Seagate Backup Plus Drive/VMP_dataprocessing/FISHDIP2020/ADCP_FishDipiceCamp02.nc')           

# To Pandas Dataframe
df_sv = ds['sv'].to_pandas()
df_att = ds['att1'].to_pandas()
df_v1 = ds['vel1'].to_pandas()        
df_v2 = ds['vel2'].to_pandas()        
df_v3 = ds['vel3'].to_pandas()        
df_v4 = ds['vel4'].to_pandas()     

U = np.array([df_v1.values,df_v2.values, df_v3.values,df_v4.values]).mean(axis=0)




plt.pcolormesh(df_att.T)


df_att_2 = ds2['att1'].to_pandas()

plt.pcolormesh(df_att_2.index, df_att_2.columns, df_att_2.T) 
plt.gca().invert_yaxis() 


# HERE!


# 5min velocity averages
df_v1_5min = df_v1.resample('5min').mean()  
plt.pcolormesh(df_v1_5min.index, df_v1_5min.columns, df_v1_5min.T) 
plt.gca().invert_yaxis() 




# Select Lake Melville
# Selection of a subset region
ds = ds.where((ds.longitude>-60.5) & (ds.longitude<-58.5), drop=True) # original one
ds = ds.where((ds.latitude>53.3) & (ds.latitude<54.1), drop=True)

# Sort by time dimension
ds = ds.sortby('time')
lat = ds.latitude.values
lon = ds.longitude.values

## ---- Plot map ---- ##
lonLims = [-61, -58] # ake Melville
latLims = [53, 54.2]
proj = 'merc'
fig_name = 'map_CTDs.png'
    
fig, ax = plt.subplots(nrows=1, ncols=1)
m = Basemap(ax=ax, projection='merc', llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='f')
levels = np.linspace(0, 15, 16)
x,y = m(lon, lat)
m.plot(x, y, '.r')
m.fillcontinents(color='tan');

# Monthly average
## ds = ds.sortby('time')
## ds_monthly = ds.resample(time="M").mean('time') 

# To Pandas Dataframe
da_temp = ds['temperature']
df_temp = da_temp.to_pandas()
da_sal = ds['salinity']
df_sal = da_sal.to_pandas()

sal = df_sal.values.reshape(1,df_sal.size)
temp = df_temp.values.reshape(1,df_temp.size)
year = np.tile(df_temp.index.year, (1,500))  

# HERE!!!!!
plt.scatter(sal, temp, c=year)

#Load winter 2019
df_temp = pd.read_pickle('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/temp_W2019.pkl')
df_sal = pd.read_pickle('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/sal_W2019.pkl')
sal1 = df_sal.values.reshape(1,df_sal.size)
temp1 = df_temp.values.reshape(1,df_temp.size)
temp1 = temp1[sal1>0]
sal1 = sal1[sal1>0]
year1 = np.repeat(2019, sal1.size)

plt.figure()
plt.scatter(sal, temp, c=year, year1)
plt.scatter(sal1, temp1, c=np.repeat(2019, sal1.size)) 

