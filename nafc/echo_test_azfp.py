'''
Try to process AZFP data with echopype
see /home/cyrf0006/research/AZFP

file ex:
/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-01/55139/17071115.01A


'''
#import cmocean

import glob
import numpy as np
#path = '/home/cyrf0006/data_orwell/S27_AZFP/STN27_AZFP_01/55139'
#path = '/home/cyrf0006/data_orwell/S27_AZFP/STN27_AZFP_01/55140'
#path = '/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-02/55139'
path = '/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-02/55140'
#path = '/media/cyrf0006/ExternalDrive/STN27-AZFP-02/55139'
#path = '/media/cyrf0006/ExternalDrive/STN27_AZFP_01/55139'
#path = '/media/cyrf0006/ExternalDrive/STN27_AZFP_01/55140'

# Setup paths
xml_path = glob.glob(path + '/*.XML')[0]
filenames = glob.glob(path + '/*.01A')

# convert the data file-by-file
import echopype as ep
for filename in filenames:
    print('convert ' + filename)
    data_tmp = ep.convert.ConvertAZFP(filename, xml_path)
    data_tmp.raw2nc()


# Calibrate the data
nc_filenames = glob.glob(path + '/*.nc')
from echopype.model import EchoData
for filename in nc_filenames:
    print('calibrate ' + filename)
    data = EchoData(filename)
    data.calibrate()

    # Now we pass coeff for T=5, S=32, D=80 (should be improve in Echopype)
    abs_coeff = data.calc_range().frequency*0+np.array([.009778, .019828, .030685, .042934])
    data.ABS = 2*abs_coeff*data.calc_range()
    data.TVG = 20*np.log10(data.calc_range())
    
    data.remove_noise(noise_est_range_bin_size=5, noise_est_ping_size=20, save=True)  
    data.get_MVBS(source='Sv_clean',  MVBS_range_bin_size=5, MVBS_ping_size=12, save=True)  


keyboard


# Open entire dataset    
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt     
import numpy as np
import matplotlib.pyplot as plt     

path = '/media/cyrf0006/ExternalDrive/STN27_AZFP_01/55139'
ds = xr.open_mfdataset(path + '/*_Sv_clean.nc')
#ds = xr.open_mfdataset(path + '/*.nc')
#ds = xr.open_mfdataset(path + '/1808*_MVBS.nc')
# Select a depth range
ds = ds.sel(range_bin=ds['range_bin']<=166)
ds = ds.sel(range_bin=ds['range_bin']>=5)
# Select frequency
ds_38 = ds.sel(frequency=ds['frequency']==38000.0).squeeze()
ds_67 = ds.sel(frequency=ds['frequency']==67000.0).squeeze()
ds_125 = ds.sel(frequency=ds['frequency']==125000.0).squeeze()
#ds_200 = ds.sel(frequency=ds['frequency']==200000.0).squeeze()

da_38 = ds_38['Sv_clean']
df_38 = da_38.to_pandas()
da_67 = ds_67['Sv_clean']
df_67 = da_67.to_pandas()
da_125 = ds_125['Sv_clean']
df_125 = da_125.to_pandas()


# Time average
#df_38 = df_38.resample('1min').mean()

# mean value test
df_test38 = df_38[(df_38.index>=pd.to_datetime('2017-07-18 00:00:00')) & (df_38.index<=pd.to_datetime('2017-07-18 09:10:00')) ]
df_test67 = df_67[(df_67.index>=pd.to_datetime('2017-07-18 00:00:00')) & (df_67.index<=pd.to_datetime('2017-07-18 09:10:00')) ]
df_test125 = df_125[(df_125.index>=pd.to_datetime('2017-07-18 00:00:00')) & (df_125.index<=pd.to_datetime('2017-07-18 09:10:00')) ]

A = 10**(df_test67/10)
B = np.nanmean(A[:])
10*np.log10(B) 



 # plot
#plt.pcolormesh(df_38.index, 166-df_38.columns, df_38.T, vmin=-120, vmax=-40, cmap=cmocean.cm.thermal)
plt.pcolormesh(df_38.index, 166-df_38.columns, df_38.T, vmin=-120, vmax=-40, cmap='gray_r')
plt.xlim([pd.to_datetime('2017-07-18 00:00:00'), pd.to_datetime('2017-07-18 09:00:00')])
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()



## DPL 2
path = '/media/cyrf0006/ExternalDrive/STN27-AZFP-02/55139'
ds = xr.open_mfdataset(path + '/1807*_Sv_clean.nc')
#ds = xr.open_mfdataset(path + '/1808*_MVBS.nc')
# Select a depth range
ds = ds.sel(range_bin=ds['range_bin']<=166)
ds = ds.sel(range_bin=ds['range_bin']>=40)
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


plt.pcolormesh(df_38.index, 166-df_38.columns, df_38.T, vmin=-120, vmax=-40, cmap='gray_r')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()


