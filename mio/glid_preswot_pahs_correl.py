'''
A new script for glider processing of preswot mission.

Ideally, a lot of material from here should be put in a function (e.g. track projection)
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import seawater
import pyglider as pg
import gsw
import cmocean as cmo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

# ** This should also be in a separate function.
# SeaExplorer limits & target
origin =[38.9425, 2.96987]
target = [37.88908, 2.64376]
timeLims = [pd.to_datetime('2018-05-04 22:00:00'),  pd.to_datetime('2018-05-10 00:00:00')] # transect1
#timeLims = [pd.to_datetime('2018-05-09 20:0:00'),  pd.to_datetime('2018-05-13 23:30:00')] # transect2
#timeLims = [pd.to_datetime('2018-05-13 19:00:00'), pd.to_datetime('2018-05-15 14:00:00')] # transect3

outname = 'correl_transect1.pkl'

## ---- Load both dataset ---- ##
dsSX = xr.open_dataset('/home/cyrf0006/data/gliders_data/netcdf_preswot/SeaExplorer/SEA003_20180503_l2.nc')
latVec_orig = dsSX['latitude'].values
lonVec_orig = dsSX['longitude'].values

#save lat_lon in csv - SeaExplorer
df_lat = dsSX['latitude'].to_pandas()
df_lat.name = 'latitude' 
df_lon = dsSX['longitude'].to_pandas()
df_lon.name = 'longitude'
df = pd.concat([df_lat, df_lon], axis=1) 
df.to_csv('preswot_SX_lat-lon.csv', float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S')
del df_lat, df_lon, df



# Lat/Lon SeaExplorer
dsSX = dsSX.sel(time=slice(timeLims[0], timeLims[1]))
latVec = dsSX['latitude'].values
lonVec = dsSX['longitude'].values
Z = dsSX.depth.values

##  ---- Load deployment file (maybe improve with another dpl file that specifies the processing info) ---- ##
calib_file = '/home/cyrf0006/data/gliders_data/SEA003/20190501/config/Info.dpl'
dpl = {}
with open(calib_file) as f:
    for line in f:
        if (line[0]=='%') | (line == '\n'):
            continue
        else:
            line_core = line.split(';')[0]
            line_core = line_core.split('%')[0]
            line_core = line_core.strip('\n')
            line_core = line_core.replace(" ", "") 
            line_core = line_core.replace("'", "") 
            (key, val) = line_core.split('=')
            dpl[key] = val


       
## ---- 1. SeaExplorer ---- ##
print('Extract SeaExplorer variables:')
for var in list(dsSX.var()):
    print(' -> ' + var)
    exec('sx_' + var + ' = dsSX[var].to_pandas()')
    
# fillna because of stupid way Alseamar handle their sampling (repeated values are removed!)
# ** Need to put this in a function (although not bad now in a loop...)
ppitch = sx_pitch.mean(axis=1) # <0: downcast | >0: upcast
print('Fill NaNs in matrices  - SeaExplorer')
for var in list(dsSX.var()):
    if var != 'profile_index':
        exec('sx_' + var + '[ppitch<0] = sx_' + var + '[ppitch<0].fillna(method="ffill", limit=5, axis=1)')
        exec('sx_' + var + '[ppitch>0] = sx_' + var + '[ppitch>0].fillna(method="bfill", limit=5, axis=1)')
print('  Done!')
                    
# TEOS-10 conversion SeaExplorer
P = sx_pressure.values
lon0, lat0 = dsSX['longitude'].values.mean(), dsSX['latitude'].values.mean()
SA = gsw.SA_from_SP(sx_salinity.values, P, lon0, lat0)
CT = gsw.CT_from_t(SA, sx_temperature.values, P)
sigma0 = gsw.sigma0(SA, CT)
N2, p_mid = gsw.Nsquared(SA, CT, P, lat=lat0)

# Oxygen conversion
Ofreq = sx_oxygen_frequency.values
Soc = float(dpl['calibration.SBE43F.soc'])
Foffset = float(dpl['calibration.SBE43F.foffset'])
A = float(dpl['calibration.SBE43F.a'])
B = float(dpl['calibration.SBE43F.b'])
C = float(dpl['calibration.SBE43F.c'])
E = float(dpl['calibration.SBE43F.e'])

K = CT + 273.15 # Temp in Kelvin
O2sol = gsw.O2sol(SA, CT, P, lon0, lat0); #umol/Kg
O2 =  Soc*(Ofreq+Foffset)*(1.0+A*CT + B*CT**2 + C*CT**3)*O2sol*np.exp(E*P/K); 

# MiniFluo 
DARK = float(dpl['calibration.MFL.DARK']) 
# Get Relative units (e.g., scaled)
TRY_ru = (sx_fluorescence_270_340 - DARK) / (sx_fluorescence_monitoring_270_340 - DARK) 
NAP_ru = TRY_ru.copy() # Same as TRY
PHE_ru = (sx_fluorescence_255_360 - DARK) / (sx_fluorescence_monitoring_255_360 - DARK)
FLU_ru = (sx_fluorescence_260_315 - DARK) / (sx_fluorescence_monitoring_260_315 - DARK)
PYR_ru = (sx_fluorescence_270_376 - DARK) / (sx_fluorescence_monitoring_270_376 - DARK)

# Other variables SeaExplorer
CHL = sx_chlorophyll.values    
CDOM = sx_cdom.values    
BB700 = sx_backscatter_700.values
    
## ---- Projection on definited transect ---- ##
import coord_list
## 1. SeaExplorer
dist, angle = seawater.dist([origin[0], target[0]], [origin[1], target[1]]) # not used 
interval = 20.0 #meters
azimuth = coord_list.calculateBearing(origin[0], origin[1], target[0], target[1]) # this works but angle above not.
coords = np.array(coord_list.main(interval,azimuth,origin[0], origin[1], target[0], target[1]))
lats = coords[:,0]
lons = coords[:,1]

I2 = np.argmin(np.abs(latVec-target[0]) + np.abs(lonVec-target[1]))
I1 = np.argmin(np.abs(latVec-origin[0]) + np.abs(lonVec-origin[1]))
theIndex = np.arange(np.min([I1, I2]), np.max([I1, I2]))
del I1, I2
distVec = np.full_like(theIndex, np.nan, dtype=np.double)  
new_lat = np.full_like(theIndex, np.nan, dtype=np.double)  
new_lon = np.full_like(theIndex, np.nan, dtype=np.double)  
for re_idx, idx in enumerate(theIndex):
    idx_nearest = np.argmin(np.abs(latVec[idx]-lats) + np.abs(lonVec[idx]-lons))
    new_lat[re_idx] = coords[idx_nearest,0]
    new_lon[re_idx] = coords[idx_nearest,1]

    d = seawater.dist([origin[0], lats[idx_nearest]], [origin[1], lons[idx_nearest]])
    distVec[re_idx] = d[0]
    
## ----  Reduce data ---- ##
temp = CT[theIndex, :].T
sal = SA[theIndex, :].T
sig = sigma0[theIndex, :].T
chl = CHL[theIndex, :].T
cdom = CDOM[theIndex, :].T
pres = P[theIndex, :].T
o2 = O2[theIndex, :].T
try_ru = TRY_ru.values[theIndex, :].T
phe_ru = PHE_ru.values[theIndex, :].T
flu_ru = FLU_ru.values[theIndex, :].T
pyr_ru = PYR_ru.values[theIndex, :].T
# Create also distance and Z values
x_grid, z_grid = np.meshgrid(distVec, Z)
lt_grid, z_grid = np.meshgrid(latVec[theIndex], Z)
ln_grid, z_grid = np.meshgrid(lonVec[theIndex], Z)
# Create a time vector
df_t = dsSX['time'].to_pandas()
df_t = df_t.iloc[theIndex]
t_grid, z_grid = np.meshgrid(df_t.values, Z)


## ---- Reshape everything to vectors ---- ##
temp_vector = temp.reshape(chl.size)     
sal_vector = sal.reshape(chl.size)     
sig_vector = sig.reshape(chl.size)     
chl_vector = chl.reshape(chl.size)     
o2_vector = o2.reshape(o2.size)
cdom_vector = cdom.reshape(cdom.size)
pres_vector = pres.reshape(pres.size)
trp_vector = try_ru.reshape(try_ru.size)   
phe_vector = phe_ru.reshape(phe_ru.size)   
flu_vector = flu_ru.reshape(flu_ru.size)   
pyr_vector = pyr_ru.reshape(pyr_ru.size)   
x_vector = x_grid.reshape(x_grid.size)   
z_vector = z_grid.reshape(z_grid.size)   
t_vector = t_grid.reshape(t_grid.size)   
lt_vector = lt_grid.reshape(lt_grid.size)   
ln_vector = ln_grid.reshape(ln_grid.size)   



## ---- Convert to DataFrame and remove NaNs---- ##
df_temp = pd.DataFrame(temp_vector, index = t_vector, columns = ['temperature'])
df_sal = pd.DataFrame(sal_vector, index = t_vector, columns = ['salinity'])
df_sig = pd.DataFrame(sig_vector, index = t_vector, columns = ['sigma0'])
df_chl = pd.DataFrame(chl_vector, index = t_vector, columns = ['chlorophyll'])
df_o2 = pd.DataFrame(o2_vector, index = t_vector, columns = ['oxygen'])
df_cdom = pd.DataFrame(cdom_vector, index = t_vector, columns = ['cdom'])
df_pres = pd.DataFrame(pres_vector, index = t_vector, columns = ['sea_pressure'])
df_trp = pd.DataFrame(trp_vector, index = t_vector, columns = ['trp_like'])
df_phe = pd.DataFrame(phe_vector, index = t_vector, columns = ['phe_like'])
df_flu = pd.DataFrame(flu_vector, index = t_vector, columns = ['flu_like'])
df_pyr = pd.DataFrame(pyr_vector, index = t_vector, columns = ['pyr_like'])
df_x = pd.DataFrame(x_vector, index = t_vector, columns = ['distance'])
df_z = pd.DataFrame(z_vector, index = t_vector, columns = ['depth'])
df_lt = pd.DataFrame(lt_vector, index = t_vector, columns = ['latitude'])
df_ln = pd.DataFrame(ln_vector, index = t_vector, columns = ['longitude'])

df = pd.concat([df_temp, df_sal, df_sig, df_chl, df_o2, df_cdom, df_pres, df_trp, df_phe, df_flu, df_pyr, df_x, df_z, df_lt, df_ln], axis=1)
df = df.dropna(how='any')

df.to_pickle(outname)

keyboard

## ## ---- merge them all ---- ##

df1 = pd.read_pickle('correl_transect1.pkl')
df2 = pd.read_pickle('correl_transect2.pkl')
df3 = pd.read_pickle('correl_transect3.pkl')

# create a new column
transect1 = np.full(df1.index.size, 1)
df1['transect'] = transect1
transect2 = np.full(df2.index.size, 2)
df2['transect'] = transect2
transect3 = np.full(df3.index.size, 3)
df3['transect'] = transect3

df = pd.concat([df1, df2, df3])

## ---- Some cleaning on outliers---- ##

plt.hist(df.trp_like, 1000)
plt.hist(df.phe_like, 1000)
plt.hist(df.flu_like, 1000)
plt.hist(df.pyr_like, 1000)

df = df[df.trp_like<=.16]
df = df[df.phe_like<=.055]
df = df[df.flu_like<=.12]
df = df[df.pyr_like<=.223]



df.to_csv('preswot_all_column_data.csv', float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S')
df.to_pickle('preswot_all_column_data.pkl')



