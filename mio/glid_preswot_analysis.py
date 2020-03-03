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
#import pyglider as pg
import gsw
import cmocean as cmo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import os

# ** This should also be in a separate function.
# SeaExplorer limits & target
origin =[38.9425, 2.96987]
target = [37.88908, 2.64376]
#timeLims = [pd.to_datetime('2018-05-04 22:00:00'),  pd.to_datetime('2018-05-10 00:00:00')] # transect1
timeLims = [pd.to_datetime('2018-05-09 20:0:00'),  pd.to_datetime('2018-05-13 23:30:00')] # transect2
#timeLims = [pd.to_datetime('2018-05-13 19:00:00'), pd.to_datetime('2018-05-15 14:00:00')] # transect3

# Slocum limits and target
origin_sl =[38.9425, 2.91]
target_sl = [37.88908, 2.5847]
#timeLims_sl = [pd.to_datetime('2018-05-05 13:00:00'),  pd.to_datetime('2018-05-09 20:00:00')] # transect1
timeLims_sl = [pd.to_datetime('2018-05-09 13:00:00'),  pd.to_datetime('2018-05-15 8:00:00')] # transect2
#timeLims_sl = []

v1 = np.arange(27,30, .2)
ZMAX = 300
XLIM = 103
XLIM_sl = 120
lonLims = [2.2, 3.2]
latLims = [37.8, 39.3]


## ---- Load both dataset ---- ##
dsSX = xr.open_dataset('/home/cyrf0006/data/gliders_data/netcdf_preswot/SeaExplorer/SEA003_20180503_l2.nc')
dsSL = xr.open_dataset('/home/cyrf0006/data/gliders_data/netcdf_preswot/Slocum/dep0023_sdeep00_scb-sldeep000_L2_2018-05-03_data_dt.nc')
latVec_orig = dsSX['latitude'].values
lonVec_orig = dsSX['longitude'].values
latVec_orig_sl = dsSL['latitude'].values
lonVec_orig_sl = dsSL['longitude'].values

#save lat_lon in csv - SeaExplorer
df_lat = dsSX['latitude'].to_pandas()
df_lat.name = 'latitude' 
df_lon = dsSX['longitude'].to_pandas()
df_lon.name = 'longitude' 
df = pd.concat([df_lat, df_lon], axis=1) 
df.to_csv('preswot_SX_lat-lon.csv', float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S')
del df_lat, df_lon, df
#save lat_lon in csv - Slocum
df_lat = dsSL['latitude'].to_pandas()
df_lat.name = 'latitude' 
df_lon = dsSL['longitude'].to_pandas()
df_lon.name = 'longitude' 
df = pd.concat([df_lat, df_lon], axis=1) 
df.to_csv('preswot_SL_lat-lon.csv', float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S')
del df_lat, df_lon, df

# Lat/Lon SeaExplorer
dsSX = dsSX.sel(time=slice(timeLims[0], timeLims[1]))
latVec = dsSX['latitude'].values
lonVec = dsSX['longitude'].values
Z = dsSX.depth.values
# Lat/Lon Slocum
dsSL = dsSL.sel(time=slice(timeLims_sl[0], timeLims_sl[1]))
latVec_sl = dsSL['latitude'].values
lonVec_sl = dsSL['longitude'].values
Z_sl = dsSL.depth.values

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
SA_sort = np.array(list(map(lambda x, y: y[x], np.argsort(sigma0, axis=1), SA)))
CT_sort = np.array(list(map(lambda x, y: y[x], np.argsort(sigma0, axis=1), CT)))
P_sort = np.array(list(map(lambda x, y: y[x], np.argsort(sigma0, axis=1), P)))
N2, p_mid = gsw.Nsquared(SA_sort, CT_sort, P_sort, lat=lat0, axis=1)

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

# MiniFluo calib (# Assume new version)
TRY_calib = float(dpl['calibration.MFL.TRY_std'])
NAP_calib = float(dpl['calibration.MFL.NAP_spf'])
PHE_calib = float(dpl['calibration.MFL.PHE_spf'])
FLU_calib = float(dpl['calibration.MFL.FLU_std'])
PYR_calib = float(dpl['calibration.MFL.PYR_std'])
TRY_blank = float(dpl['calibration.MFL.TRY_std_blank'])
NAP_blank = float(dpl['calibration.MFL.NAP_spf_blank'])
PHE_blank = float(dpl['calibration.MFL.PHE_spf_blank'])
FLU_blank = float(dpl['calibration.MFL.FLU_std_blank'])
PYR_blank = float(dpl['calibration.MFL.PYR_std_blank'])
DARK = float(dpl['calibration.MFL.DARK']) 
# Get Relative units (e.g., scaled)
TRY_ru = (sx_fluorescence_270_340 - DARK) / (sx_fluorescence_monitoring_270_340 - DARK) 
NAP_ru = TRY_ru.copy() # Same as TRY
PHE_ru = (sx_fluorescence_255_360 - DARK) / (sx_fluorescence_monitoring_255_360 - DARK)
FLU_ru = (sx_fluorescence_260_315 - DARK) / (sx_fluorescence_monitoring_260_315 - DARK)
PYR_ru = (sx_fluorescence_270_376 - DARK) / (sx_fluorescence_monitoring_270_376 - DARK)
# Get concentration (at the moment assume the environment blank is provided, might need an if statement here..)
TRY = (TRY_ru - TRY_blank) / TRY_calib;
NAP = (NAP_ru - NAP_blank) / NAP_calib;
PHE = (PHE_ru - PHE_blank) / PHE_calib;
FLU = (FLU_ru - FLU_blank) / FLU_calib;
PYR = (PYR_ru - PYR_blank) / PYR_calib;

# Other variables SeaExplorer
CHL = sx_chlorophyll.values    
CDOM = sx_cdom.values    
BB700 = sx_backscatter_700.values


## ---- 2. Slocum ---- ##
print('Extract Slocum variables:')
for var in list(dsSL.var()):
    print(' -> ' + var)
    exec('sl_' + var + ' = dsSL[var].to_pandas()')
         
# fillna for slocum
print('Fill NaNs in matrices - slocum')
for var in list(dsSL.var()):
    if var != 'profile_index':
        exec('sl_' + var + ' = sl_' + var + '.fillna(method="pad", limit=5, axis=1)')
print('  Done!')
      
# TEOS-10 conversion Slocum
P_sl = sl_pressure.values
SA_sl = gsw.SA_from_SP(sl_salinity.values, P_sl, lon0, lat0)
CT_sl = gsw.CT_from_t(SA_sl, sl_temperature.values, P_sl)
sigma0_sl = gsw.sigma0(SA_sl, CT_sl)
SA_sl_sort = np.array(list(map(lambda x, y: y[x], np.argsort(sigma0_sl, axis=1), SA_sl)))
CT_sl_sort = np.array(list(map(lambda x, y: y[x], np.argsort(sigma0_sl, axis=1), CT_sl)))
P_sl_sort = np.array(list(map(lambda x, y: y[x], np.argsort(sigma0_sl, axis=1), P_sl)))
N2_sl, p_mid_sl = gsw.Nsquared(SA_sl_sort, CT_sl_sort, P_sl_sort, lat=lat0, axis=1)

# Other variables Slocum
CHL_sl = sl_chlorophyll.values    
TUR_sl = sl_turbidity.values  
O2_sl = sl_oxygen_concentration.values    

    
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

    
## 2. Slocum
azimuth_sl = coord_list.calculateBearing(origin_sl[0], origin_sl[1], target_sl[0], target_sl[1]) # this works but angle above not.
coords_sl = np.array(coord_list.main(interval,azimuth_sl,origin_sl[0], origin_sl[1], target_sl[0], target_sl[1]))
lats_sl = coords_sl[:,0]
lons_sl = coords_sl[:,1]

I2 = np.argmin(np.abs(latVec_sl-target_sl[0]) + np.abs(lonVec_sl-target_sl[1]))
I1 = np.argmin(np.abs(latVec_sl-origin_sl[0]) + np.abs(lonVec_sl-origin_sl[1]))
theIndex_sl = np.arange(np.min([I1, I2]), np.max([I1, I2]))

distVec_sl = np.full_like(theIndex_sl, np.nan, dtype=np.double)  
new_lat_sl = np.full_like(theIndex_sl, np.nan, dtype=np.double)  
new_lon_sl = np.full_like(theIndex_sl, np.nan, dtype=np.double)  
for re_idx, idx in enumerate(theIndex_sl):
    idx_nearest = np.argmin(np.abs(latVec_sl[idx]-lats_sl) + np.abs(lonVec_sl[idx]-lons_sl))
    new_lat_sl[re_idx] = coords_sl[idx_nearest,0]
    new_lon_sl[re_idx] = coords_sl[idx_nearest,1]

    d = seawater.dist([origin_sl[0], lats_sl[idx_nearest]], [origin_sl[1], lons_sl[idx_nearest]])
    distVec_sl[re_idx] = d[0]
    
        
# Map to check SeaExplorer:
projection=ccrs.Mercator()
extent = [lonLims[0], lonLims[1], latLims[0], latLims[1]]
fig = plt.figure(figsize=(13.3, 10))                      
ax = fig.add_subplot(111, projection=projection)
lon_labels = np.arange(lonLims[0]-2, lonLims[1]+2, 1)
lat_labels = np.arange(latLims[0]-2, latLims[1]+2, 1)
gl = ax.gridlines(draw_labels=True, xlocs=lon_labels, ylocs=lat_labels)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'fontsize':15}
gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'fontsize':15}
ax.set_extent(extent, crs=ccrs.PlateCarree())
coastline_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                        edgecolor='k', alpha=1,
                                        facecolor=cfeature.COLORS['land'], zorder=100)
ax.add_feature(coastline_10m)

plt.plot(lonVec_orig, latVec_orig) 
plt.plot([origin[1], target[1]], [origin[0], target[0]], 'k') 
plt.plot(lonVec[theIndex], latVec[theIndex], '.m') 
plt.plot(new_lon, new_lat, '.r')  
# Add Gates Locations
plt.plot(lonVec_orig, latVec_orig,
         color='steelblue', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot([origin[1], target[1]], [origin[0], target[0]],
         color='black', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(lonVec[theIndex], latVec[theIndex],
         color='m', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(new_lon, new_lat,
         color='red', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.legend(['glider track', 'projection', 'glider yos', 'projected position' ])
plt.title('Map projection SeaExplorer')  
# Save figure
fig_name = 'preswot_transect2_projection.png'
fig.set_size_inches(w=18, h=12)
fig.savefig(fig_name, dpi=200)
#print('Figure trimmed!')
#os.system('convert -trim ' + fig_name + ' ' + fig_name)

# Map to check Slocum:
projection=ccrs.Mercator()
extent = [lonLims[0], lonLims[1], latLims[0], latLims[1]]
fig = plt.figure(figsize=(13.3, 10))                      
ax = fig.add_subplot(111, projection=projection)
lon_labels = np.arange(lonLims[0]-2, lonLims[1]+2, 1)
lat_labels = np.arange(latLims[0]-2, latLims[1]+2, 1)
gl = ax.gridlines(draw_labels=True, xlocs=lon_labels, ylocs=lat_labels)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'black', 'weight': 'bold', 'fontsize':15}
gl.ylabel_style = {'color': 'black', 'weight': 'bold', 'fontsize':15}
ax.set_extent(extent, crs=ccrs.PlateCarree())
coastline_10m = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                        edgecolor='k', alpha=1,
                                        facecolor=cfeature.COLORS['land'], zorder=100)
ax.add_feature(coastline_10m)

# plot glider track and projection
plt.plot(lonVec_orig_sl, latVec_orig_sl) 
plt.plot([origin_sl[1], target_sl[1]], [origin_sl[0], target_sl[0]], 'k') 
plt.plot(lonVec_sl[theIndex_sl], latVec_sl[theIndex_sl], '.m') 
plt.plot(new_lon_sl, new_lat_sl, '.r')  
plt.plot(lonVec_orig_sl, latVec_orig_sl,
         color='steelblue', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.plot([origin_sl[1], target_sl[1]], [origin_sl[0], target_sl[0]],
         color='black', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(lonVec_sl[theIndex_sl], latVec_sl[theIndex_sl],
         color='m', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.scatter(new_lon_sl, new_lat_sl,
         color='red', alpha=.9,
         transform=ccrs.PlateCarree(), zorder=20
         )
plt.legend(['glider track', 'projection', 'glider yos', 'projected position' ])   
plt.title('Map projection Slocum')  
# Save figure
fig_name = 'preswot_sl_transect2_projection.png'
fig.set_size_inches(w=18, h=12)
fig.savefig(fig_name, dpi=200)
#print('Figure trimmed!')
#os.system('convert -trim ' + fig_name + ' ' + fig_name)



## fig = plt.figure()
## ax = plt.subplot2grid((1, 1), (0, 0))
## plt.plot(lonVec_orig, latVec_orig) 
## plt.plot(lonVec[theIndex], latVec[theIndex], '.m') 
## plt.plot([origin[1], target[1]], [origin[0], target[0]], 'k') 
## plt.plot(new_lon, new_lat, '.r')  
## plt.ylabel('Latitude', fontWeight = 'bold')
## plt.xlabel('Longitude', fontWeight = 'bold')

# ****Cleaning to be implemented:
## %% Clean trajectory (remove repetition + backward trajectory)
## % ---  Version 1  --- %
## theIndexUnique = theIndex;
## Inegative = find(diff(distVec)<0);
## p = polyfit(1:length(distVec), distVec, 1);
## if p(1)>0 % origin -> target
##     while ~isempty(Inegative)
##         Inegative = find(diff(distVec)<0);
##         if ~isempty(Inegative)
##             distVec(Inegative+1) = [];
##             theIndexUnique(Inegative+1) = [];
##         end
##         Inegative = find(diff(distVec)<0);
##     end
## else % target -> origin (return trajectory)
##     while ~isempty(Inegative)
##         Inegative = find(diff(distVec)>0);
##         if ~isempty(Inegative)
##             distVec(Inegative+1) = [];
##             theIndexUnique(Inegative+1) = [];
##         end
##         Inegative = find(diff(distVec)>0);
##     end
## end
## [distVec_unique, I] = unique(distVec, 'last');
## distVec = interp1(theIndexUnique(I), distVec_unique, theIndex);

## if ~isempty(varargin)
##     theIndex = index_orig(theIndex);
## end



    
## ## ---- Some figures ---- ##
# T-S diagram
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
x = np.reshape(distVec, [distVec.size,1])
x = np.repeat(x, repeats=Z.size, axis=1) 
plt.scatter(SA[theIndex, :], CT[theIndex, :], c=x) 
cbar = plt.colorbar()
ax.set_ylabel(r'$\rm \Theta$ ($\rm ^{\circ}C$)', fontWeight = 'bold')
ax.set_xlabel(r'$\rm S_A$ ($\rm g\,Kg^{-1}$)', fontWeight = 'bold')
ax.set_title(r'T-S diagram')
cbar.ax.set_ylabel('Along-transect distance (Km)')
fig_name = 'preswot_transect2_TS.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# T-S diagram Slocum
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
x = np.reshape(distVec_sl, [distVec_sl.size,1])
x = np.repeat(x, repeats=Z_sl.size, axis=1) 
plt.scatter(SA_sl[theIndex_sl, :], CT_sl[theIndex_sl, :], c=x) 
cbar = plt.colorbar()
ax.set_ylabel(r'$\rm \Theta$ ($\rm ^{\circ}C$)', fontWeight = 'bold')
ax.set_xlabel(r'$\rm S_A$ ($\rm g\,Kg^{-1}$)', fontWeight = 'bold')
ax.set_title(r'T-S diagram')
cbar.ax.set_ylabel('Along-transect distance (Km)')
fig_name = 'preswot_sl_transect2_TS.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

## ---- Figures SeaExplorer ---- ##

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, CT[theIndex, :].T, cmap=cmo.cm.thermal)
plt.clim([13, 18])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Conservative Temperature ($\rm ^{\circ}C$)')
fig_name = 'preswot_transect2_temperature.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, SA[theIndex, :].T, cmap=cmo.cm.haline)
plt.clim([37.3, 38.8])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Absolute Salinity ($\rm g\,Kg^{-1}$)')
fig_name = 'preswot_transect2_salinity.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# *N2*
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z[0:-1]+(np.diff(Z[0:2])/2), np.log10(N2[theIndex, :].T), cmap=cmo.cm.balance)
plt.clim([-5, -3])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'$\rm N^2$ ($\rm s^{-2}$)')
fig_name = 'preswot_transect2_buoy.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, O2[theIndex, :].T, cmap=cmo.cm.ice)
plt.clim([150, 250])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Oxygen concentration ($\rm \mu mol\,Kg^{-1}$)')
fig_name = 'preswot_transect2_oxygen.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
  
fig = plt.figure()
# ax1
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, CHL[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([0, 1.4])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Chlorophyll-a concentration ($\rm \mu g\,L^{-1}$)')
fig_name = 'preswot_transect2_chl.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
#c = plt.pcolormesh(distVec, Z, TRY.values[theIndex, :].T, cmap=cmo.cm.algae)
#plt.clim([1.5, 3])
c = plt.pcolormesh(distVec, Z, TRY_ru.values[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([.1, .15])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
#ax.set_title(r'Try-like concentration ($\rm ng\,L^{-1}$)')
ax.set_title(r'peak T (FI, relative units)')
fig_name = 'preswot_transect2_try.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
#c = plt.pcolormesh(distVec, Z, PHE.values[theIndex, :].T*1000)
#plt.clim([15, 22])
c = plt.pcolormesh(distVec, Z, PHE_ru.values[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([.035, .05])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
#ax.set_title(r'Phe-like concentration ($\rm ng\,L^{-1}$)')
ax.set_title(r'peak A/M (FI, relative units)')
fig_name = 'preswot_transect2_phe.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

## fig = plt.figure()
## ax = plt.subplot2grid((1, 1), (0, 0))
## c = plt.pcolormesh(distVec, Z, NAP.values[theIndex, :].T*1000)
## plt.clim([100, 300])
## cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
## ax.set_ylim([0, ZMAX])
## ax.set_xlim([0,  XLIM])
## ax.set_ylabel('Depth (m)', fontWeight = 'bold')
## ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
## ax.invert_yaxis()
## plt.colorbar(c)
## ax.set_title(r'Naph-like concentration ($\rm ng\,L^{-1}$)')
## fig_name = 'preswot_transect2_naph.png'
## fig.savefig(fig_name, dpi=150)
## os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
#c = plt.pcolormesh(distVec, Z, FLU.values[theIndex, :].T*1000)
#plt.clim([-100, 50])
c = plt.pcolormesh(distVec, Z, FLU_ru.values[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([.07, .12])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
#ax.set_title(r'Flu-like concentration ($\rm ng\,L^{-1}$)')
ax.set_title(r'peak B (FI, relative units)')
fig_name = 'preswot_transect2_flu.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
#c = plt.pcolormesh(distVec, Z, PYR.values[theIndex, :].T*1000)
#plt.clim([1300, 1700])
c = plt.pcolormesh(distVec, Z, PYR_ru.values[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([.17, .2])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
#ax.set_title(r'Pyr-like concentration ($\rm ng\,L^{-1}$)')
ax.set_title(r'peak N (FI, relative units)')
fig_name = 'preswot_transect2_pyr.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
# ax1
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, CDOM[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([1, 1.5])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'peak C (IF, QSU)')
fig_name = 'preswot_transect2_cdom.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## ---- Figures transect Slocum ---- ##

# CT
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec_sl, Z_sl, CT_sl[theIndex_sl, :].T, cmap=cmo.cm.thermal)
plt.clim([13, 18])
cc = plt.contour(distVec_sl, Z_sl, sigma0_sl[theIndex_sl, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM_sl])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Conservative Temperature ($\rm ^{\circ}C$)')
fig_name = 'preswot_sl_transect2_temperature.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# SA
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec_sl, Z_sl, SA_sl[theIndex_sl, :].T, cmap=cmo.cm.haline)
plt.clim([37.3, 38.8])
cc = plt.contour(distVec_sl, Z_sl, sigma0_sl[theIndex_sl, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM_sl])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Absolute Salinity ($\rm g\,Kg^{-1}$)')
fig_name = 'preswot_sl_transect2_salinity.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# *N2*
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec_sl, Z_sl[0:-1]+(np.diff(Z_sl[0:2])/2), np.log10(N2_sl[theIndex_sl, :].T), cmap=cmo.cm.balance)
plt.clim([-5, -3])
cc = plt.contour(distVec_sl, Z_sl, sigma0_sl[theIndex_sl, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM_sl])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'$\rm N^2$ ($\rm s^{-2}$)')
fig_name = 'preswot_sl_transect2_buoy.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# O2
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec_sl, Z_sl, O2_sl[theIndex_sl, :].T, cmap=cmo.cm.ice)
plt.clim([150, 250])
cc = plt.contour(distVec_sl, Z_sl, sigma0_sl[theIndex_sl, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM_sl])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Oxygen concentration ($\rm \mu mol\,Kg^{-1}$)')
fig_name = 'preswot_sl_transect2_oxygen.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# CHL
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec_sl, Z_sl, CHL_sl[theIndex_sl, :].T, cmap=cmo.cm.algae)
plt.clim([0, 1.4])
cc = plt.contour(distVec_sl, Z_sl, sigma0_sl[theIndex_sl, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM_sl])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Chlorophyll-a concentration ($\rm \mu g\,L^{-1}$)')
fig_name = 'preswot_sl_transect2_chl.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# Turbidity
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec_sl, Z_sl, TUR_sl[theIndex_sl, :].T, cmap=cmo.cm.turbid)
plt.clim([1, 1.25])
cc = plt.contour(distVec_sl, Z_sl, sigma0_sl[theIndex_sl, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM_sl])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Turbidity (NTU)')
fig_name = 'preswot_sl_transect2_cdom.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## fig = plt.figure()
## # ax1
## ax = plt.subplot2grid((1, 1), (0, 0))
## c = plt.pcolormesh(distVec, Z, np.log10(BB700[theIndex, :].T), cmap=cmo.cm.turbid)
## plt.clim([-5, -4])
## ax.set_xlim([0,  XLIM])
## cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
## ax.set_ylim([0, ZMAX])
## ax.set_ylabel('Depth (m)', fontWeight = 'bold')
## ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
## ax.invert_yaxis()
## plt.colorbar(c)
## ax.set_title(r'$\rm log_{10} (BB700 / m^{-1})$')
## fig_name = 'preswot_transect2_bb700.png'
## fig.savefig(fig_name, dpi=150)
## os.system('convert -trim ' + fig_name + ' ' + fig_name)

## ---- profiles plots ---- ##

chl = CHL[theIndex, :].T
cdom = CDOM[theIndex, :].T
pres = P[theIndex, :].T
o2 = O2[theIndex, :].T
try_ru = TRY_ru.values[theIndex, :].T
phe_ru = PHE_ru.values[theIndex, :].T
flu_ru = FLU_ru.values[theIndex, :].T
pyr_ru = PYR_ru.values[theIndex, :].T

x0 = 85
xf = 105
z0 = 0
zf = 100

## chl_prof = np.nanmean(chl[(Z>=z0) & (Z<=zf),(distVec>=x0) & (distVec<=xf)], axis=1)
## trp_prof = np.nanmean(try_ru[(Z>=z0) & (Z<=zf),(distVec>=x0) & (distVec<=xf)], axis=1)
## phe_prof = np.nanmean(phe_ru[(Z>=z0) & (Z<=zf),(distVec>=x0) & (distVec<=xf)], axis=1)
## flu_prof = np.nanmean(flu_ru[(Z>=z0) & (Z<=zf),(distVec>=x0) & (distVec<=xf)], axis=1)
## pyr_prof = np.nanmean(pyr_ru[(Z>=z0) & (Z<=zf),(distVec>=x0) & (distVec<=xf)], axis=1)

## plt.plot(chl_prof/np.nanmax(chl_prof), Z)
## plt.plot(try_prof/np.nanmax(try_prof), Z)
## plt.plot(phe_prof/np.nanmax(phe_prof), Z)
## plt.plot(flu_prof/np.nanmax(flu_prof), Z)
## plt.plot(pyr_prof/np.nanmax(pyr_prof), Z)
## plt.gca().invert_yaxis()
## plt.legend(['chl', 'try', 'phe', 'flu', 'pyr'])

# Restrict fluorophores in depth and distance ranges.
z_idx = (Z>=z0) & (Z<=zf)
x_idx = (distVec>=x0) & (distVec<=xf)

chl_vector = chl[np.squeeze(np.where(z_idx)),:]
chl_vector = chl_vector[:,np.squeeze(np.where(x_idx))]
chl_vector = chl_vector.reshape(chl_vector.size)     

o2_vector = o2[np.squeeze(np.where(z_idx)),:]
o2_vector = o2_vector[:,np.squeeze(np.where(x_idx))]
o2_vector = o2_vector.reshape(o2_vector.size)

cdom_vector = cdom[np.squeeze(np.where(z_idx)),:]
cdom_vector = cdom_vector[:,np.squeeze(np.where(x_idx))]
cdom_vector = cdom_vector.reshape(cdom_vector.size)

pres_vector = pres[np.squeeze(np.where(z_idx)),:]
pres_vector = pres_vector[:,np.squeeze(np.where(x_idx))]
pres_vector = pres_vector.reshape(pres_vector.size)

trp_vector = try_ru[np.squeeze(np.where(z_idx)),:]
trp_vector = trp_vector[:,np.squeeze(np.where(x_idx))]
trp_vector = trp_vector.reshape(trp_vector.size)   

phe_vector = phe_ru[np.squeeze(np.where(z_idx)),:]
phe_vector = phe_vector[:,np.squeeze(np.where(x_idx))]
phe_vector = phe_vector.reshape(phe_vector.size)   

flu_vector = flu_ru[np.squeeze(np.where(z_idx)),:]
flu_vector = flu_vector[:,np.squeeze(np.where(x_idx))]
flu_vector = flu_vector.reshape(flu_vector.size)

pyr_vector = pyr_ru[np.squeeze(np.where(z_idx)),:]
pyr_vector = pyr_vector[:,np.squeeze(np.where(x_idx))]
pyr_vector = pyr_vector.reshape(pyr_vector.size)   

idx_good = ~np.isnan(chl_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]
flu_vector = flu_vector[idx_good]
pyr_vector = pyr_vector[idx_good]

idx_good = ~np.isnan(trp_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]
flu_vector = flu_vector[idx_good]
pyr_vector = pyr_vector[idx_good]

idx_good = ~np.isnan(phe_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]
flu_vector = flu_vector[idx_good]
pyr_vector = pyr_vector[idx_good]

idx_good = ~np.isnan(flu_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]
flu_vector = flu_vector[idx_good]
pyr_vector = pyr_vector[idx_good]
idx_good = ~np.isnan(pyr_vector)

chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]
flu_vector = flu_vector[idx_good]
pyr_vector = pyr_vector[idx_good]

idx_good = ~np.isnan(o2_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]
flu_vector = flu_vector[idx_good]
pyr_vector = pyr_vector[idx_good]

idx_good = ~np.isnan(cdom_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]
flu_vector = flu_vector[idx_good]
pyr_vector = pyr_vector[idx_good]

idx_good = ~np.isnan(pres_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]
flu_vector = flu_vector[idx_good]
pyr_vector = pyr_vector[idx_good]

plt.scatter(chl_vector, trp_vector, c=pres_vector)   

## For all transect, single variables
# montage preswot_transect1_salinity.png preswot_sl_transect1_salinity.png preswot_transect2_salinity.png preswot_sl_transect2_salinity.png preswot_transect3_salinity.png -tile 2x3 -geometry +10+10  -background white  preswot_salinity.png
# montage preswot_transect1_temperature.png preswot_sl_transect1_temperature.png preswot_transect2_temperature.png preswot_sl_transect2_temperature.png preswot_transect3_temperature.png -tile 2x3 -geometry +10+10  -background white  preswot_temperature.png
# montage preswot_transect1_chl.png preswot_sl_transect1_chl.png preswot_transect2_chl.png preswot_sl_transect2_chl.png preswot_transect3_chl.png -tile 2x3 -geometry +10+10  -background white  preswot_chl.png
# montage preswot_transect1_oxygen.png preswot_sl_transect1_oxygen.png preswot_transect2_oxygen.png preswot_sl_transect2_oxygen.png preswot_transect3_oxygen.png -tile 2x3 -geometry +10+10  -background white  preswot_oxygen.png

## For PAHS transects
# montage preswot_transect1_chl.png preswot_transect1_cdom.png preswot_transect1_try.png preswot_transect1_phe.png preswot_transect1_flu.png preswot_transect1_pyr.png -tile 2x3 -geometry +10+10  -background white  preswot_pahs_transect1.png
# montage preswot_transect2_chl.png preswot_transect2_cdom.png preswot_transect2_try.png preswot_transect2_phe.png preswot_transect2_flu.png preswot_transect2_pyr.png -tile 2x3 -geometry +10+10  -background white  preswot_pahs_transect2.png


## For transect montage
# montage preswot_transect1_temperature.png preswot_transect1_salinity.png preswot_transect1_oxygen.png preswot_transect1_chl.png preswot_transect1_cdom.png preswot_transect1_try.png -tile 2x3 -geometry +10+10  -background white  preswot_transect1.png
# montage preswot_transect2_temperature.png preswot_transect2_salinity.png preswot_transect2_oxygen.png preswot_transect2_chl.png preswot_transect2_cdom.png preswot_transect2_try.png -tile 2x3 -geometry +10+10  -background white  preswot_transect2.png
