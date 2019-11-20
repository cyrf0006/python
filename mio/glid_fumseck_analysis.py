'''
A new script for glider processing.
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
import os

# ** This should also be in a separate function.
# SeaExplorer limits & target
target =[43.204, 8.543]
origin = [43.73, 8.35]
timeLims = [pd.to_datetime('2019-05-01 9:00:00'),  pd.to_datetime('2019-05-04 05:00:00')] # transect1
#timeLims = [pd.to_datetime('2019-05-04 04:00:00'),  pd.to_datetime('2019-05-06 04:00:00')] # transect2

# profiles to flag:
idx_flag = [39, 40, 81]


v1 = np.arange(27,30, .2)
ZMAX = 300
XLIM = 50
lonLims = [7, 10]
latLims = [42, 45]

## ---- Load both dataset ---- ##
ds = xr.open_dataset('/home/cyrf0006/data/gliders_data/SEA003/20190501/netcdf/SEA003_20190501_l2.nc')
latVec_orig = ds['latitude'].values
lonVec_orig = ds['longitude'].values

# Lat/Lon SeaExplorer
ds = ds.sel(time=slice(timeLims[0], timeLims[1]))
latVec = ds['latitude'].values
lonVec = ds['longitude'].values
Z = ds.depth.values


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
for var in list(ds.var()):
    if var != 'profile_index':
        print(' -> ' + var)
        exec('sx_' + var + ' = ds[var].to_pandas()')

# fillna because of stupid way Alseamar handle their sampling (repeated values are removed!)
# ** Need to put this in a function (although not bad now in a loop...)
ppitch = sx_pitch.mean(axis=1) # <0: downcast | >0: upcast
print('Fill NaNs in matrices  - SeaExplorer')
for var in list(ds.var()):
    if var != 'profile_index':
        exec('sx_' + var + '[ppitch<0] = sx_' + var + '[ppitch<0].fillna(method="ffill", limit=10, axis=1)')
        exec('sx_' + var + '[ppitch>0] = sx_' + var + '[ppitch>0].fillna(method="bfill", limit=10, axis=1)')
        exec('sx_' + var + ' = sx_' + var + '[~pd.isna(sx_' + var + '.iloc[:,0:50].mean(axis=1))]') # correction specific for this deployment
print('  Done!')

         
# TEOS-10 conversion SeaExplorer
P = sx_pressure.values
lon0, lat0 = ds['longitude'].values.mean(), ds['latitude'].values.mean()
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
distVec = np.full_like(theIndex, np.nan, dtype=np.double)  
new_lat = np.full_like(theIndex, np.nan, dtype=np.double)  
new_lon = np.full_like(theIndex, np.nan, dtype=np.double)  
for re_idx, idx in enumerate(theIndex):
    idx_nearest = np.argmin(np.abs(latVec[idx]-lats) + np.abs(lonVec[idx]-lons))
    new_lat[re_idx] = coords[idx_nearest,0]
    new_lon[re_idx] = coords[idx_nearest,1]

    d = seawater.dist([origin[0], lats[idx_nearest]], [origin[1], lons[idx_nearest]])
    distVec[re_idx] = d[0]




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
fig_name = 'fumseck_transect1_projection.png'
fig.set_size_inches(w=18, h=12)
fig.savefig(fig_name, dpi=200)
#print('Figure trimmed!')
#os.system('convert -trim ' + fig_name + ' ' + fig_name)


    
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
fig_name = 'fumseck_transect1_TS.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## ---- Figures SeaExplorer ---- ##

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, CT[theIndex, :].T, cmap=cmo.cm.thermal)
plt.clim([13, 16])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Conservative Temperature ($\rm ^{\circ}C$)')
fig_name = 'fumseck_transect1_temperature.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, SA[theIndex, :].T, cmap=cmo.cm.haline)
plt.clim([38.2, 39])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Absolute Salinity ($\rm g\,Kg^{-1}$)')
fig_name = 'fumseck_transect1_salinity.png'
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
fig_name = 'fumseck_transect1_buoy.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, O2[theIndex, :].T, cmap=cmo.cm.ice)
plt.clim([180, 300])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Oxygen concentration ($\rm \mu mol\,Kg^{-1}$)')
fig_name = 'fumseck_transect1_oxygen.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
  
fig = plt.figure()
# ax1
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, CHL[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([0, 2])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Chlorophyll-a concentration ($\rm \mu g\,L^{-1}$)')
fig_name = 'fumseck_transect1_chl.png'
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
fig_name = 'fumseck_transect1_try.png'
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
fig_name = 'fumseck_transect1_phe.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
#c = plt.pcolormesh(distVec, Z, FLU.values[theIndex, :].T*1000)
#plt.clim([-100, 50])
c = plt.pcolormesh(distVec, Z, FLU_ru.values[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([.07, .15])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
#ax.set_title(r'Flu-like concentration ($\rm ng\,L^{-1}$)')
ax.set_title(r'peak B (FI, relative units)')
fig_name = 'fumseck_transect1_flu.png'
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
fig_name = 'fumseck_transect1_pyr.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
# ax1
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, CDOM[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([.1, .6])
cc = plt.contour(distVec, Z, sigma0[theIndex, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontWeight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'peak C (IF, QSU)')
fig_name = 'fumseck_transect1_cdom.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)



# montage fumseck_transect1_chl.png fumseck_transect1_cdom.png fumseck_transect1_try.png fumseck_transect1_phe.png fumseck_transect1_flu.png fumseck_transect1_pyr.png -tile 2x3 -geometry +10+10  -background white  fumseck_pahs_transect1.png
# montage fumseck_transect2_chl.png fumseck_transect2_cdom.png fumseck_transect2_try.png fumseck_transect2_phe.png fumseck_transect2_flu.png fumseck_transect2_pyr.png -tile 2x3 -geometry +10+10  -background white  fumseck_pahs_transect2.png

#montage fumseck_transect2_temperature.png fumseck_transect2_salinity.png fumseck_transect2_oxygen.png fumseck_transect2_chl.png fumseck_transect2_cdom.png fumseck_transect2_try.png  -tile 2x3 -geometry +10+10  -background white  fumseck_transect2.png

#montage fumseck_transect1_temperature.png fumseck_transect1_salinity.png fumseck_transect1_oxygen.png fumseck_transect1_chl.png fumseck_transect1_cdom.png fumseck_transect1_try.png  -tile 2x3 -geometry +10+10  -background white  fumseck_transect1.png
