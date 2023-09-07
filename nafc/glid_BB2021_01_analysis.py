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
import gsw
import cmocean as cmo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import os

# ** This should also be in a separate function.
# SeaExplorer limits & target
origin =[48.73, -52.97]
#target = [50.00, -49.00] # original BB section
target = [50.00, -50.7] # BB modified
timeLims = [pd.to_datetime('2021-07-21 15:30:00'),  pd.to_datetime('2021-07-30 12:35:00')] # transect out
#timeLims = [pd.to_datetime('2021-07-30 12:30:00'), pd.to_datetime('2021-08-08 17:10:00')] # transect in
downcast_only = False
upcast_only = True
check_casts = True # check for empty casts
empty_thresh = .9 # ignore casts if 95% empty

v1 = np.arange(24,30, .25)
ZMAX = 350
XLIM = 225
XLIM_sl = 120
lonLims = [-54, -48]
latLims = [47, 51]


## ---- Load dataset ---- ##
ds = xr.open_dataset('/home/cyrf0006/data/gliders_data/SEA024/20210721/netcdf/SEA024_20210721_l2.nc')
latVec_orig = ds['latitude'].values
lonVec_orig = ds['longitude'].values

#save lat_lon in csv - SeaExplorer
df_lat = ds['latitude'].to_pandas()
df_lat.name = 'latitude' 
df_lon = ds['longitude'].to_pandas()
df_lon.name = 'longitude' 
df = pd.concat([df_lat, df_lon], axis=1) 
df.to_csv('BB2021_03_SX_lat-lon.csv', float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S')
del df_lat, df_lon, df

# Lat/Lon SeaExplorer
ds = ds.sel(time=slice(timeLims[0], timeLims[1]))
latVec = ds['latitude'].values
lonVec = ds['longitude'].values
Z = ds.depth.values

##  ---- Load deployment file (maybe improve with another dpl file that specifies the processing info) ---- ##
calib_file = '/home/cyrf0006/data/gliders_data/SEA024/20210721/config/Info.dpl'
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
    print(' -> ' + var)
    exec('sx_' + var + ' = ds[var].to_pandas()')
    
# fillna because of stupid way Alseamar handle their sampling (repeated values are removed!)
# ** Need to put this in a function (although not bad now in a loop...)
ppitch = sx_pitch.mean(axis=1) # <0: downcast | >0: upcast
print('Fill NaNs in matrices  - SeaExplorer')
for var in list(ds.var()):
    if var != 'profile_index':
        exec('sx_' + var + '[ppitch<0] = sx_' + var + '[ppitch<0].fillna(method="ffill", limit=5, axis=1)')
        exec('sx_' + var + '[ppitch>0] = sx_' + var + '[ppitch>0].fillna(method="bfill", limit=5, axis=1)')
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

# clean N2
df_N2 = pd.DataFrame(N2, index=sx_salinity.index, columns=sx_salinity.keys()[1:,])    
df_N2 = df_N2.fillna(method="pad", limit=5, axis=1)
N2x = df_N2.values

# Oxygen conversion
## Ofreq = sx_oxygen_frequency.values
## Soc = float(dpl['calibration.SBE43F.soc'])
## Foffset = float(dpl['calibration.SBE43F.foffset'])
## A = float(dpl['calibration.SBE43F.a'])
## B = float(dpl['calibration.SBE43F.b'])
## C = float(dpl['calibration.SBE43F.c'])
## E = float(dpl['calibration.SBE43F.e'])

## K = CT + 273.15 # Temp in Kelvin
## O2sol = gsw.O2sol(SA, CT, P, lon0, lat0); #umol/Kg
## O2 =  Soc*(Ofreq+Foffset)*(1.0+A*CT + B*CT**2 + C*CT**3)*O2sol*np.exp(E*P/K); 

# With the Rinko:
O2 = sx_oxygen_concentration.values

# MiniFluo calib (# Assume new version)
TRY_calib = float(dpl['calibration.MFL.TRY_std'])
NAP_calib = float(dpl['calibration.MFL.NAP_spf'])
PHE_calib = float(dpl['calibration.MFL.PHE_spf'])
TRY_blank = float(dpl['calibration.MFL.TRY_std_blank'])
NAP_blank = float(dpl['calibration.MFL.NAP_spf_blank'])
PHE_blank = float(dpl['calibration.MFL.PHE_spf_blank'])
DARK = float(dpl['calibration.MFL.DARK']) 
# Get Relative units (e.g., scaled)
TRY_ru = (sx_fluorescence_270_340 - DARK) / (sx_fluorescence_monitoring_270_340 - DARK) 
NAP_ru = TRY_ru.copy() # Same as TRY
PHE_ru = (sx_fluorescence_255_360 - DARK) / (sx_fluorescence_monitoring_255_360 - DARK)
# Get concentration (at the moment assume the environment blank is provided, might need an if statement here..)
TRY = (TRY_ru - TRY_blank) / TRY_calib;
NAP = (NAP_ru - NAP_blank) / NAP_calib;
PHE = (PHE_ru - PHE_blank) / PHE_calib;

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

# Check if downcast only
if downcast_only:
    theIndexCTD = theIndex[ppitch[theIndex]<0]
    distVecCTD = distVec[ppitch[theIndex]<0]
if upcast_only:
    theIndexCTD = theIndex[ppitch[theIndex]>0]
    distVecCTD = distVec[ppitch[theIndex]>0]
else:
    theIndexCTD = theIndex
    distVecCTD = distVec


# Check if empty cast (when sampling every 2 casts...)
if check_casts:
    # the Index
    good_casts = sx_temperature.iloc[theIndex].isna().sum(axis=1)/sx_temperature.shape[1]<empty_thresh
    theIndex = theIndex[good_casts]
    distVec = distVec[good_casts]
    print(str(len(good_casts[good_casts==False])) + ' casts ignored')
    # the Index CTD
    good_castsCTD = sx_temperature.iloc[theIndexCTD].isna().sum(axis=1)/sx_temperature.shape[1]<empty_thresh
    theIndexCTD = theIndexCTD[good_castsCTD]
    distVecCTD = distVecCTD[good_castsCTD]    
    
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
fig_name = 'BB2021_01_transect1_projection.png'
fig.set_size_inches(w=18, h=12)
fig.savefig(fig_name, dpi=200)
#print('Figure trimmed!')
#os.system('convert -trim ' + fig_name + ' ' + fig_name)


## ---- Some figures ---- ##
# T-S diagram
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
x = np.reshape(distVecCTD, [distVecCTD.size,1])
x = np.repeat(x, repeats=Z.size, axis=1) 
plt.scatter(SA[theIndexCTD, :], CT[theIndexCTD, :], c=x) 
cbar = plt.colorbar()
ax.set_ylabel(r'$\rm \Theta$ ($\rm ^{\circ}C$)', fontweight = 'bold')
ax.set_xlabel(r'$\rm S_A$ ($\rm g\,Kg^{-1}$)', fontweight = 'bold')
ax.set_title(r'T-S diagram')
cbar.ax.set_ylabel('Along-transect distance (Km)')
fig_name = 'TB2021_3_transect1_TS.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)



## ---- Figures SeaExplorer ---- ##

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVecCTD, Z, CT[theIndexCTD, :].T, cmap=cmo.cm.thermal)
plt.clim([-2, 10])
cc = plt.contour(distVecCTD, Z, sigma0[theIndexCTD, :].T, v1, colors='lightgray', linewidths=1)
plt.clabel(cc, inline=1, fontsize=10, colors='lightgrey', fmt='%.2f')
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontweight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontweight = 'bold')
#ax.text(10,340,'A', fontweight = 'bold')
ax.text(10,340,'A - Bonavista', fontweight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Conservative Temperature ($\rm ^{\circ}C$)')
fig_name = 'BB2021_01_transect1_temperature.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVecCTD, Z, SA[theIndexCTD, :].T, cmap=cmo.cm.haline)
plt.clim([31.5, 34.5])
cc = plt.contour(distVecCTD, Z, sigma0[theIndexCTD, :].T, v1, colors='lightgray', linewidths=1)
plt.clabel(cc, inline=1, fontsize=10, colors='lightgrey', fmt='%.2f')
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontweight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontweight = 'bold')
ax.text(10,340,'B', fontweight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Absolute Salinity ($\rm g\,Kg^{-1}$)')
fig_name = 'BB2021_01_transect1_salinity.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

# *N2*
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVecCTD, Z[0:-1]+(np.diff(Z[0:2])/2), np.log10(N2x[theIndexCTD, :].T), cmap=cmo.cm.balance)
#c = plt.contourf(distVec, Z[0:-1]+(np.diff(Z[0:2])/2), np.log10(N2[theIndex, :].T), cmap=cmo.cm.balance)
plt.clim([-5, -3])
cc = plt.contour(distVecCTD, Z, sigma0[theIndexCTD, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontweight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontweight = 'bold')
ax.text(10,340,'D', fontweight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'$\rm N^2$ ($\rm s^{-2}$)')
fig_name = 'BB2021_01_transect1_buoy.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, O2[theIndex, :].T, cmap=cmo.cm.ice)
plt.clim([280, 440])
cc = plt.contour(distVecCTD, Z, sigma0[theIndexCTD, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontweight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontweight = 'bold')
ax.text(10,340,'C', fontweight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Oxygen concentration ($\rm \mu mol\,L^{-1}$)')
fig_name = 'BB2021_01_transect1_oxygen.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
  
fig = plt.figure()
# ax1
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, CHL[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([0, 1.4])
cc = plt.contour(distVecCTD, Z, sigma0[theIndexCTD, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontweight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontweight = 'bold')
ax.text(10,340,'E', fontweight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'Chlorophyll-a concentration ($\rm \mu g\,L^{-1}$)')
fig_name = 'BB2021_01_transect1_chl.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
#c = plt.pcolormesh(distVec, Z, TRY.values[theIndex, :].T, cmap=cmo.cm.algae)
#plt.clim([2.5, 5])
c = plt.pcolormesh(distVec, Z, TRY_ru.values[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([0.2, 0.4])
cc = plt.contour(distVecCTD, Z, sigma0[theIndexCTD, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontweight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontweight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
#ax.set_title(r'Try-like concentration ($\rm ng\,L^{-1}$)')
ax.set_title(r'peak T (FI, relative units)')
fig_name = 'BB2021_01_transect1_try.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
#c = plt.pcolormesh(distVec, Z, PHE.values[theIndex, :].T*1000)
#plt.clim([15, 22])
c = plt.pcolormesh(distVec, Z, PHE_ru.values[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([.12, .16])
cc = plt.contour(distVecCTD, Z, sigma0[theIndexCTD, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontweight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontweight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
#ax.set_title(r'Phe-like concentration ($\rm ng\,L^{-1}$)')
ax.set_title(r'peak A/M (FI, relative units)')
fig_name = 'BB2021_01_transect1_phe.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


fig = plt.figure()
# ax1
ax = plt.subplot2grid((1, 1), (0, 0))
c = plt.pcolormesh(distVec, Z, CDOM[theIndex, :].T, cmap=cmo.cm.algae)
plt.clim([2, 2.6])
cc = plt.contour(distVecCTD, Z, sigma0[theIndexCTD, :].T, v1, colors='lightgray', linewidths=1)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontweight = 'bold')
ax.set_xlabel('Along-transect distance (km)', fontweight = 'bold')
ax.text(10,340,'F', fontweight = 'bold')
ax.invert_yaxis()
plt.colorbar(c)
ax.set_title(r'peak C (IF, QSU)')
fig_name = 'BB2021_01_transect1_cdom.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)



## ---- profiles plots ---- ##

chl = CHL[theIndex, :].T
cdom = CDOM[theIndex, :].T
pres = P[theIndex, :].T
o2 = O2[theIndex, :].T
try_ru = TRY_ru.values[theIndex, :].T
phe_ru = PHE_ru.values[theIndex, :].T

x0 = 85
xf = 105
z0 = 0
zf = 100


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

idx_good = ~np.isnan(chl_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]

idx_good = ~np.isnan(o2_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]

idx_good = ~np.isnan(cdom_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]

idx_good = ~np.isnan(pres_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]

idx_good = ~np.isnan(trp_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]

idx_good = ~np.isnan(phe_vector)
chl_vector = chl_vector[idx_good]
o2_vector = o2_vector[idx_good]
cdom_vector = cdom_vector[idx_good]
pres_vector = pres_vector[idx_good]
trp_vector = trp_vector[idx_good]
phe_vector = phe_vector[idx_good]

plt.scatter(chl_vector, cdom_vector, c=pres_vector)   

#montage BB2021_01_transect1_temperature.png BB2021_01_transect1_salinity.png BB2021_01_transect1_oxygen.png BB2021_01_transect1_buoy.png BB2021_01_transect1_chl.png BB2021_01_transect1_cdom.png -tile 2x3 -geometry +10+10  -background white  BB2021_01_transect1.png

# montage BB2021_01_transect1_temperature.png BB2021_01_transect1_salinity.png  BB2021_01_transect1_try.png BB2021_01_transect1_phe.png BB2021_01_transect1_chl.png BB2021_01_transect1_cdom.png -tile 2x3 -geometry +10+10  -background white  BB2021_01_transect1_minifluo.png

