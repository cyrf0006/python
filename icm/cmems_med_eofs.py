import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime

from eofs.xarray import Eof
import pickle

# To define the coordinate system
import cartopy.crs as ccrs
# To add the pre-defined features; all are small-scale (1:110m) Natural Earth datasets
import cartopy.feature as cfeature
# To add attributes used to determine draw time behaviour of the gridlines and labels
# Used in 1st plotting method
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# Used in 2nd plotting method
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# To support completely configurable tick locating and formatting
import matplotlib.ticker as mticker



## Some options:
calculate_EOFs = False
annual = False

#ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/PP/cmems_mod_glo_bgc_my_0.25_P1M-m_1695129763017.nc')
#ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-sal-rean-m_1695807438765.nc')
ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-ssh-rean-m_1695807984977.nc')
# Keep only <2020 (2021 not complete)
ds = ds.sel(time=ds.time.dt.year<=2021)

# Put the coordinate variables (lat, lon) into xarray (numpy) arrays
lats = ds['lat']
lons = ds['lon']



if annual:
    ## 1. Annual average
    ssh_raw = ds['zos']
    annual_mean = ssh_raw.groupby('time.year').mean('time')
    ssh = annual_mean.rename({'year':'time'})

    # Create an EOF solver to do the EOF analysis and save it.. 
    solver = Eof(ssh)
    pickle_out = open('solver_ssh_annual.pkl', 'wb')
    pickle.dump(solver, pickle_out)
    pickle_out.close()

    # Retrieve the EOF up to order 6
    eof = solver.eofsAsCorrelation(neofs=6)
    # Retrieve the PCs time series up to order 6
    pc = solver.pcs(npcs=6, pcscaling=1)

    # Saving calculated eofs
    pickle_out = open('eof_ssh_annual.pkl', 'wb')
    # Use the highest protocol (-1) because it is way faster than the default
    eof_ssh = pickle.dump(eof, pickle_out, protocol=-1)
    pickle_out.close()

    # Saving calculated pcs (similar to saving eofs)
    pickle_out = open('pc_ssh_annual.pkl', 'wb')
    pc_ssh = pickle.dump(pc, pickle_out, protocol=-1)
    pickle_out.close()
    
else: 
    ## 2. Monthly with seasonal cycle removed
    ssh_raw = ds['zos']
    monthly_clim = ssh_raw.groupby('time.month').mean('time')
    ssh = ssh_raw.groupby('time.month') - monthly_clim

    # Create an EOF solver to do the EOF analysis and save it.. 
    solver = Eof(ssh)
    pickle_out = open('solver_ssh_monthly.pkl', 'wb')
    pickle.dump(solver, pickle_out)
    pickle_out.close()

    # Retrieve the EOF up to order 6
    eof = solver.eofsAsCorrelation(neofs=6)
    # Retrieve the PCs time series up to order 6
    pc = solver.pcs(npcs=6, pcscaling=1)

    # Saving calculated eofs
    pickle_out = open('eof_ssh_monthly.pkl', 'wb')
    # Use the highest protocol (-1) because it is way faster than the default
    eof_ssh = pickle.dump(eof, pickle_out, protocol=-1)
    pickle_out.close()

    # Saving calculated pcs (similar to saving eofs)
    pickle_out = open('pc_ssh_monthly.pkl', 'wb')
    pc_ssh = pickle.dump(pc, pickle_out, protocol=-1)
    pickle_out.close()

# Saving calculated lats (similar to saving temp)
#pickle_out = open('lat.pickle', 'wb')
#lat = pickle.dump(lat, pickle_out, protocol=-1)
#pickle_out.close()
    

## ------ Could start here using pickle stuff, but actually goes pretty fast ------ ##
## FC revisited until here. See after glorys_eofs_cartopy_maps.py


# Load calculated EOFs
if annual:
    pickle_in = open('solver_ssh_annual.pkl','rb')
else:
    pickle_in = open('solver_ssh_monthly.pkl','rb')
        
solver = pickle.load(pickle_in)
eof = solver.eofsAsCorrelation(neofs=6)
pc = solver.pcs(npcs=6, pcscaling=1)
var_frac = solver.varianceFraction(6)

    
# To create EOF figure with six subplots sharing both x and y axis
fig_eof, axes_eof = plt.subplots(nrows=1, ncols=2, figsize=(8,6), sharey=True, subplot_kw = dict(projection=ccrs.PlateCarree()))

# To create PC figure with six subplots sharing only x axis
fig_pc, axes_pc = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharey=True)

# To adjust the height reserved for space between subplots and the top of the subplots
fig_eof.subplots_adjust(wspace=0.12, hspace=0.12, top = 0.94)
fig_pc.subplots_adjust(wspace=0.12, hspace=0.25, top = 0.87)

# To loop over all EOFs and PCs data to plot 
for i in range(2):
    
    # Range for the Mediterranean
    axes_eof.flat[i].set_extent([-6, 37, 30, 46], ccrs.PlateCarree())
    
    # To specify the color levels for the map
    clevs = np.linspace(-1, 1, 21)
    
    # To make a filled contour plot 
    fill = axes_eof.flat[i].contourf(lons, lats, eof[i].squeeze(), 
                                     clevs,
                                     transform = ccrs.PlateCarree(), 
                                     cmap = plt.cm.seismic,
                                     extend='both') # To set the colorbar extend shape
    
    # To draw polygon edges to  contourf(), add line contours with calls to contour()
    axes_eof.flat[i].contour(lons, lats, eof[i].squeeze(), 
                             clevs, 
                             transform=ccrs.PlateCarree(), 
                             cmap=plt.cm.seismic) # RdBu_r and jet work better than Spectral_r
    
    # To add land polygons, including major islands (Land masking)
    axes_eof.flat[i].add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='k') 

    # To add natural and artificial lakes
    axes_eof.flat[i].add_feature(cfeature.LAKES)

    # To create a feature for States/Province regions at 1:50m from Natural Earth Data
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    # To add state and provincial boundaries and polygons for the United States and Canada
    axes_eof.flat[i].add_feature(states_provinces, edgecolor='w')
    
    # To determine the tick locations (or the gridlines)
    #axes_eof.flat[i].set_xticks([-70, -60, -50, -40, -30], crs=ccrs.PlateCarree())
    #axes_eof.flat[i].set_yticks([30, 40, 50, 60, 70], crs=ccrs.PlateCarree())
    #axes_eof.flat[i].set_ylim(32, 72)
    #axes_eof.flat[i].set_xlim(-70,-30)

    # To define our format for lat and lon tick lables
    lon_formatter = LongitudeFormatter(number_format='.0f', # To whether include decimal 
                                       degree_symbol='', # To whether drop degree symbol 
                                       dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.0f',
                                      degree_symbol='')
    
    # To override the default locator using the __call__ method; Pass it to x,y instance 
    axes_eof.flat[i].xaxis.set_major_formatter(lon_formatter)
    axes_eof.flat[i].yaxis.set_major_formatter(lat_formatter)
    
    # To set the axes grids
    axes_eof.flat[i].grid(linewidth = 0.8, 
                          color = 'gray', 
                          alpha = 0.3,   # float (0.0 transparent through 1.0 opaque) 
                          linestyle = '--')
    
    # To add title to each subplot
    axes_eof.flat[i].set_title('EOF mode order = %d (%d%% variance)' %(i+1, np.round(var_frac[i]*100)))
    #axes_eof.flat[i].set_ylabel('Latitude')
    axes_eof.flat[i].set_xlabel('Longitude')
    
    #### Plot the PCs ####
    # To plot PC data: 2nd Method (make filled polygons between two curves)
    x= pc['time'].values   # Converting datetime to numpy array
    x = pd.to_datetime(x, format='%Y').values
    y= pc[:, i]
    axes_pc.flat[i].fill_between(x, y, where=np.array(y>=0), facecolor='red', interpolate=True)
    axes_pc.flat[i].fill_between(x, y, where=np.array(y<=0), facecolor='blue', interpolate=True)
    
    #axes_pc.flat[i].plot(NAO_anom.index, NAO_anom.rolling(window=12, center=True).mean(), 'k')

    # To set y axis limits
    axes_pc.flat[i].set_ylim(-2.1, 2.1)
    axes_pc.flat[i].grid()
    #axes_pc.flat[i].set_xlim([pd.to_datetime('1993'), pd.to_datetime('2016')])

    # To add title to each subplot
    axes_pc.flat[i].set_title('PC mode order = %d' %(i+1))
    
    # To set view limits to data limits; Only for x axis
    axes_pc.flat[i].set_ylabel('Normalized units')

axes_eof.flat[0].set_ylabel('Latitude')
#axes_pc.flat[0].set_ylabel('Normalized units')
axes_pc.flat[0].get_xaxis().set_ticklabels([])
axes_pc.flat[1].set_xlabel('Year')  

# To add a colorbar to the plot; This makes the colorbar look a bit out of place
cb = fig_eof.colorbar(fill, 
                      ax = axes_eof, 
                      orientation='horizontal', 
                      shrink=.7) # To multiply the size of the colorbar

# To improve and adjust the colorbar position
# To get the original position of colorbar
pos = cb.ax.get_position()

# To shift the colorbar up a bit
cb.ax.set_position([pos.x0, pos.y0 + 0.08, pos.width, pos.height])

# To set the colorbar label
cb.set_label('correlation coefficient', fontsize=12)

# To add figure title
#fig_eof.suptitle('Possible spatial modes of variablity of temperature data', fontsize=16)
#fig_pc.suptitle('Principal component time series of temperature data', fontsize=16)

# To save figures
if annual:
     fig_eof.savefig("eof_annual.png",dpi=200, bbox_inches='tight')
     fig_pc.savefig("pc_annual.png",dpi=200, bbox_inches='tight')
else:
    fig_eof.savefig("eof_monthly.png",dpi=200, bbox_inches='tight')
    fig_pc.savefig("pc_monthly.png",dpi=200, bbox_inches='tight')


keyboard

### ----- Second figure with combined PCs ----- ###
fig_comb, axes_comb = plt.subplots(nrows=1, ncols=1, figsize=(8,6))

# To adjust the height reserved for space between subplots and the top of the subplots
fig_comb.subplots_adjust(wspace=0.12, hspace=0.25, top = 0.87)

df_combine = pd.Series(data=(pc[:, 0]+pc[:, 1])/2.0, index=pc['time'].values)
x = df_combine.index
x = pd.to_datetime(x, format='%Y').values
y = df_combine.rolling(window=5, center=True).mean()
axes_comb.fill_between(x, 0, y, where=y >= 0, facecolor='red', interpolate=True)
axes_comb.fill_between(x, 0, y, where=y <= 0, facecolor='blue', interpolate=True)
#axes_comb.plot(Sanomaly_nut.index, Sanomaly_nut.rolling(window=12, center=True).mean(), '--k', linewidth=2)
#axes_comb.plot(NO3_deep_anom.index, NO3_deep_anom.rolling(window=3, center=True).mean(), 'k', linewidth=3)
# To set y axis limits
axes_comb.set_ylim(-1.6, 1.6)
axes_comb.grid()
axes_comb.set_xlim([pd.to_datetime('1993'), pd.to_datetime('2016')])
axes_comb.legend(['NO3'], loc=3)

# To add title to each subplot
axes_comb.set_title('Average (PC1 & PC2)')

# To set view limits to data limits; Only for x axis
#plt.autoscale(enable=True, axis='x', tight=True)
axes_comb.set_ylabel('Normalized units')
axes_comb.set_xlabel('Year')  
# To save figures
fig_comb.savefig("pc_1and2_temp_dpt100m.png",dpi=200, bbox_inches='tight')


### ----- Statistics ----- ###
y2 = NO3_deep_anom
y2 = y2[y2.index.year<=2015]

df_y1 = pd.Series(data=pc[:, 0], index=pc['time'].values)
y1 = df_y1.resample('As').mean()
y1 = y1[y1.index.year>=1999]
pd.np.corrcoef(y1.values.flatten(), y2.values.flatten())

df_y1 = pd.Series(data=pc[:, 1], index=pc['time'].values)
y1 = df_y1.resample('As').mean()
y1 = y1[y1.index.year>=1999]
pd.np.corrcoef(y1.values.flatten(), y2.values.flatten())

y1 = df_combine.rolling(window=5, center=True).mean()
y1 = y1.resample('As').mean()
y1 = y1[y1.index.year>=1999]
pd.np.corrcoef(y1.values.flatten(), y2.values.flatten())
