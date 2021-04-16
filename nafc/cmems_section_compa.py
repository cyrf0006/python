"""
Comparision between AZMP observations on standard sections and CMEMS equivalent.
To be use for Copernicus State of the Ocean report.

Frederic.Cyr@dfo-mpo.gc.ca
January 2021

"""


import re
import pandas as pd
import matplotlib.pyplot as plt
#import pfiles_basics
import numpy as np
#import time as tt
import xarray as xr
import netCDF4
import os
#from sys import version_info
import cmocean
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import interp1d  # to remove NaNs in profiles
from scipy.interpolate import griddata
# AZMP stuff
import azmp_sections_tools as azst


## ---- Some parameters ---- ## 
SECTION = 'SI'
SECTION_BATHY = SECTION
SEASON = 'summer'
VAR = 'temperature'
#VAR = 'salinity'
YEAR = 2020
STATION_BASED=True
ZMAX = 500

if VAR == 'temperature':
    v = np.arange(-2,11,1)
    v_anom = np.linspace(-3.5, 3.5, 15)
    v_anom = np.delete(v_anom, np.where(v_anom==0)) 
    CMAP = cmocean.cm.thermal
    cmems_var = 'Temperature [degrees_C]'
elif VAR == 'salinity':
    v = np.arange(29,36,.5)
    v_anom = np.linspace(-1.5, 1.5, 16)
    CMAP = cmocean.cm.haline    
    cmems_var = 'Salinity [1e-3]'
elif VAR == 'sigma-t':
    v = np.arange(24,28.4,.2)
    v_anom = np.linspace(-1.5, 1.5, 16)
    CMAP = cmocean.cm.haline
else:
    v = 10
    v_anom = 10        


## ---- Get bathymetry ---- ## 
bathymetry = azst.section_bathymetry(SECTION_BATHY)
        
## ---- Get this year's section ---- ## 
df_section_stn, df_section_itp = azst.get_section(SECTION, YEAR, SEASON, VAR)

# Use itp or station-based definition
if STATION_BASED:
    df_section = df_section_stn
else:
    df_section = df_section_itp

## ---- CMEMS data ---- ## 
df = pd.read_csv('SealIsland_2020_global-analysis-forecast-phy-001-024.csv')
df = df[['Depth [m]', 'STATION', cmems_var]]
df = df.pivot(index='Depth [m]', columns='STATION', values=cmems_var)

# Bin data onto df_section depths
depth_range = np.arange(-.5, 2000, 5) # range to look for data
reg_depth = df_section.columns #(equivalent to reg_depth = (depth_range[1:] + depth_range[:-1]) / 2)
df = df.groupby(pd.cut(df.index, depth_range)).mean() # <--- This is cool!
df.index = reg_depth # replace range by mean depth
# vertically fill NaNs
df.interpolate(axis=0, limit_area='inside', inplace=True)
# Rename
df_cmems = df.T
df_cmems.index = df_section.index

## --- Calculate difference ---- ## 
df_diff =  df_cmems - df_section
df_diff = df_diff.reset_index(level=0, drop=True)

## ---- plot Figure ---- ##
XLIM = df_section_itp.index[-1][1]
fig = plt.figure()
# ax1
ax = plt.subplot2grid((3, 1), (0, 0))
if len(df_section.index) > 1 & len(df_section.columns>1):
    c = plt.contourf(df_section.index.droplevel(0), df_section.columns, df_section.T, v, cmap=CMAP, extend='max')
    plt.colorbar(c)
    if VAR == 'temperature':
        c_cil_itp = plt.contour(df_section.index.droplevel(0), df_section.columns, df_section.T, [0,], colors='k', linewidths=2)
ax.set_ylim([0, ZMAX])
ax.set_xlim([0,  XLIM])
ax.set_ylabel('Depth (m)', fontWeight = 'bold')
ax.invert_yaxis()
Bgon = plt.Polygon(bathymetry,color=np.multiply([1,.9333,.6667],.4), alpha=0.8)
ax.add_patch(Bgon)
ax.xaxis.label.set_visible(False)
ax.tick_params(labelbottom='off')
ax.set_title(VAR + ' for section ' + SECTION + ' - ' + SEASON + ' ' + str(YEAR))

# ax2
ax2 = plt.subplot2grid((3, 1), (1, 0))
c = plt.contourf(df_cmems.index.droplevel(0), df_cmems.columns, df_cmems.T, v, cmap=CMAP, extend='max')
plt.colorbar(c)
if VAR == 'temperature':
    c_cil_itp = plt.contour(df_cmems.index.droplevel(0), df_cmems.columns, df_cmems.T, [0,], colors='k', linewidths=2)
ax2.set_ylim([0, ZMAX])
ax2.set_xlim([0,  XLIM])
ax2.set_ylabel('Depth (m)', fontWeight = 'bold')
ax2.invert_yaxis()
Bgon = plt.Polygon(bathymetry,color=np.multiply([1,.9333,.6667],.4), alpha=0.8)
ax2.add_patch(Bgon)
ax2.xaxis.label.set_visible(False)
ax2.tick_params(labelbottom='off')
ax2.set_title('1/12th degree CMEMS global analysis')

# ax3
ax3 = plt.subplot2grid((3, 1), (2, 0))
df_diff.shape
if len(df_section.index) > 1 & len(df_section.columns>1):
    c = plt.contourf(df_diff.index, df_diff.columns, df_diff.T, v_anom, cmap=cmocean.cm.balance, extend='both')
    plt.colorbar(c)
ax3.set_ylim([0, ZMAX])
ax3.set_xlim([0,  XLIM])
ax3.set_ylabel('Depth (m)', fontWeight = 'bold')
ax3.set_xlabel('Distance (km)', fontWeight = 'bold')
ax3.invert_yaxis()
Bgon = plt.Polygon(bathymetry,color=np.multiply([1,.9333,.6667],.4), alpha=0.8)
ax3.add_patch(Bgon)
ax3.set_title(r'CMEMS - Observation')

fig.set_size_inches(w=8,h=12)
fig_name = 'CMEMS_OBS_COMPA_' + VAR + '_' + SECTION + '_' + SEASON + '_' + str(YEAR) + '.png' 
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## ---- Montage Figure ---- ##
command = 'montage CMEMS_OBS_COMPA_temperature_' + SECTION + '_summer_' + str(YEAR) + '.png CMEMS_OBS_COMPA_salinity_' + SECTION + '_summer_' + str(YEAR) + '.png  -tile 2x1 -geometry +10+10  -background white ' + 'CMEMS_OBS_COMPA_' + SECTION + '_' + str(YEAR) + '.png'
os.system(command)
