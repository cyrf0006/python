'''
Test EOFs using XEOFS.
Example borrowed from here:

https://xeofs.readthedocs.io/en/latest/auto_examples/1eof/plot_eof-smode.html

Frederic.Cyr@dfo-mpo.gc.ca
October 2023

'''

# Load packages and data:
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import EqualEarth, PlateCarree
import numpy as np
from xeofs.models import EOF
import os
import cmocean as cmo

var = 'tem'
var = 'sal'

## derived parameters
if var == 'sal':
    datafile = '/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-sal-rean-m_1698921096294.nc'
    myvar = 'so'
    VARLABEL = 'S'
    VS = np.linspace(37.5, 39, 16)
    CMAPS = cmo.cm.haline
elif var == 'tem':
    datafile = '/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-tem-rean-m_1698921282139.nc'
    myvar = 'thetao'
    VARLABEL = r'T ($^{\circ}$C)'
    VS = np.linspace(10, 27, 18)
    CMAPS = cmo.cm.thermal
    

##  ---- Load and prepare the data ---- ##
ds = xr.open_dataset(datafile)

# Seasonal averages
ds = ds.resample(time='Q').mean()

# Select 3 sections:
dsN = ds.where((ds['lat'] == 41.979168), drop=True) 
dsC = ds.where((ds['lat'] == 40.979168), drop=True) 
dsS = ds.where((ds['lat'] == 40.520832), drop=True) 

## Loop on years
for i in np.arange(1999, 2021):

    print('Processing year ' + str(i))
    
    ## 1. Spring data
    dfN = dsN.where((dsN['time.year'] == i) & (dsN['time.month'] == 6), drop=True)[myvar].squeeze(drop=True).to_pandas()
    dfC = dsC.where((dsN['time.year'] == i) & (dsN['time.month'] == 6), drop=True)[myvar].squeeze(drop=True).to_pandas()
    dfS = dsS.where((dsN['time.year'] == i) & (dsN['time.month'] == 6), drop=True)[myvar].squeeze(drop=True).to_pandas()

    # plot North
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfN.keys(), dfN.index, dfN.values, VS, extend='both', cmap=CMAPS)
    plt.ylabel('Depth (m)')
    plt.xlabel(r'Longitude ($^{\circ}$E)')
    plt.ylim([0, 875])
    plt.xlim([.75, 6])
    plt.title(r'North ($42^{\circ}$N) - Spring')
    plt.text(5.4, 850, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(r'S', fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=7.5, h=3)
    figname = var + '_spring_north_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    # Plot Center
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfC.keys(), dfC.index, dfC.values, VS, extend='both', cmap=CMAPS)
    plt.ylabel('Depth (m)', fontsize=11, fontweight='bold')
    plt.xlabel(r'Longitude ($^{\circ}$E)')
    plt.ylim([0, 875])
    plt.xlim([.75, 6])
    plt.title('Center ($41^{\circ}$N) - Spring')
    plt.text(5.4, 850, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(VARLABEL, fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=7.5, h=3)
    figname = var + '_spring_center_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    # Plot South
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfS.keys(), dfS.index, dfS.values, VS, extend='both', cmap=CMAPS)
    plt.ylabel('Depth (m)', fontsize=11, fontweight='bold')
    plt.ylim([0, 875])
    plt.xlim([.75, 6])
    plt.title('South ($40.5^{\circ}$N) - Spring')
    plt.text(5.4, 850, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(VARLABEL, fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=7.5, h=3)
    figname = var + '_spring_south_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    ## 2. Summer data
    dfN = dsN.where((dsN['time.year'] == i) & (dsN['time.month'] == 9), drop=True)[myvar].squeeze(drop=True).to_pandas()
    dfC = dsC.where((dsN['time.year'] == i) & (dsN['time.month'] == 9), drop=True)[myvar].squeeze(drop=True).to_pandas()
    dfS = dsS.where((dsN['time.year'] == i) & (dsN['time.month'] == 9), drop=True)[myvar].squeeze(drop=True).to_pandas()

    # plot North
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfN.keys(), dfN.index, dfN.values, VS, extend='both', cmap=CMAPS)
    plt.ylabel('Depth (m)', fontsize=11, fontweight='bold')
    plt.ylim([0, 875])
    plt.xlim([.75, 6])
    plt.title(r'North ($42^{\circ}$N) - Summer')
    plt.text(5.4, 850, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(VARLABEL, fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=7.5, h=3)
    figname = var + '_summer_north_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    # Plot Center
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfC.keys(), dfC.index, dfC.values, VS, extend='both', cmap=CMAPS)
    plt.ylabel('Depth (m)', fontsize=11, fontweight='bold')
    plt.ylim([0, 875])
    plt.xlim([.75, 6])
    plt.title('Center ($41^{\circ}$N) - Summer')
    plt.text(5.4, 850, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(VARLABEL, fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=7.5, h=3)
    figname = var + '_summer_center_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    # Plot South
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfS.keys(), dfS.index, dfS.values, VS, extend='both', cmap=CMAPS)
    plt.ylabel('Depth (m)', fontsize=11, fontweight='bold')
    plt.ylim([0, 875])
    plt.xlim([.75, 6])
    plt.title('South ($40.5^{\circ}$N) - Summer')
    plt.text(5.4, 850, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(VARLABEL, fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=7.5, h=3)
    figname = var + '_summer_south_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    
## Montage
