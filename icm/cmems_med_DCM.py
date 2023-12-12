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

##  ---- Load and prepare the data ---- ##
ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-sal-rean-m_1698918649350.nc')

# Seasonal averages
ds = ds.resample(time='Q').mean()

# Select 3 sections:
dsN = ds.where((ds['latitude'] == 41.979168), drop=True) 
dsC = ds.where((ds['latitude'] == 40.979168), drop=True) 
dsS = ds.where((ds['latitude'] == 40.520832), drop=True) 

V = np.linspace(0, 1, 21)
## Loop on years
for i in np.arange(1999, 2021):

    print('Processing year ' + str(i))
    
    ## 1. Spring data
    dfN = dsN.where((dsN['time.year'] == i) & (dsN['time.month'] == 6), drop=True)['chl'].squeeze(drop=True).to_pandas()
    dfC = dsC.where((dsN['time.year'] == i) & (dsN['time.month'] == 6), drop=True)['chl'].squeeze(drop=True).to_pandas()
    dfS = dsS.where((dsN['time.year'] == i) & (dsN['time.month'] == 6), drop=True)['chl'].squeeze(drop=True).to_pandas()

    # plot North
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfN.keys(), dfN.index, dfN.values, V, extend='both', cmap=cmo.cm.algae)
    plt.ylabel('Depth (m)')
    plt.xlabel(r'Longitude ($^{\circ}$E)')
    plt.ylim([0, 150])
    plt.xlim([.75, 6])
    plt.title(r'North ($42^{\circ}$N) - Spring')
    plt.text(5.4, 145, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(r'[Chl] ($\rm mg m^{-3}$)', fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=6, h=3)
    figname = 'chl_spring_north_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    # Plot Center
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfC.keys(), dfC.index, dfC.values, V, extend='both', cmap=cmo.cm.algae)
    plt.ylabel('Depth (m)', fontsize=15, fontweight='bold')
    plt.xlabel(r'Longitude ($^{\circ}$E)')
    plt.ylim([0, 150])
    plt.xlim([.75, 6])
    plt.title('Center ($41^{\circ}$N) - Spring')
    plt.text(5.4, 145, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(r'[Chl] ($\rm mg m^{-3}$)', fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=6, h=3)
    figname = 'chl_spring_center_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    # Plot South
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfS.keys(), dfS.index, dfS.values, V, extend='both', cmap=cmo.cm.algae)
    plt.ylabel('Depth (m)', fontsize=15, fontweight='bold')
    plt.ylim([0, 150])
    plt.xlim([.75, 6])
    plt.title('South ($40.5^{\circ}$N) - Spring')
    plt.text(5.4, 145, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(r'[Chl] ($\rm mg m^{-3}$)', fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=6, h=3)
    figname = 'chl_spring_south_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    ## 2. Summer data
    dfN = dsN.where((dsN['time.year'] == i) & (dsN['time.month'] == 9), drop=True)['chl'].squeeze(drop=True).to_pandas()
    dfC = dsC.where((dsN['time.year'] == i) & (dsN['time.month'] == 9), drop=True)['chl'].squeeze(drop=True).to_pandas()
    dfS = dsS.where((dsN['time.year'] == i) & (dsN['time.month'] == 9), drop=True)['chl'].squeeze(drop=True).to_pandas()

    # plot North
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfN.keys(), dfN.index, dfN.values, V, extend='both', cmap=cmo.cm.algae)
    plt.ylabel('Depth (m)', fontsize=15, fontweight='bold')
    plt.ylim([0, 150])
    plt.xlim([.75, 6])
    plt.title(r'North ($42^{\circ}$N) - Summer')
    plt.text(5.4, 145, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(r'[Chl] ($\rm mg m^{-3}$)', fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=6, h=3)
    figname = 'chl_summer_north_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    # Plot Center
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfC.keys(), dfC.index, dfC.values, V, extend='both', cmap=cmo.cm.algae)
    plt.ylabel('Depth (m)', fontsize=15, fontweight='bold')
    plt.ylim([0, 150])
    plt.xlim([.75, 6])
    plt.title('Center ($41^{\circ}$N) - Summer')
    plt.text(5.4, 145, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(r'[Chl] ($\rm mg m^{-3}$)', fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=6, h=3)
    figname = 'chl_summer_center_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    # Plot South
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    c = plt.contourf(dfS.keys(), dfS.index, dfS.values, V, extend='both', cmap=cmo.cm.algae)
    plt.ylabel('Depth (m)', fontsize=15, fontweight='bold')
    plt.ylim([0, 150])
    plt.xlim([.75, 6])
    plt.title('South ($40.5^{\circ}$N) - Summer')
    plt.text(5.4, 145, str(i))
    ax.invert_yaxis()
    cax = fig.add_axes([0.91, .15, 0.01, 0.7])
    cb = plt.colorbar(c, cax=cax, orientation='vertical')
    cb.set_label(r'[Chl] ($\rm mg m^{-3}$)', fontsize=12, fontweight='normal')
    # Save Figure
    fig.set_size_inches(w=6, h=3)
    figname = 'chl_summer_south_' + str(i) + '.png'
    fig.savefig(figname, dpi=200)
    os.system('convert -trim ' + figname + ' ' + figname)

    
## Montage
