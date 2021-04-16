'''
Snippet code on how to read AZMP section data from NetCDF archive.

Frederic.Cyr@dfo-mpo.gc.ca
April 2021

'''
import os
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
import azmp_sections_tools as azst
import cmocean
from math import radians, cos, sin, asin, sqrt
import azmp_sections_tools as azst
import gsw
from shapely.geometry import LineString
 
# Function to calculte distance
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km
    df_stn, df_itp = azst.get_section('BB', 2018, 'summer', 'temperature')



## ---- Region parameters (to be edited by user) ---- ##
VAR = 'temperature'
SECTION = 'SI'
SEASON = 'summer'
ZMAX = 300
#FILE = 'SI_2000-2020.nc'
SECTION_BATHY = SECTION
v = np.arange(24,28,.2)
CMAP = cmocean.cm.haline
EXTEND='both'
FRONT_DEPTH = 50

years = np.arange(2000, 2021)

## ---- Retrieve bathymetry using function (not provided here) ---- ##
bathymetry = azst.section_bathymetry(SECTION_BATHY)

# initialize the front position
front =  np.empty((years.shape)) 
front[:] = np.NaN

for i, year in enumerate(years):
    print(str(year))
    #T_stn, T_itp = azst.get_section('SI', year, 'summer', 'temperature')
    #S_stn, S_itp = azst.get_section('SI', year, 'summer', 'salinity')
    T_itp = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/sections_plots/temperature_SI_summer_' + str(year) + '_itp.pkl')
    S_itp = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/sections_plots/salinity_SI_summer_' + str(year) + '_itp.pkl')

    lat = 53
    lon = -53
    SA = gsw.SA_from_SP(S_itp,S_itp.columns,lon, lat)
    CT = gsw.CT_from_t(SA,T_itp,S_itp.columns)
    SIG = gsw.sigma0(SA, CT)
    
    df = pd.DataFrame(SIG, index=S_itp.index, columns=S_itp.columns)
        
    # figure
    plt.close('all')    
    fig = plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0))
    distance = df.index.droplevel(0)
    c = plt.contourf(distance, df.columns, df.T, v, cmap=CMAP, extend=EXTEND)
    ct = plt.contour(distance, df.columns, df.T, v, colors='darkgray')

    # add 27.5 contours
    c_out = plt.contour(distance, df.columns, df.T, [27,], colors='k', linewidths=2)
    ax.set_ylim([0, ZMAX])
    ax.set_ylabel('Depth (m)', fontWeight = 'bold')
    ax.set_xlabel('Distance (km)', fontWeight = 'bold')
    ax.invert_yaxis()
    Bgon = plt.Polygon(bathymetry,color=np.multiply([1,.9333,.6667],.4), alpha=1, zorder=10)
    ax.add_patch(Bgon)
    for i in range(0,len(distance)):
        plt.plot(np.array([distance[i], distance[i]]), np.array([0, ZMAX]), '--k', linewidth=0.5, zorder=5)   
    plt.colorbar(c)
    ax.set_title(VAR + ' for section ' + SECTION + ' - ' + SEASON + ' ' + str(year))

    #HERE!!
    verts = np.shape(c_out.collections[0].get_paths())
    v_tmp = []
    for idx in np.arange(verts[0]):
        v_tmp.append(c_out.collections[0].get_paths()[idx].vertices)
    v1 = np.concatenate(v_tmp)
        
    # Find the front
    xi = distance
    yi = distance*0+FRONT_DEPTH
    ls1 = LineString(v1)
    ls2 = LineString(np.c_[xi, yi])
    points = ls1.intersection(ls2)
    x, y = points.x, points.y
    # plot the front
    plt.plot([x, x], [0, ZMAX], "--r")
    # save the front
    front[i] = x
    del x,y,xi,yi,points,ls1,ls2,v1, c_out
    
    fig.set_size_inches(w=8,h=6)
    fig_name = SECTION + '_front_' + SEASON + '_' + str(year) + '.png' 
    fig.savefig(fig_name, dpi=200)

    
