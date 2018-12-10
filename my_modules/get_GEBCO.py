# get_GEBCO - Extract subset of worldwide GEBCO bathymetry

# INPUTS:
# - nc_file
# - lims = [lat_min lat_max lon_min lon_max]
#
# OUTPUT:
# -lat (1xm vector)
#- lon (1xn vector)
# - z (nxm matrix).
#
# usage ex:
# lat, lon, z = get_GEBCO.main('/home/cyrf0006/Data/GEBCO/GEBCO_08.nc', [-52, -42, -60, -42])
#  
    
# Extract from GEBCO documentation:

# The gridded data are stored in a netCDF data file.
#
# Within the GEBCO_08 Grid and GEBCO_08 SID Grid netCDF files, the
# grids are stored as one dimensional arrays of 2-byte siged
# integer values.

#The complete data sets provide global coverages. Each data set
#consists of 21,600 rows x 43,200 columns, resulting in a total of
#933,120,000 data points. The data start at the Northwest corner of
#the file, i.e. for the global file, position 89 59'45"N,
#179 59'45"W, and are arranged in latitudinal bands of 360 degrees
#x 120 points/degree = 43,200 values. The daa range eastward from
#179 59'45"W to 179 59'45"E. Thus, the first band contains 43,200
#values for 89 59'45"N, then followed by a band of 43,200 values at
#89 59'15"N and so on at 30 arc-second latitude intervals down to
#89 59'45"S.

#The data values are pixel centred registered i.e. they refer to
#elevations at the centre of grid cells. For example, the data
#point at 179 59'45"W, 89 59'45"N represents the data value at the
#centre of a grid cell of dimension 30 arc-seconds of latitude and
#longitude centred on this position i.e. 179 59'30"W - 180 W;
#89 59'30"N - 90 N. 
    
# Frederic.Cyr@dfo-mpo.gc.ca - April 2017


import numpy as np
import netCDF4


def main(nc_file, lims):
    '''
    This version is kept for legacy, but is similar to version08. Will be remove soon.
    '''
    dataset = netCDF4.Dataset(nc_file)
    lat_min = lims[0]
    lat_max = lims[1]
    lon_min = lims[2]
    lon_max = lims[3]
    x = dataset.variables['x_range']
    y = dataset.variables['y_range']
    z = dataset.variables['z']
    spacing = dataset.variables['spacing']
    nx = (x[-1]-x[0])/spacing[0]   # num pts in x-dir
    ny = (y[-1]-y[0])/spacing[1]   # num pts in y-dir

    lon = np.linspace(x[0],x[-1],nx)
    lat = np.linspace(y[0],y[-1],ny)

    # Define bounding box
    BB = dict(
        lon=[lon_min, lon_max],
        lat=[lat_min, lat_max]
        )

    # nonzero returns a tuple of idx per dimension
    # we're unpacking the tuple here so we can lookup max and min
    (latidx,) = np.logical_and(lat >= BB['lat'][0], lat < BB['lat'][1]).nonzero()
    (lonidx,) = np.logical_and(lon >= BB['lon'][0], lon < BB['lon'][1]).nonzero()

    # initial count
    line_skip = latidx[0]-1
    start = line_skip*len(lon)+lonidx[0];

    # loop to append indices
    zz = [];
    for i in  range(len(latidx)):
        lineidx = np.arange(start,start+len(lonidx))
        lineZ = dataset.variables['z'][lineidx]
        zz = np.concatenate((zz,lineZ))
        start = start + len(lon)
    # get rid of the non used lat/lon now
    lat = lat[latidx]
    lon = lon[lonidx]

    # reshape topo
    Z = zz[:].reshape( len(lat), len(lon))

    ## import matplotlib.pyplot as plt
    ## v = [-2000, -1000, -500, -100, 0]
    ## plt.contour(lon,lat,Z,v)
    ## plt.colorbar()
    ## plt.gca().invert_yaxis()
    ## plt.show()
        
    return lat,lon,Z

def version08(nc_file, lims):
    '''
    for GEBCO_08.nc file
    '''
    dataset = netCDF4.Dataset(nc_file)
    lat_min = lims[0]
    lat_max = lims[1]
    lon_min = lims[2]
    lon_max = lims[3]
    x = dataset.variables['x_range']
    y = dataset.variables['y_range']
    z = dataset.variables['z']
    spacing = dataset.variables['spacing']
    nx = (x[-1]-x[0])/spacing[0]   # num pts in x-dir
    ny = (y[-1]-y[0])/spacing[1]   # num pts in y-dir

    lon = np.linspace(x[0],x[-1],nx)
    lat = np.linspace(y[0],y[-1],ny)

    # Define bounding box
    BB = dict(
        lon=[lon_min, lon_max],
        lat=[lat_min, lat_max]
        )

    # nonzero returns a tuple of idx per dimension
    # we're unpacking the tuple here so we can lookup max and min
    (latidx,) = np.logical_and(lat >= BB['lat'][0], lat < BB['lat'][1]).nonzero()
    (lonidx,) = np.logical_and(lon >= BB['lon'][0], lon < BB['lon'][1]).nonzero()

    # initial count
    line_skip = latidx[0]-1
    start = line_skip*len(lon)+lonidx[0];

    # loop to append indices
    zz = [];
    for i in  range(len(latidx)):
        lineidx = np.arange(start,start+len(lonidx))
        lineZ = dataset.variables['z'][lineidx]
        zz = np.concatenate((zz,lineZ))
        start = start + len(lon)
    # get rid of the non used lat/lon now
    lat = lat[latidx]
    lon = lon[lonidx]

    # reshape topo
    Z = zz[:].reshape( len(lat), len(lon))

    ## import matplotlib.pyplot as plt
    ## v = [-2000, -1000, -500, -100, 0]
    ## plt.contour(lon,lat,Z,v)
    ## plt.colorbar()
    ## plt.gca().invert_yaxis()
    ## plt.show()
        
    return lat,lon,Z

def version14(nc_file, lims):
    '''
    for GEBCO_2014_1D.nc file
    '''
    dataset = netCDF4.Dataset(nc_file)
    lat_min = lims[0]
    lat_max = lims[1]
    lon_min = lims[2]
    lon_max = lims[3]

    x = [-179-59.75/60, 179+59.75/60] # to correct bug in 30'' dataset?
    y = [-89-59.75/60, 89+59.75/60]
    z = dataset.variables['z']
    spacing = dataset.variables['spacing']

    # Compute Lat/Lon
    nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
    ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir
    lon = np.linspace(x[0],x[-1],nx)
    lat = np.linspace(y[0],y[-1],ny)

    # Define bounding box
    BB = dict(
        lon=[lon_min, lon_max],
        lat=[lat_min, lat_max]
        )

    # nonzero returns a tuple of idx per dimension
    # we're unpacking the tuple here so we can lookup max and min
    (latidx,) = np.logical_and(lat >= BB['lat'][0], lat < BB['lat'][1]).nonzero()
    (lonidx,) = np.logical_and(lon >= BB['lon'][0], lon < BB['lon'][1]).nonzero()

    # initial count
    line_skip = latidx[0]-1
    start = line_skip*len(lon)+lonidx[0];

    # loop to append indices
    zz = [];
    for i in  range(len(latidx)):
        lineidx = np.arange(start,start+len(lonidx))
        lineZ = dataset.variables['z'][lineidx]
        zz = np.concatenate((zz,lineZ))
        start = start + len(lon)
    # get rid of the non used lat/lon now
    lat = lat[latidx]
    lon = lon[lonidx]

    # reshape topo
    Z = zz[:].reshape( len(lat), len(lon))

    ## import matplotlib.pyplot as plt
    ## v = [-2000, -1000, -500, -100, 0]
    ## plt.contour(lon,lat,Z,v)
    ## plt.colorbar()
    ## plt.gca().invert_yaxis()
    ## plt.show()
        
    return lat,lon,Z
