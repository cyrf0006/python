'''
Explore Aviso product netcdf file (SSH).
Example derived from http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html

'''

import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import datetime # Python standard library datetime module
from netCDF4 import Dataset,netcdftime,num2date # http://unidata.github.io/netcdf4-python/

# ANomaly box size
box_size = 1 #degree
model_resolution = 1/12.0
box_size_pixel = np.int(box_size/model_resolution)
## map stuff

# Zoom SPM shelf
## lat_min = 45
## lat_max = 48
## lon_min = -59
## lon_max = -52

# Close zoom SPM
lat_min = 46.5
lat_max = 47.5
lon_min = -57
lon_max = -55.5

# East NL shelf
## lat_min = 46
## lat_max = 49
## lon_min = -54
## lon_max = -51.5

# Zoom St.27
## lat_min = 46.5
## lat_max = 48.5
## lon_min = -53
## lon_max = -51.5

decim_scale = 1 # for bathymetry

## ---- Bathymetry ---- ####
# Load data
dataFile = '/home/cyrf0006/Data/GEBCO/GRIDONE_1D.nc'
dataset = Dataset(dataFile)

# Extract variables
x = dataset.variables['x_range']
y = dataset.variables['y_range']
spacing = dataset.variables['spacing']

# Compute Lat/Lon
nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir

lon_bathym = np.linspace(x[0],x[-1],nx)
lat_bathym = np.linspace(y[0],y[-1],ny)

# Reshape data
zz = dataset.variables['z']
Z = zz[:].reshape(ny, nx)
Z = np.flipud(Z)


# Reduce data according to Region params
idx_lon = np.where(np.logical_and(lon_bathym>=lon_min, lon_bathym<=lon_max))
idx_lat = np.where(np.logical_and(lat_bathym>=lat_min, lat_bathym<=lat_max))
lon_bathym = lon_bathym[idx_lon]
lat_bathym = lat_bathym[idx_lat]
Z = Z[np.ix_(np.squeeze(idx_lat), np.squeeze(idx_lon))]
lon_bathym = lon_bathym[::decim_scale]
lat_bathym = lat_bathym[::decim_scale]
Z = Z[::decim_scale, ::decim_scale]


## ------- Aviso ------- ##
# load file
nc_f = '/home/cyrf0006/Data/Copernicus/hourly/global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh_1511459141562.nc'
nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file

# Extract data from NetCDF file
lats = nc_fid.variables['latitude'][:]  # extract/copy the data
lons = nc_fid.variables['longitude'][:]
time = nc_fid.variables['time'][:]
zos_tmp = nc_fid.variables['zos'][:]  # shape is time, lat, lon as shown above
zos = np.reshape(zos_tmp, (len(time), len(lats), len(lons)))

# Reduce data according to Region params
idx_lon = np.where(np.logical_and(lons>=lon_min-2, lons<=lon_max+2))
idx_lat = np.where(np.logical_and(lats>=lat_min-2, lats<=lat_max+2))
lons = lons[idx_lon]
lats = lats[idx_lat]

# Caldendar time
time_unit = nc_fid.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"
try :
    t_cal = nc_fid.variables['time'].calendar
except AttributeError : # Attribute doesn't exist
    t_cal = u"gregorian" # or standard

tvalue = num2date(time,units = time_unit,calendar = t_cal)
str_time = [i.strftime("%Y-%m-%d %H:%M") for i in tvalue] # to display dates as string

# plot ZOS in timeframe
for time_idx in np.arange(0,len(time)):
    print str_time[time_idx]

    # raw SSH
    ssh = np.squeeze(zos[time_idx,:,:])
    ssh = ssh[np.ix_(np.squeeze(idx_lat), np.squeeze(idx_lon))]

    ssh_anom = np.empty(np.shape(ssh))
    ssh_anom[:] = np.NAN
    
    # Compute anomaly
    for i in np.arange(box_size_pixel/2,len(lats)-box_size_pixel/2):
        for j in np.arange(box_size_pixel/2,len(lons)-box_size_pixel/2):
            ssh_anom[i,j] = ssh[i,j] -  np.nanmean(ssh[i-box_size_pixel/2:i+box_size_pixel/2,j-box_size_pixel/2:j+box_size_pixel/2])

    
    fig = plt.figure(1)
    fig.clf()
    #fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,\
                llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='h', lon_0=np.mean((lon_min, lon_max)))
    m.drawcoastlines()
    m.drawmapboundary()
    m.fillcontinents(color='gray');
    m.drawparallels(np.arange(lat_min,lat_max,2), labels=[1,0,0,0], fontsize=12, fontweight='normal');
    m.drawmeridians(np.arange(lon_min, lon_max, 2), labels=[0,0,0,1], fontsize=12, fontweight='normal');
    #m.drawparallels(np.arange(lat_min,lat_max,1), labels=[1,0,0,0], fontsize=12, fontweight='normal');
    #m.drawmeridians(np.arange(lon_min, lon_max, 1), labels=[0,0,0,1], fontsize=12, fontweight='normal');

    # Create 2D lat/lon arrays for Basemap
    lon2d, lat2d = np.meshgrid(lons, lats)
    # Transforms lat/lon into plotting coordinates for projection
    x, y = m(lon2d, lat2d)
    
    # plot contours
    #V = np.arange(-1.6, 1.6, .2)
    V = np.arange(-2.4, 2.4, .2)
    v = np.linspace(100, 500, 5)
    cs = m.contourf(x, y, ssh_anom*100, V, cmap=plt.cm.Spectral_r, extend='both')
    lon2d, lat2d = np.meshgrid(lon_bathym, lat_bathym)
    x,y = m(lon2d, lat2d)
    c = m.contour(x, y, -Z, v, colors='k');
    #cbar = plt.colorbar(cs, orientation='horizontal')
    cbar = plt.colorbar(cs, orientation='vertical')

    # Plot stations St 27
    #x,y = m(-52.587, 47.547) #St 27
    x,y = m(-56.3, 47.1) # Rade Miquelon
    m.scatter(x,y,15,marker='D',color='cyan')
    
    cbar.set_label(r"$\eta'$ (cm)")
    plt.title(str_time[time_idx])

    
    # save fig.
    outname = 'frame_'+'%04d' % time_idx+'.png'
    fig.set_size_inches(w=6, h=8)
    fig.set_dpi(150)
    fig.savefig(outname)
