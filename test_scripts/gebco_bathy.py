# copied from :
# http://stackoverflow.com/questions/21889079/plot-gebco-data-in-python-basemap

import netCDF4
from mpl_toolkits.basemap import Basemap

# Load data
dataset = netCDF4.Dataset('/home/cyrf0006/Data/GEBCO/GEBCO_08.nc')

# Extract variables
x = dataset.variables['x_range']
y = dataset.variables['y_range']
spacing = dataset.variables['spacing']

# Compute Lat/Lon
nx = (x[-1]-x[0])/spacing[0]   # num pts in x-dir
ny = (y[-1]-y[0])/spacing[1]   # num pts in y-dir

lon = np.linspace(x[0],x[-1],nx)
lat = np.linspace(y[0],y[-1],ny)
lon = np.linspace(x[0],x[-1],1) # by-pass previous 2 lines
lat = np.linspace(y[0],y[-1],1)

# Reshape data
zz = dataset.variables['z']
Z = zz[:].reshape(ny, nx)

# setup basemap.
m = Basemap(llcrnrlon=-30,llcrnrlat=45.0,urcrnrlon=5.0,urcrnrlat=65.0,
            resolution='i',projection='stere',lon_0=-15.0,lat_0=55.0)

x,y = m(*np.meshgrid(lon,lat))

m.contourf(x, y, flipud(Z));
m.fillcontinents(color='grey');
m.drawparallels(np.arange(10,70,10), labels=[1,0,0,0]);
m.drawmeridians(np.arange(-80, 5, 10), labels=[0,0,0,1]);

