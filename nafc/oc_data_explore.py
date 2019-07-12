'''
Script written to process explore OceanChoice data that were submitted by Jeff Burke.

May 2019

Frederic.Cyr@dfo-mpo.gc.ca

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
import pandas as pd
import os
import netCDF4
from mpl_toolkits.basemap import Basemap


# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)

## ---- prepare data ---- ##
# Cannot download directly becasaue I receive the content of the html file rather than the data.
#data_file = '/home/cyrf0006/research/oceanChoice/KH - Catch by Fishing Zone Trip 5 (2019).xlsx'
data_file = '/home/cyrf0006/research/oceanChoice/Lynx - T6 - Haul Overview.xlsx'
df = pd.read_excel(data_file)


df = df.set_index('Haul b')
df.index = pd.to_datetime(df.index)
df.index.name = 'Haul_begin'
# Drop sum column:
df = df.loc[pd.notnull(df.index)]

lat = df[[u'Lat (Begin)', u'Lat (End)']].mean(axis=1)
lon = df[[u'Lon (Begin)', u'Lon (End)']].mean(axis=1)
temp = df[[u'Sea temp min', u'Sea temp max']].mean(axis=1)
catch = df['Catch']
towtime = df['Towtime']
rel_catch = catch/towtime
catch_norm = rel_catch/rel_catch.max()*100



## ---- Bathymetry ---- ####
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
decim_scale = 4
v = np.linspace(0, 4000, 9)

# Load data
dataset = netCDF4.Dataset(dataFile)

# Extract variables
x = dataset.variables['x_range']
y = dataset.variables['y_range']
spacing = dataset.variables['spacing']

# Compute Lat/Lon
nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir

lonz = np.linspace(x[0],x[-1],nx)
latz = np.linspace(y[0],y[-1],ny)

# Reshape data
zz = dataset.variables['z']
Z = zz[:].reshape(ny, nx)

# Reduce data according to Region params
lonz = lonz[::decim_scale]
latz = latz[::decim_scale]
Z = Z[::decim_scale, ::decim_scale]


## ---- Plot stations position on map ---- ##
lon_0 = -50
lat_0 = 50

lonLims = [-62, -50]
latLims = [50, 60]
proj = 'merc'

fig = plt.figure()

m = Basemap(projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='l')

# Bathymetry
x,y = m(*np.meshgrid(lonz,latz))
v = np.linspace(-4000, 0, 9)
c = m.contourf(x, y, np.flipud(Z), v, cmap=plt.cm.PuBu_r, extend="both");
m.fillcontinents(color='grey');

# Add casts identification
x, y = m(lon.values, lat.values)
pts1 = m.scatter(x,y, c=temp, s=catch_norm, marker='o', alpha=.5)
cb = plt.colorbar(pts1, orientation='vertical')
cb.set_label(r'Temperature ($\rm ^{\circ}C$)', fontSize=15, fontWeight='bold')

# Add Grid Lines
m.drawparallels(np.arange(latLims[0], latLims[1], 2.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(lonLims[0], lonLims[1], 2.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Title
plt.title('test with Lynx data', fontweight='bold')


fig.set_size_inches(w=9,h=9)
fig_name = 'test_lynx_data.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)



