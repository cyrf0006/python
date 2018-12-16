'''
/home/cyrf0006/research/tag_lebris
Snipet code to extract shrimp fishing area from shapefile

From https://pypi.python.org/pypi/pyshp:

A record in a shapefile contains the attributes for each shape in the
collection of geometry. Records are stored in the dbf file. The link between
geometry and attributes is the foundation of all geographic information systems.
This critical link is implied by the order of shapes and corresponding records
in the shp geometry file and the dbf attribute file.

The field names of a shapefile are available as soon as you read a shapefile.
You can call the "fields" attribute of the shapefile as a Python list. Each
field is a Python list with the following information:

* Field name: the name describing the data at this column index.
* Field type: the type of data at this column index. Types can be: Character,
Numbers, Longs, Dates, or Memo. The "Memo" type has no meaning within a
GIS and is part of the xbase spec instead.
* Field length: the length of the data found at this column index. Older GIS
software may truncate this length to 8 or 11 characters for "Character"
fields.
* Decimal length: the number of decimal places found in "Number" fields.

To see the fields for the Reader object above (sf) call the "fields"
attribute:
'''

## plt.show()


import netCDF4
from mpl_toolkits.basemap import Basemap
import numpy as  np
import matplotlib.pyplot as plt
import openpyxl, pprint
import shapefile 
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

## ---- Get polygon of interest ---- ##
myshp = open('/home/cyrf0006/research/tag_lebris/SFA4to7_polygons.shp', 'rb')
mydbf = open('/home/cyrf0006/research/tag_lebris/SFA4to7_polygons.dbf', 'rb')
r = shapefile.Reader(shp=myshp, dbf=mydbf)
records = r.records()
shapes = r.shapes()

# Fill dictionary with NAFO divisions
shrimp_area = {}
for idx, rec in enumerate(records):
    if rec[-1] == '':
        continue
    else:
        shrimp_area[rec[-1]] = np.array(shapes[idx].points)
        
polygon4 = Polygon(shrimp_area[4])
polygon5 = Polygon(shrimp_area[5])
polygon6 = Polygon(shrimp_area[6])
polygon7 = Polygon(shrimp_area[7])

## ---- Get hydrographic data ---- ##
ds = xr.open_mfdataset('/home/cyrf0006/data/dev_database/*.nc')

# rough selection
ds = ds.sel(level=ds['level']<1000)
ds = ds.where((ds.longitude>-70) & (ds.longitude<-45), drop=True) # Encompass SFA
ds = ds.where((ds.latitude>45) & (ds.latitude<65), drop=True)
ds = ds.sortby('time')
ds = ds.sel(time=(ds['time.year']>=1990))

# get lat/lon
lons = np.array(ds.longitude)
lats = np.array(ds.latitude)

# To Pandas Dataframe
da = ds['temperature']
df = da.to_pandas()

# Remove empty columns
idx_empty_rows = df.isnull().all(1).nonzero()[0]
df = df.dropna(axis=0,how='all')
lons = np.delete(lons,idx_empty_rows)
lats = np.delete(lats,idx_empty_rows)        

# At this stage I should do some QC on temperature data


## ---- Seasonal and annual means per Zone ---- ##
# Maybe the ideal would be to loop on time to build yearly cube and extract average profile
# For thee moment, I will limit myself to average ever

idx_sfa4 = []
idx_sfa5 = []
idx_sfa6 = []
for i, ll in enumerate(lons):
    point = Point(lons[i], lats[i])
    if  polygon4.contains(point):
        idx_sfa4.append(i)
    elif polygon5.contains(point):
        idx_sfa5.append(i) 
    elif polygon6.contains(point):
        idx_sfa6.append(i) 

df_sfa4 = df.iloc[idx_sfa4] 
df_sfa5 = df.iloc[idx_sfa5] 
df_sfa6 = df.iloc[idx_sfa6] 

# Get number of casts per year per region
df_sfa4_castsno = df_sfa4.resample('As').count().max(axis=1)
df_sfa5_castsno = df_sfa5.resample('As').count().max(axis=1)
df_sfa6_castsno = df_sfa6.resample('As').count().max(axis=1)
df_sfa4_castsno.name='SFA-4'
df_sfa5_castsno.name='SFA-5'
df_sfa6_castsno.name='SFA-6'  
df_castno = pd.concat([df_sfa4_castsno, df_sfa5_castsno, df_sfa6_castsno], axis=1)
df_castno.index=df_castno.index.year

# Seasonal average
df_sfa4 = df_sfa4.resample('Qs').mean()
df_sfa5 = df_sfa5.resample('Qs').mean()
df_sfa6 = df_sfa6.resample('Qs').mean()

# Select ice-free months (this could be changed)
df_sfa4 = df_sfa4[df_sfa4.index.month>=4]
df_sfa5 = df_sfa5[df_sfa5.index.month>=4]
df_sfa6 = df_sfa6[df_sfa6.index.month>=4]

# Annual average of seasonal average (equal weight for each season)
df_sfa4 = df_sfa4.resample('As').mean()
df_sfa5 = df_sfa5.resample('As').mean()
df_sfa6 = df_sfa6.resample('As').mean()


## ---- plot Figure ---- ##

fig = plt.figure()
# ax1
ax1 = plt.subplot2grid((3, 1), (0, 0))
plt.plot(df_sfa4.index, df_sfa4[df_sfa4.columns[df_sfa4.columns<=25]].mean(axis=1))
plt.plot(df_sfa4.index, df_sfa4[df_sfa4.columns[(df_sfa4.columns>=50) & (df_sfa4.columns<=200)]].mean(axis=1))
#plt.plot(df_sfa4.index, df_sfa4[df_sfa4.columns[(df_sfa4.columns>=200) & (df_sfa4.columns<=300)]].mean(axis=1))
ax1.xaxis.label.set_visible(False)
ax1.tick_params(labelbottom='off')
ax1.set_ylabel(r'$\rm T (^{\circ}C)$')
ax1.set_title(r'SFA-4')
ax1.legend(['0-25m', '50-200m'])
ax1.grid()

# ax2
ax2 = plt.subplot2grid((3, 1), (1, 0))
plt.plot(df_sfa5.index, df_sfa5[df_sfa5.columns[df_sfa5.columns<=25]].mean(axis=1))
plt.plot(df_sfa5.index, df_sfa5[df_sfa5.columns[(df_sfa5.columns>=50) & (df_sfa5.columns<=200)]].mean(axis=1))
#plt.plot(df_sfa5.index, df_sfa5[df_sfa5.columns[(df_sfa5.columns>=200) & (df_sfa5.columns<=500)]].mean(axis=1))
ax2.xaxis.label.set_visible(False)
ax2.tick_params(labelbottom='off')
ax2.set_title(r'SFA-5')
ax2.set_ylabel(r'$\rm T (^{\circ}C)$')
ax2.grid()

# ax3
ax3 = plt.subplot2grid((3, 1), (2, 0))
plt.plot(df_sfa6.index, df_sfa6[df_sfa6.columns[df_sfa6.columns<=25]].mean(axis=1))
plt.plot(df_sfa6.index, df_sfa6[df_sfa6.columns[(df_sfa6.columns>=50) & (df_sfa6.columns<=200)]].mean(axis=1))
#plt.plot(df_sfa6.index, df_sfa6[df_sfa6.columns[(df_sfa6.columns>=200) & (df_sfa6.columns<=500)]].mean(axis=1))
ax3.set_title(r'SFA-6')
ax3.set_ylabel(r'$\rm T (^{\circ}C)$')
ax3.grid()

fig.set_size_inches(w=8,h=12)
fig_name = 'SFA_shrimp_temp.png'
fig.set_dpi(300)
fig.savefig(fig_name)



## ----- Second figure on number of Obs. ---- ##
fig = plt.figure()
ax = plt.subplot()
df_castno.plot.bar(ax=ax, rot=0)
plt.ylabel('No. of observations')

fig.set_size_inches(w=18,h=8)
fig_name = 'SFA_noObs.png'
fig.set_dpi(300)
fig.savefig(fig_name)
