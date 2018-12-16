# A first test to read Excel nutrient file and export to Pandas.

# Check in:
#  /home/cyrf0006/research/AZMP_database/biochem

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from seawater import extras as swx
import netCDF4
from mpl_toolkits.basemap import Basemap
from scipy import stats
import matplotlib.gridspec as gridspec

# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)

## ---- Region parameters ---- ##
dataFile = '/home/cyrf0006/data/GEBCO/GRIDONE_1D.nc'
lon_0 = -50
lat_0 = 50
lonLims = [-60, -44] # FishHab region
latLims = [39, 56]
lonLims = [-65, -40] # 
latLims = [39, 60]
proj = 'merc'
zmax = 1000 # do try to compute bottom temp below that depth
dz = 5 # vertical bins
dc = .5
lon_reg = np.arange(lonLims[0]+dc/2, lonLims[1]-dc/2, dc)
lat_reg = np.arange(latLims[0]+dc/2, latLims[1]-dc/2, dc)
lon_grid, lat_grid = np.meshgrid(lon_reg,lat_reg)
season = 'all'
variable = 'f_pw'
depth_min = 50
depth_max = 300

## ----  Load biochemical data ---- ##
df = pd.read_excel('/home/cyrf0006/github/AZMP-NL/data/AZMP_Nutrients_1999_2016_Good_Flags_Only.xlsx')
# Set date as index
df = df.set_index('sas_date')
# Drop other time-related columns
df = df.drop(['Day', 'Month', 'Year'], axis=1)
# Compute Saturation O2 and add to dataframe
df['satO2'] = swx.satO2(df['salinity'], df['temp'])
df['satO2_perc'] = df['oxygen']/df['satO2']*100
df['AOU'] = df['satO2'] - df['oxygen']
df['ratio_NO3-PO4'] = df['NO3']/df['PO4']
df[df['ratio_NO3-PO4']==np.inf]=np.nan # replace inf by nan
df['f_pw'] = (df.NO3 - (17.499*df.PO4 - 3.072)) / ((12.368*df.PO4 - 10.549) - (17.499*df.PO4 - 3.072)) #Pacific Water fraction



## ---- Loop on stations ---- ##
lat_list = []
lon_list = []
slope_list = []
for stn in df.sname.unique():
    df_tmp = df[df.sname==stn]
    df_tmp = df_tmp[df_tmp.depth>=depth_min]
    df_tmp = df_tmp[df_tmp.depth<=depth_max]
    if season == 'all':
        df_tmp = df_tmp.resample('A').mean() # annual
    else:
        df_tmp = df_tmp.resample('Q').mean() # Quarter-annual
        if season == 'spring':
            df_tmp = df_tmp[df_tmp.index.month==6]
        elif season == 'summer':
            df_tmp = df_tmp[df_tmp.index.month==9]
        elif season == 'fall':
            df_tmp = df_tmp[df_tmp.index.month==12]

    # Re-average in 5-year blocks (TEST!!!)
    #df_tmp = df_tmp.resample('5Y').mean()
        
    x = df_tmp.index.year
    y = df_tmp[variable].values # <-------- here we specify the variable
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    
    print stn, y.size 
    if y.size > 10: # At least 3 5-year period years in timeseries
        #if y.size<8:
        print x           
        lat_list.append(df_tmp.Latitude.mean())
        lon_list.append(df_tmp.Longitude.mean())
        #A = np.vstack([df_tmp.index.year, np.ones(len(df_tmp.index.year,))]).T
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        #m, c = np.linalg.lstsq(A, df_tmp.satO2.values, rcond=None)[0]
        slope_list.append(m)

slope = np.array(slope_list)
lat_slope = np.array(lat_list)
lon_slope = np.array(lon_list)
lat_slope = lat_slope[~np.isnan(slope)]
lon_slope = lon_slope[~np.isnan(slope)]
slope = slope[~np.isnan(slope)]


    
## ---- Bathymetry ---- ####
print('Load and grid bathymetry')
# Load data
dataset = netCDF4.Dataset(dataFile)
# Extract variables
x = dataset.variables['x_range']
y = dataset.variables['y_range']
spacing = dataset.variables['spacing']
# Compute Lat/Lon
nx = int((x[-1]-x[0])/spacing[0]) + 1  # num pts in x-dir
ny = int((y[-1]-y[0])/spacing[1]) + 1  # num pts in y-dir
lon = np.linspace(x[0],x[-1],nx)
lat = np.linspace(y[0],y[-1],ny)
# interpolate data on regular grid (temperature grid)
# Reshape data
zz = dataset.variables['z']
Z = zz[:].reshape(ny, nx)
Z = np.flipud(Z) # <------------ important!!!
# Reduce data according to Region params
idx_lon = np.where((lon>=lonLims[0]) & (lon<=lonLims[1]))
idx_lat = np.where((lat>=latLims[0]) & (lat<=latLims[1]))
Z = Z[idx_lat[0][0]:idx_lat[0][-1]+1, idx_lon[0][0]:idx_lon[0][-1]+1]
lon = lon[idx_lon[0]]
lat = lat[idx_lat[0]]

print(' -> Done!')


## ------  plot with inset ------- ##
df_section = df.copy()
df_section = df_section[df_section.depth>=depth_min]
df_section = df_section[df_section.depth<=depth_max]
df_BB = df_section[df_section.section=='BB']
df_SI = df_section[df_section.section=='SI']
df_SEGB = df_section[df_section.section=='SEGB']
df_SESPB = df_section[df_section.section=='SESPB']
if season == 'all':
    df_BB = df_BB.resample('A').mean() # annual
    df_SI = df_SI.resample('A').mean()
    df_SEGB = df_SEGB.resample('A').mean()
    df_SESPB = df_SESPB.resample('A').mean()
else:
    df_BB = df_BB.resample('Q').mean() # annual
    df_SI = df_SI.resample('Q').mean()
    df_SEGB = df_SEGB.resample('Q').mean()
    df_SESPB = df_SESPB.resample('Q').mean()
    if season == 'spring':
        df_BB = df_BB[df_BB.index.month==6]
        df_SI = df_SI[df_SI.index.month==6]
        df_SEGB = df_SEGB[df_SEGB.index.month==6]
        df_SESPB = df_SESPB[df_SESPB.index.month==6]
    elif season == 'summer':
        df_BB = df_BB[df_BB.index.month==9]
        df_SI = df_SI[df_SI.index.month==9]
        df_SEGB = df_SEGB[df_SEGB.index.month==9]
        df_SESPB = df_SESPB[df_SESPB.index.month==9]
    elif season == 'fall':
        df_BB = df_BB[df_BB.index.month==12]
        df_SI = df_SI[df_SI.index.month==12]
        df_SEGB = df_SEGB[df_SEGB.index.month==12]
        df_SESPB = df_SESPB[df_SESPB.index.month==12]
    

## S1
plt.clf()
fig = plt.figure()
gs1 = gridspec.GridSpec(3, 2)
gs1.update(left=0.04, right=0.98, wspace=0.4)

ax1 = plt.subplot(gs1[:, :-1])
m = Basemap(ax=ax1, projection='merc',lon_0=lon_0,lat_0=lat_0, llcrnrlon=lonLims[0],llcrnrlat=latLims[0],urcrnrlon=lonLims[1],urcrnrlat=latLims[1], resolution='i')
levels = np.linspace(0, 15, 16)
x,y = m(*np.meshgrid(lon,lat))
cc = m.contour(x, y, -Z, [100, 500, 1000, 4000], colors='grey');
m.fillcontinents(color='tan');
#plt.title(variable + ' trends ' + np.str(depth_min) + '-' + np.str(depth_max) + 'm' + ' (annual)')

m.drawparallels([40, 45, 50, 55, 60], labels=[1,0,0,0], fontsize=12, fontweight='normal');
m.drawmeridians([-60, -55, -50, -45], labels=[0,0,0,1], fontsize=12, fontweight='normal');

# plot slope data
x, y = m(lon_slope, lat_slope)
limit = np.mean([slope.max(), np.abs(slope.min())])
#limit = .5
s = m.scatter(x,y, c=slope, vmin=-limit, vmax=+limit, cmap='RdBu_r')
# Write station names
x,y = m(np.array(df[df.sname=='BB15'].Longitude.mean()), np.array(df[df.sname=='BB15'].Latitude.mean()))
plt.text(x, y, ' BB ', horizontalalignment='left', verticalalignment='center', fontsize=10, color='k', fontweight='bold')
x,y = m(np.array(df[df.sname=='SI14'].Longitude.mean()), np.array(df[df.sname=='SI14'].Latitude.mean()))
plt.text(x, y, ' SI ', horizontalalignment='left', verticalalignment='center', fontsize=10, color='k', fontweight='bold')
x,y = m(np.array(df[df.sname=='SEGB19'].Longitude.mean()), np.array(df[df.sname=='SEGB19'].Latitude.mean()))
plt.text(x, y, ' SEGB ', horizontalalignment='left', verticalalignment='center', fontsize=10, color='k', fontweight='bold')
# Colorbar
cax = plt.axes([0.42,0.15,0.015,0.7])
cb = plt.colorbar(s, cax=cax)
#cb.set_label(r'$\rm C_{units} / year$', fontsize=12, fontweight='normal')



## S2
ax2 = plt.subplot(gs1[0, -1])
df_SI[variable].plot(ax=ax2)
ax2.xaxis.label.set_visible(False)
ax2.tick_params(labelbottom='off')
ax2.text(pd.to_datetime('2015'), df_SI[variable].max() - .12*(df_SI[variable].max()-df_SI[variable].min()), 'SI')
# fit
x = df_SI.index.year
y = df_SI[variable].values
if y[~np.isnan(y)].size!=0: # e.g., no SI in spring...    
    m, b, r_value, p_value, std_err = stats.linregress(x[(~np.isnan(y)) & (~np.isinf(y))], y[(~np.isnan(y)) & (~np.isinf(y))])
    fit = pd.Series(x*m+b, index=df_SI.index)
    fit.plot(ax=ax2)
ax2.text(pd.to_datetime('1999'), df_SI[variable].min() + .1*(df_SI[variable].max()-df_SI[variable].min()), 'm = ' + np.str(np.round(m, 4)), fontsize=9)
ax2.grid()
   
## S3
ax3 = plt.subplot(gs1[1, -1])
df_BB[variable].plot(ax=ax3)
ax3.xaxis.label.set_visible(False)
ax3.tick_params(labelbottom='off')
ax3.grid()
ax3.text(pd.to_datetime('2015'), df_BB[variable].max() - .12*(df_BB[variable].max()-df_BB[variable].min()), 'BB')
# fit
x = df_BB.index.year
y = df_BB[variable].values
if y[~np.isnan(y)].size!=0:    
    m, b, r_value, p_value, std_err = stats.linregress(x[(~np.isnan(y)) & (~np.isinf(y))], y[(~np.isnan(y)) & (~np.isinf(y))])
    fit = pd.Series(x*m+b, index=df_BB.index)
    fit.plot(ax=ax3)
ax3.text(pd.to_datetime('1999'), df_BB[variable].min() + .1*(df_BB[variable].max()-df_BB[variable].min()), 'm = ' + np.str(np.round(m, 4)), fontsize=9)
ax3.grid()

## S4
ax4 = plt.subplot(gs1[2, -1])
df_SEGB[variable].plot(ax=ax4)
ax4.xaxis.label.set_visible(False)
ax4.grid()
ax4.text(pd.to_datetime('2013'), df_SEGB[variable].max() - .12*(df_SEGB[variable].max()-df_SEGB[variable].min()), 'SEGB')
# fit
x = df_SEGB.index.year
y = df_SEGB[variable].values
if y[~np.isnan(y)].size!=0:    
    m, b, r_value, p_value, std_err = stats.linregress(x[(~np.isnan(y)) & (~np.isinf(y))], y[(~np.isnan(y)) & (~np.isinf(y))])
    fit = pd.Series(x*m+b, index=df_SEGB.index)
    fit.plot(ax=ax4)
ax4.text(pd.to_datetime('1999'), df_SEGB[variable].min() + .1*(df_SEGB[variable].max()-df_SEGB[variable].min()), 'm = ' + np.str(np.round(m, 4)), fontsize=9)
ax4.grid()

#### ---- Save Figure ---- ####
fig.suptitle(variable + ' trends ' + season + ' ([units]/yr)', fontsize=12)
fig.set_size_inches(w=10, h=6)
fig.set_dpi(200)
fig_name = 'map_' + variable + '_trends_' + season + '_subplots' + np.str(depth_min) + '-' + np.str(depth_max) + 'm.png'
fig.savefig(fig_name)


