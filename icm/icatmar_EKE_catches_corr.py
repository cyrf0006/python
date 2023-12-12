'''
Calculate EKE and correlate with Seine data

Frederic.Cyr@dfo-mpo.gc.ca
November 2023

'''

# Load packages and data:
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import EqualEarth, PlateCarree
import numpy as np
from xeofs.models import EOF
import pandas as pd
import seaborn as sn
import cmocean as cmo
from matplotlib.colors import from_levels_and_colors
import os

## Some options:
annual = True
detrend = True
deseason = False
tmode = False #<----- Doesn't seem to work.
REGION = 'NNW' # 'west', 'NW' or 'med' or 'NNW'

#var = 'zos' # ssh
#var = 'so' # surf salinity
#var = 'so_075' # 75m salinity
#var = 'so_200' # 400m salinity
#var = 'so_400' # 200m salinity
#var = 'thetao' # surf. temp
#var = 'thetao_075' # 75m temp
#var = 'thetao_200' # 400m temp
#var = 'thetao_400' # 200m temp
#var = 'bottomT' # Bottom temperature
#var = 'KE' # Eddy Kinetic Energy
#var = 'EKE' # Eddy Kinetic Energy
#var = 'MLD' # Mixed Layer Depth
#var = 'wMLD' # Winter Mixed Layer Depth
var = 'chlat' # Surface Chlorophyll-a
#var = 'chla' # Surface Chlorophyll-a
#var = 'chla75' # Chlorophyll-a @75m
#var = 'phyc' # phytoplankton in carbon


# For detrending (https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f)
def detrend(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

##  ---- Load and prepare Re-analysis the data ---- ##
var_text = var
if var == 'zos':
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-ssh-rean-m_1695807984977.nc')
elif var == 'thetao':
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-tem-rean-m_1695808094028.nc')
elif var == 'thetao_075':
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-tem_075m-rean-m_1698656156898.nc')
    var_text = var
    var = 'thetao'
elif var == 'thetao_200':
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-tem_200m-rean-m_1697450717425.nc')
    var_text = var
    var = 'thetao'
elif var == 'thetao_400':
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-tem_400m-rean-m_1697027180336.nc')
    var_text = var
    var = 'thetao'
elif var == 'so':   
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-sal-rean-m_1695807438765.nc')
elif var == 'so_075':   
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-sal_075m-rean-m_1698655665693.nc')
    var_text = var
    var = 'so'
elif var == 'so_200':   
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-sal_200m-rean-m_1695807438765.nc')
    var_text = var
    var = 'so'
elif var == 'so_400':   
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-sal_400m-rean-m_1695807438765.nc')
    var_text = var
    var = 'so'
elif var == 'bottomT':    
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-tem-rean-m_1695808053925.nc')
elif var == 'EKE':    
    ds_u = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-cur-rean-m_1698666573702.nc')
    ds_v = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-cur-rean-m_1698667081200.nc')
    u = ds_u['uo']
    v = ds_v['vo']
    u_bar = u.mean('time')
    v_bar = v.mean('time')
    u_prime = u - u_bar
    v_prime = u - u_bar
    eke = u_prime**2 + v_prime**2
    ds = eke.to_dataset(name = 'EKE')
elif var == 'KE':    
    ds_u = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-cur-rean-m_1698666573702.nc')
    ds_v = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-cur-rean-m_1698667081200.nc')
    u = ds_u['uo']
    v = ds_v['vo']
    eke = u**2 + v**2
    ds = eke.to_dataset(name = 'KE')        
elif var == 'MLD':    
    var_text = var
    var = 'mlotst'
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-mld-rean-m_1698652196437.nc')
elif var == 'wMLD':    
    var_text = var
    var = 'mlotst'
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-mld-rean-m_1698652196437.nc')
    # Select only Jan-Feb.
    ds = ds.where(ds['time.month'] <= 2, drop=True)
elif var == 'chla':    
    var_text = var
    var = 'chl'
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-ogs-pft-rean-m_1698392095150.nc')
    # Select only May-Oct.
    #ds = ds.where(ds['time.month'] >= 5, drop=True)
    #ds = ds.where(ds['time.month'] <= 10, drop=True)
    ds = ds.rename({'longitude': 'lon','latitude': 'lat'})
elif var == 'chlat':    
    var_text = var
    var = 'chl'
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-ogs-pft-rean-m_1701763169126.nc')
    # Select only May-Oct.
    ds = ds.where(ds['time.month'] >= 5, drop=True)
    ds = ds.where(ds['time.month'] <= 10, drop=True)
    ds = ds.rename({'longitude': 'lon','latitude': 'lat'})
    # vertical average
    ds = ds.where(ds['depth'] >= 10, drop=True)
    ds = ds.where(ds['depth'] <= 150, drop=True)
    #ds = ds.mean('depth')
elif var == 'chla75':    
    var_text = var
    var = 'chl'
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-ogs-pft_075m-rean-m_1698392827618.nc')
    ds = ds.rename({'longitude': 'lon','latitude': 'lat'})
elif var == 'phyc':    
    var_text = var
    var = 'phyc'
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-ogs-pft-rean-m_1698752084987.nc')
    ds = ds.rename({'longitude': 'lon','latitude': 'lat'})
    ds = ds.mean('depth')
    
# Keep only <2020 (2021 not complete)
ds = ds.where(ds['time.year'] <= 2020, drop=True)
# Slect sub-region:
if REGION == 'west':
    print('West Med')
    ds = ds.where(ds['lon'] <= 22, drop=True) #_west
elif REGION == 'NW':
    print('NW Med')
    ds = ds.where(ds['lon'] <= 9, drop=True) #_NW
elif REGION == 'NNW':
    print('NNW Med')
    ds = ds.where(ds['lon'] <= 9, drop=True) #_NNW
    ds = ds.where(ds['lat'] >= 38, drop=True) #_NNW
    #ds = ds.where(ds['lat'] >= 39, drop=True) #_NNW
else:
    print('Use all Med sea')

# get the raw (monthly) data
data_raw = ds[var]
#data_raw = ds[['lat', 'lon', 'time']]

if detrend:
    print('De-trend the data')
    data_raw = detrend(data_raw, 'time', deg=1)
    
# Check if annual mean, de-season or raw
if annual:
    print('Annual EOFs')
    annual_mean = data_raw.groupby('time.year').mean('time')
    my_data = annual_mean.rename({'year':'time'})
elif deseason:
    print('Monthly EOFs, climatological cycle removed')
    monthly_clim = data_raw.groupby('time.month').mean('time')
    my_data = data_raw.groupby('time.month') - monthly_clim
else: #raw data
    print('EOFs on raw data')
    my_data = data_raw.copy()

# average data into Series
if var_text=='wMLD':
    data_series = my_data.mean(['lon', 'lat']).to_pandas()
else:
    data_series = my_data.mean(['depth', 'lon', 'lat']).to_pandas()
data_anom = (data_series - data_series.mean()) / data_series.std()
#data_anom = data_anom.resample('M').mean() #to have same month as cpue

##  ---- Load and prepare catches data ---- ##
# Series 1 (zos, temp 0-400, sal 0-400) or Series 2 (temp 75, MLD, EKE, chla)
series1 = False # Series 2 does not work well here; see *_regional.py

# Load data
df_landings = pd.read_csv('20231017_Landings_PS_00-22.csv')

df_cpue = df_landings.groupby(['Date', 'Subgroup']).mean()['Average weight (Kg / day * vessel)']
df_cpue = df_cpue.unstack()
df_cpue.index = pd.to_datetime(df_cpue.index)
if annual:
    df_cpue = df_cpue.resample('As').mean()
else:
    df_cpue = df_cpue.resample('M').mean()
    data_anom = data_anom.resample('M').mean()
        
# Seclect most important species
cpue30 = df_cpue.sum()[df_cpue.sum()>=df_cpue.sum().sort_values().iloc[-5]].index
df_cpue = df_cpue[cpue30.values]
df_cpue = df_cpue.interpolate(axis=1)

# index to year
if annual:
    df_cpue.index = df_cpue.index.year

## Anomalies
df_anom = (df_cpue - df_cpue.mean()) / df_cpue.std()


## ---- Concat data ---- ##
df = pd.concat([data_anom, df_anom], axis=1)
if deseason:
    df = df[df.index.year>=2000]
    df = df[df.index.year<=2020]
    df = df.rolling(12).mean()
else:
    df = df[df.index>=2000]
    df = df[df.index<=2020]  


## ---- Plot correlation ---- ##
from scipy.stats import pearsonr
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

# Build the colormap
vmin = -1.0
vmax = 1.0
midpoint = 0.0
levels = np.linspace(vmin, vmax, 21)
midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
colvals = np.interp(midp, [vmin, midpoint, vmax], [-1, 0., 1])
normal = plt.Normalize(-1.0, 1.0)
reds = plt.cm.Reds(np.linspace(0,1, num=6))
blues = plt.cm.Blues_r(np.linspace(0,1, num=6))
whites = [(.95,.95,.95,.95)]*9
colors = np.vstack((blues[0:-1,:], whites, reds[1:,:]))
colors = np.concatenate([[colors[0,:]], colors, [colors[-1,:]]], 0)
cmap, norm = from_levels_and_colors(midp, colors, extend='both')

## For Values
# correlation matric and pvalues
df = df.dropna(thresh=16, axis=1)
corrMatrix = df.corr().round(2)
pvalues = calculate_pvalues(df)
## Restrict correlation matrix
#corrMatrix = corrMatrix.iloc[0:35,35:,]
#pvalues = pvalues.iloc[0:35,35:,]
# Text
annot_text  = corrMatrix.astype('str')
corrMatrix_text = corrMatrix.copy()

for i in np.arange(pvalues.shape[0]):
    for j in np.arange(pvalues.shape[1]):
        if pvalues.iloc[i,j]>=.05:
            #annot_text.iloc[i,j] = annot_text.iloc[i,j]+'*'
            corrMatrix.iloc[i,j] = 0            
            corrMatrix_text.iloc[i,j] = ' '
            
plt.close('all')
fig = plt.figure(3)
#mask = np.zeros_like(corrMatrix)
#mask[np.triu_indices_from(mask)] = False
#np.fill_diagonal(mask, 0)
sn.heatmap(corrMatrix, annot=corrMatrix_text.astype('str'), fmt='s', linewidths=.2, cmap=cmap, cbar=None, vmin=-1.05, vmax=1.05)
plt.title('Pearson correlation coefficients')
# tweak yticklabels
LABELS = corrMatrix.keys().to_list()
ax = plt.gca()
XTICKS = np.arange(0.5, corrMatrix.shape[1]+.5, 1)
plt.xticks(XTICKS)
ax.set_xticklabels(LABELS)
fig.set_size_inches(w=20,h=14)
if series1:
    fig_name = 'Correlation_PC-CPUE_PS.png'
else:
    fig_name = 'Correlation_EKE-CPUE_PS.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



plt.close('all')
df['Sardina pilchardus'].plot()
df['Engraulis encrasicolus'].plot()
df[0].plot()
plt.legend(['Sardina', 'Anchoa', 'VAR'])
plt.show()
