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
var = 'EKE' # Eddy Kinetic Energy
#var = 'MLD' # Mixed Layer Depth
#var = 'wMLD' # Winter Mixed Layer Depth
#var = 'chlat' # Total chl-a 0-200m
#var = 'chla' # Surface Chlorophyll-a
#var = 'chla75' # Chlorophyll-a @75m
#var = 'phyc' # phytoplankton in carbon


# For detrending (https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f)
def detrend(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

##  ---- Load and prepare the data ---- ##
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
    ds = ds.mean('depth')
elif var == 'chla':    
    var_text = var
    var = 'chl'
    ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-ogs-pft-rean-m_1698392095150.nc')
    # Select only May-Oct.
    ds = ds.where(ds['time.month'] >= 5, drop=True)
    ds = ds.where(ds['time.month'] <= 10, drop=True)
    ds = ds.rename({'longitude': 'lon','latitude': 'lat'})
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
    # vertical average
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

## ---- Calculate OEFs ---- ##
model = EOF(n_modes=5, standardize=False)
if tmode:
    # T-Mode EOFs
    print('Maximize Time Variance')
    model.fit(my_data, dim=('lat', 'lon'))
else:
    # S-Mode EOFs
    print('Maximize Spatial Variance')
    model.fit(my_data, dim=('time'))

    #model = EOF(n_modes=20, standardize=True, use_coslat=True)
    #model.fit(t2m, dim='time')

expvar = model.explained_variance_ratio()
components = model.components()
scores = model.scores()
expvar * 100

## ----  create figure ---- ##
if tmode:
    proj = EqualEarth(central_longitude=10)
    kwargs = {'cmap' : 'RdBu_r', 'transform': PlateCarree()}

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, width_ratios=[2, 1])
    ax0 = [fig.add_subplot(gs[i, 0], projection=proj) for i in range(3)]
    ax1 = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    for i, (a0, a1) in enumerate(zip(ax0, ax1)):
        scores.sel(mode=i+1).plot(ax=a0, **kwargs)
        a0.coastlines(color='.5')
        components.sel(mode=i+1).plot(ax=a1)

        a0.set_xlabel('')
        TITLE = a0.get_title() + ' (' + str(np.round(expvar[i].values*100, 2)) + '% variance explained)'
        a0.set_title(TITLE)

else: # S-mode
    proj = EqualEarth(central_longitude=10)
    kwargs = {
        'cmap' : 'RdBu_r', 'vmin' : -.01, 'vmax': .01, 'transform': PlateCarree()
    }

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, width_ratios=[1, 2])
    ax0 = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax1 = [fig.add_subplot(gs[i, 1], projection=proj) for i in range(3)]

    for i, (a0, a1) in enumerate(zip(ax0, ax1)):
        scores.sel(mode=i+1).plot(ax=a0)
        a1.coastlines(color='.5')
        components.sel(mode=i+1).plot(ax=a1, **kwargs)

        a0.set_xlabel('')
        TITLE = a1.get_title() + ' (' + str(np.round(expvar[i].values*100, 2)) + '% variance explained)'
        a0.set_title(TITLE)

        
# Save
plt.tight_layout()

if tmode:
    if annual:
        plt.savefig('eof-tmode_' + var_text + '_annual.jpg')
    elif deseason:
        plt.savefig('eof-tmode_' + var_text + '_deseason.jpg')
    else: 
        plt.savefig('eof-tmode_' + var_text + '_raw.jpg')
else:   
    if annual:
        plt.savefig('eof-smode_' + var_text + '_annual_' + REGION + '.jpg')
        # Preserve the scores
        scores.to_pandas().to_pickle('scores_' + var_text + '_annual_' + REGION + '.pkl')
    elif deseason:
        plt.savefig('eof-smode_' + var_text + '_deseason' + REGION + '.jpg')
        # Preserve the scores
        scores.to_pandas().to_pickle('scores_' + var_text + '_deseason' + REGION + '.pkl')
    else: 
        plt.savefig('eof-smode_' + var_text + '_raw' + REGION + '.jpg')
        # Preserve the scores
        scores.to_pandas().to_pickle('scores_' + var_text + '_raw' + REGION + '.pkl')

    
