import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime

#ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/PP/cmems_mod_glo_bgc_my_0.25_P1M-m_1695129763017.nc')
ds = xr.open_dataset('/home/cyrf0006/data/CMEMS/Med_reanalysis/med-cmcc-sal-rean-m_1695807438765.nc')

# 1. Nw Med region:
ds_NW = ds.where((ds.longitude>=0) & (ds.longitude<=9), drop=True) # original one
ds_NW = ds_NW.where((ds_NW.latitude>=38) & (ds_NW.latitude<=45), drop=True)
# Select depth range
ds_NW = ds_NW.where((ds_NW.depth<=100), drop=True) # original one
# mean depth and space
ds_NW = ds_NW.mean(dim=['depth', 'latitude', 'longitude'])
# To pandas
#phy = ds_NW['phyc'].to_dataframe()
pp = ds_NW['nppv'].to_dataframe()
# Save data
pp_annual_NW = pp.resample('As').mean()
pp_annual_NW.index = pp_annual_NW.index.year
pp_annual_NW.rename(columns={'nppv':'NW Med'}, inplace=True)
pp_annual_NW.to_csv('cmems_PP_NWMed.csv')
del pp

# 2. Entire Med Region:
ds_MED = ds.where((ds.longitude>=-8) & (ds.longitude<=42), drop=True) # original one
ds_MED = ds_MED.where((ds_MED.latitude>=26) & (ds_MED.latitude<=45), drop=True)
# Select depth range
ds_MED = ds_MED.where((ds_MED.depth<=100), drop=True) # original one
# mean depth and space
ds_MED = ds_MED.mean(dim=['depth', 'latitude', 'longitude'])
# To pandas
#phy = ds_MED['phyc'].to_dataframe()
pp = ds_MED['nppv'].to_dataframe()
# Save data
pp_annual_MED = pp.resample('As').mean()
pp_annual_MED.index = pp_annual_MED.index.year
pp_annual_MED.rename(columns={'nppv':'Whole Med'}, inplace=True)
pp_annual_MED.to_csv('cmems_PP_Med.csv')

# 3. plot
pp_annual = pd.concat([pp_annual_NW, pp_annual_MED], axis=1)

fig = plt.figure()
# ax1
ax = plt.subplot2grid((1, 1), (0, 0))
pp_annual.plot(ax=ax)
plt.ylabel(r'PP $\rm (mgC\,m^{-3}\,d^{-1})$')
plt.grid()
plt.title('1/12th degree CMEMS global analysis')
# save
fig.set_size_inches(w=8,h=12)
fig_name = 'CMEMS_PP_NW-vs-MED.png' 
fig.savefig(fig_name, dpi=200)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

