'''
A new script for glider processing.
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime


ds = xr.open_dataset('/home/cyrf0006/data/gliders_data/SEA024/20190807/netcdf/SEA024_20190807_l2.nc')

da = ds['temperature']
df = da.to_pandas()
plt.pcolor(df.index, df.columns, df.T.fillna(method='ffill'))
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()     

da = ds['salinity']
df = da.to_pandas()
plt.pcolor(df.index, df.columns, df.T)
plt.gca().invert_yaxis()
plt.show()     

da = ds['turbidity']
df = da.to_pandas()
plt.pcolor(df.index, df.columns, np.log10(df.T))
plt.gca().invert_yaxis()
plt.show()     

da = ds['oxygen_concentration']
df = da.to_pandas()
plt.pcolor(df.index, df.columns, df.T)
plt.gca().invert_yaxis()
plt.show()     




da_head = ds['heading']
df_head = da_head.to_pandas()

da_depth = ds['depth']
df_depth = da_depth.to_pandas()


plt.scatter(df_head[40000:70000]*10, df_depth[40000:70000], c=np.arange(0,30000, 1))
#plt.scatter(df_head[40000:50000]*10.0, df_depth[40000:50000], c=df_head.index.minute[40000:50000])
plt.ylabel('Depth (m)')
plt.xlabel('Heading')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()


