import glidertools as gt
from cmocean import cm as cmo  # we use this for colormaps
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

filename = '/home/cyrf0006/data/gliders_data/SEA003/20190501/netcdf/SEA003_20190501_l1.nc'
ds = xr.open_dataset(filename)

dives = ds.profile_index
depth = ds.depth
salt = ds.salinity

x = np.array(dives)  # ensures these are arrays
y = np.array(depth)

# plot original data
gt.plot(dives, depth, salt, cmap=cmo.haline, robust=True)
plt.title('Original Data')
plt.show()

# Global filtering: outlier limits (IQR & STD)
salt_iqr = gt.cleaning.outlier_bounds_iqr(salt, multiplier=4)
salt_std = gt.cleaning.outlier_bounds_std(salt, multiplier=4)

gt.plot(x, y, salt_iqr, cmap=cmo.haline, robust=True)
plt.title('Outlier Bounds IQR Method')

gt.plot(x, y, salt_std, cmap=cmo.haline, robust=True)
plt.title('Outlier Bounds Stdev Method')
plt.show()


# Horizontal filtering: differential outliers
salt_horz = gt.cleaning.horizontal_diff_outliers(
    x, y, salt,
    multiplier=4,
    depth_threshold=400,
    mask_frac=0.1
)

gt.plot(x, y, salt, cmap=cmo.haline)
plt.title('Original dataset')
plt.show()

gt.plot(x, y, salt_horz, cmap=cmo.haline)
plt.title('Horizontal Differential Outliers removed')
plt.show()


# Despiking
salt_base, salt_spike = gt.cleaning.despike(salt, window_size=5, spike_method='median')

fig, ax = plt.subplots(2, 1, figsize=[9, 6], sharex=True, dpi=90)

gt.plot(x, y, salt_base, cmap=cmo.haline, ax=ax[0])
ax[0].set_title('Despiked using median filter')
ax[0].cb.set_label('Salinity baseline')
#ax[0].set_xlim(50,150)
ax[0].set_xlabel('')

gt.plot(x, y, salt_spike, cmap=cm.RdBu_r, vmin=-6e-3, vmax=6e-3, ax=ax[1])
ax[1].cb.set_label('Salinity spikes')
#ax[1].set_xlim(50,150)

plt.xticks(rotation=0)
plt.show()
