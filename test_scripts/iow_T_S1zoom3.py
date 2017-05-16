import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd

#### -------  Some infos to select ----------- #####
mab_adcp = 1.58
mooring_depth = 83.0
adcpdir = 'up'

## XLIM = [pd.Timestamp('2010-02-28 17:00:00'), pd.Timestamp('2010-02-28 23:00:00')]
## fig_name = 'S1_T_zoomPrestorm.pdf'

#XLIM = [pd.Timestamp('2010-03-01 15:00:00'), pd.Timestamp('2010-03-01 21:00:00')]
#fig_name = 'S1_T_zoomDeepening.pdf'

XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 09:00:00')]
fig_name = 'S1_T_zoomIWs.pdf'

cutoff_period_high = 1800.0  # desired cutoff in seconds (high-pass)
cutoff_period_low = 2.0*60*60 
##### ----------------------------------------- ####


# If you don't use squeeze_me = True, then Pandas doesn't like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True fixes that.
T_dict = loadmat('/media/cyrf0006/Fred32/IOW/Titp_S1_p4std_1sAve.mat',squeeze_me=True)

# Dict2Pandas
T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])
pandaTimeT = pd.date_range('2010-02-28 11:02:00', '2010-03-04 07:40:23', freq='1s')

# DataFrame with pandaTime as index
T = pd.DataFrame(T, index=pandaTimeT, columns=zVecT)

# Cut timeseries
T = T.loc[(T.index >= XLIM[0]) & (T.index <= XLIM[1])]

# for plot
days = dates.DayLocator()
min15 = dates.HourLocator(interval=1)
dfmt = dates.DateFormatter('%H:%M')
min1 = dates.MinuteLocator(interval=15)


#### --------- plots ---------- ####
fig = plt.figure()
levels = np.linspace(0,10, 41)
ax = plt.subplot(1, 1, 1)
c = plt.contourf(T.index, T.columns, T.T, levels, cmap=plt.cm.RdBu_r)
#ct = plt.contour(T.index, T.columns, T.T, [8,], colors='k' , linewidths=2)
plt.colorbar(c)
ax.set_xlim(XLIM[0], XLIM[1])
ax.set_ylim(53, 83)
ax.tick_params(labelbottom='on')
ax.set_ylabel(r'Depth (m)')
ax.invert_yaxis()
ax.xaxis.set_major_locator(min15)
ax.xaxis.set_major_formatter(dfmt)
ax.xaxis.set_minor_locator(min1)

#plt.subplots_adjust(hspace=0.35)
fig.set_size_inches(w=8, h=5)
fig.set_dpi(300)
fig.tight_layout()
fig.savefig(fig_name)
plt.show()

#fig.savefig(fig_name)
