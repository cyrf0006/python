import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
from scipy.io import loadmat
#import datetime
import pandas as pd
import seawater as sw

# Some infos:
mab_adcp = 1.58
mooring_depth = 83.0
adcpdir = 'up'

#XLIM = [pd.Timestamp('2010-03-01 15:30:00'), pd.Timestamp('2010-03-01 16:30:00')]

XLIM = [pd.Timestamp('2010-02-28 00:00:00'), pd.Timestamp('2010-03-03 6:00:00')]

#XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 06:00:00')]
fig_name = 'S1_SBE.pdf'

#XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 06:00:00')]
#fig_name = 'S1_TW_zoom_IWs.pdf'



sbe_dict = loadmat('/home/cyrf0006/research/IOW/iow_mooring/AL351_TSC2_S1_sbe_all.mat',squeeze_me=True)

Tmat = sbe_dict['Temperature']
Smat = sbe_dict['Salinity']
Dmat = sbe_dict['Density']
zVec = sbe_dict['SBE_depth']
yearday = sbe_dict['SBE_decday']
pdTime = pd.to_datetime('2010') + pd.to_timedelta(yearday - 1, unit='d')


# DataFrame with pandaTime as index
T = pd.DataFrame(Tmat, index=pdTime, columns=zVec) # now in m/s
S = pd.DataFrame(Smat, index=pdTime, columns=zVec)
D = pd.DataFrame(Dmat, index=pdTime, columns=zVec)

## # time average
Tbin = T.resample('60s').mean()
Sbin = S.resample('60s').mean()
Dbin = D.resample('60s').mean()
fs = 1/15.0       # sample rate raw, Hz
fs_bin = 1/60.0 # sample rate binned, Hz

days = dates.DayLocator()
major_xticks = dates.HourLocator(interval=1)
dfmt = dates.DateFormatter('%H:%M')
minor_xticks = dates.MinuteLocator(interval=15)

# Cut timeseries
Tbin = Tbin.loc[(Tbin.index >= XLIM[0]) & (Tbin.index <= XLIM[1])]
Sbin = Sbin.loc[(Sbin.index >= XLIM[0]) & (Sbin.index <= XLIM[1])]
Dbin = Dbin.loc[(Dbin.index >= XLIM[0]) & (Dbin.index <= XLIM[1])]

#keyboard

# Compute SW properties
N2 = sw.bfrq(Sbin.mean(), Tbin.mean(), Tbin.columns, 55)[0]
zVecN2 = (zVec[1:] + zVec[:-1]) / 2
import SW_extras as swe
N2_period_sec = 3600.0/swe.cph(N2)


#### --------- plots ---------- ####
## fig = plt.figure(1)
## #levels = np.linspace(0,10, 21)
## ax = plt.subplot(1, 1, 1)
## c = plt.contourf(Dbin.index, Dbin.columns, Dbin.T, 30, cmap=plt.cm.RdBu_r)
## #matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
## #ct = plt.contour(Wbin.index, Wbin.columns, Whigh, [0.0,], colors='w', linewidths=0.25)
## #ct = plt.contour(Wbin.index, Wbin.columns, Whigh, [.0001,], colors='k', linewidths=0.25)
## #ct = plt.contourf(Wbin.index, Wbin.columns, Whigh, [-0.1, 0.0], cmap=plt.cm.binary_r, alpha=.3 )
## plt.colorbar(c)
## ax.set_xlim(XLIM[0], XLIM[1])
## ax.set_ylim(40, 83)
## ax.tick_params(labelbottom='on')
## ax.set_ylabel(r'Depth (m)')
## ax.set_xlabel(Tbin.index[1].strftime('%d-%b-%Y'))
## ax.invert_yaxis()
## ax.xaxis.set_major_locator(major_xticks)
## ax.xaxis.set_major_formatter(dfmt)
## ax.xaxis.set_minor_locator(minor_xticks)

## fig.set_size_inches(w=8, h=5)
## fig.set_dpi(150)
## fig.tight_layout()
## #fig.savefig(fig_name)
#plt.show()



fig = plt.figure(2)
ax0 = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=1)
#ax0.plot(Dbin.mean()-1000, Dbin.columns)
ax0.plot(N2_period_sec, zVecN2)
ax0.set_ylabel(r'Depth (m)')
ax0.invert_yaxis()
ax0.set_ylim(40, 83)
ax0.set_xlim(0, 500)
ax0.invert_yaxis()
ax0.grid()
## ax1 = ax0.twiny()
## ax1.plot(N2_period_sec, zVecN2)
## ax1.grid()
## ax0.set_xlabel(r'$\rm \sigma_0 (g kg^{-1})$')
ax0.set_xlabel(r'$\rm T_{N} (s)$')
## ax1.set_ylim(40, 83)
ax0.xaxis.set_ticks([0, 150, 300, 450])

ax2 = plt.subplot2grid((1, 6), (0, 1), rowspan=1, colspan=5)
c = plt.contourf(Dbin.index, Dbin.columns, Dbin.T-1000, 30, cmap=plt.cm.RdBu_r)
plt.colorbar(c)
ax2.set_xlim(XLIM[0], XLIM[1])
ax2.set_ylim(40, 83)
ax2.tick_params(labelbottom='on', labelleft='off')
#ax2.set_ylabel(r'Depth (m)')
ax2.set_xlabel(Tbin.index[1].strftime('%d-%b-%Y'))
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(major_xticks)
ax2.xaxis.set_major_formatter(dfmt)
ax2.xaxis.set_minor_locator(minor_xticks)

fig.set_size_inches(w=8, h=5)
fig.set_dpi(150)
fig.tight_layout()
fig.savefig(fig_name)
plt.show()



