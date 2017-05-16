import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd

# Some infos:
mooring_depth = 83.0

XLIM = [pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00')]
fig_name = 'S1_Talone_toberename.png'

#### ---- NIOZ thermostor chain ---- ####
# If you don't use squeeze_me = True, then Pandas doesn't like the arrays in the dictionary
T_dict = loadmat('/media/cyrf0006/Fred32/IOW/Titp_S1_p4std_10minAve.mat',squeeze_me=True)

def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac

T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])
pandaTimeT = pd.date_range('2010-02-28 11:07:00', '2010-03-04 07:37:00', freq='10min')
T = pd.DataFrame(T, index=pandaTimeT, columns=zVecT)

# time average
#Tbin = T.resample('1s').mean()
Tbin = T

# Cut timeseries
Tbin = Tbin.loc[(Tbin.index >= XLIM[0]) & (Tbin.index <= XLIM[1])]





#### --------- plots ---------- ####
days = dates.DayLocator()
hours6 = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%b %d')
hours1 = dates.HourLocator(interval=1)

fig = plt.figure()
levels = np.linspace(0,10, 41)
levels2 = np.linspace(0,10, 11)
ax = plt.subplot(1, 1, 1)
c = plt.contourf(Tbin.index, Tbin.columns, Tbin.T, levels, cmap=plt.cm.RdBu_r)
ct = plt.contour(Tbin.index, Tbin.columns, Tbin.T, levels2, colors='k', linewidth=0.1)
plt.colorbar(c)
ax.set_xlim(XLIM[0], XLIM[1])
ax.set_ylim(53, 83)
ax.tick_params(labelbottom='on')
ax.set_ylabel(r'Depth (m)')
ax.invert_yaxis()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(dfmt)
ax.xaxis.set_minor_locator(hours6)
ax.text(XLIM[0], 81, r'  T($^{\circ}$C)', horizontalalignment='left', verticalalignment='center', fontsize=14, color='w')

## # add zoomed rectangle
## import matplotlib.dates as mdates
## start = mdates.date2num(pd.Timestamp('2010-03-02 03:00:00'))
## end = mdates.date2num(pd.Timestamp('2010-03-02 09:00:00'))
## rect_x = [start, end, end, start, start]
## rect_y = [82.9, 82.9, 53.1, 53.1, 82.9]
## rect = zip(rect_x, rect_y)
## Rgon2 = plt.Polygon(rect, color='w', ls='--', lw=2, alpha=1, fill=False)
## #Rgon = plt.Polygon(rect,color='none', edgecolor='red', facecolor="red", hatch='/')
## ax.add_patch(Rgon2)

fig.set_size_inches(w=8, h=5)
fig.set_dpi(150)
fig.tight_layout()
fig.savefig(fig_name)
plt.show()
