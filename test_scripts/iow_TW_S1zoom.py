import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd

# Some infos:
mab_adcp = 1.58
mooring_depth = 83.0
adcpdir = 'up'

fig_name = 'S1_TW_toberename.pdf'
#XLIM = [pd.Timestamp('2010-03-01 15:30:00'), pd.Timestamp('2010-03-01 16:30:00')]

#XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 06:00:00')]
XLIM = [pd.Timestamp('2010-03-01 11:00:00'), pd.Timestamp('2010-03-01 14:00:00')]
fig_name = 'S1_TW_toberename.pdf'

#XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 06:00:00')]
#fig_name = 'S1_TW_zoom_IWs.pdf'


cutoff_period_high = 1800.0  # desired cutoff in seconds (high-pass)
cutoff_period_low = 2.0*60*60 

# If you don't use squeeze_me = True, then Pandas doesn't like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True fixes that.
#T_dict = loadmat('/media/cyrf0006/Seagate1TB/IOW/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
#ADCP_dict = loadmat('/media/cyrf0006/Seagate1TB/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)
T_dict = loadmat('/media/cyrf0006/Fred32/IOW/Titp_S1_p4std_1sAve.mat',squeeze_me=True)
ADCP_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)


### --------- Using np arrays --------- ###
def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac

# Store temperature data
## T = { k: T_dict[k] for k in ['habVec', 'Tbin']}
## T['date_time'] = [matlab2datetime(tval) for tval in T_dict['timeVec']]
## T['zVec'] = [mooring_depth-tval for tval in T_dict['habVec']]


# make a new dictionary with just dependent variables we want
# (we handle the time variable separately, below)
#ADCP = { k: ADCP_dict[k] for k in ['SerNmmpersec', 'SerEmmpersec', 'SerVmmpersec']}

# To Pandas DataFrame
E = ADCP_dict['SerEmmpersec']/1000.0 # now in cm/s
N = ADCP_dict['SerNmmpersec']/1000.0
W = ADCP_dict['SerVmmpersec']/1000.0 # now positive upward
T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])


# time vector (Using datetime)
ADCPtime_list = []
for i in range(0, len(ADCP_dict['SerMon'])):
    date1 = datetime.datetime(year=int(ADCP_dict['SerYear'][i]+2000), month=int(ADCP_dict['SerMon'][i]), day=int(ADCP_dict['SerDay'][i]), hour=int(ADCP_dict['SerHour'][i]), minute=int(ADCP_dict['SerMin'][i]), second=int(ADCP_dict['SerSec'][i]))
    ADCPtime_list.append(date1)
    
timeVec = pd.DataFrame(ADCPtime_list)

if adcpdir is 'up':
    habADCP = mab_adcp + (ADCP_dict['SerBins']-ADCP_dict['SerBins'][0]) + ADCP_dict['RDIBin1Mid']
    zVec = mooring_depth - habADCP
else:
    zVec = mab_adcp + (ADCP_dict['SerBins']-ADCP_dict['SerBins'][0]) + ADCP_dict['RDIBin1Mid']
    habADCP = mooring_depth - zVec
# --------------------------------------# 

### ---------- Using Pandas -------------- ###
# Pandas time (must have the info)
pandaTime = pd.date_range('2010-02-28 10:22:38', '2010-03-04 09:31:28', freq='2s')
pandaTimeT = pd.date_range('2010-02-28 11:02:00', '2010-03-04 07:40:23', freq='1s')

# DataFrame with pandaTime as index
E = pd.DataFrame(E, index=pandaTime, columns=zVec) # now in m/s
N = pd.DataFrame(N, index=pandaTime, columns=zVec)
W = pd.DataFrame(W, index=pandaTime, columns=zVec)
T = pd.DataFrame(T, index=pandaTimeT, columns=zVecT)

## # Transpose and create DataFrame
## E = pd.DataFrame(E.T, columns=pandaTime, index=zVec) # now in m/s
## N = pd.DataFrame(N.T, columns=pandaTime, index=zVec)
## W = pd.DataFrame(W.T, columns=pandaTime, index=zVec)

# Cleaning
E[E<-32] = np.NaN
N[N<-32] = np.NaN
W[W<-32] = np.NaN

# time average
Ebin = E.resample('60s').mean()
Nbin = N.resample('60s').mean()
Wbin = W.resample('60s').mean()
Tbin = T.resample('1s').mean()
fs = 1/2.0       # sample rate raw, Hz
fs_bin = 1/10.0 # sample rate binned, Hz

# Remove "barotropic" currents (MATRIX IS TRANSPOSED!)
Ebin = Ebin.sub(Ebin.mean(axis=1), axis=0)
Nbin = Nbin.sub(Nbin.mean(axis=1), axis=0)

days = dates.DayLocator()
min15 = dates.HourLocator(interval=1)
dfmt = dates.DateFormatter('%H:%M')
min1 = dates.MinuteLocator(interval=15)

# Cut timeseries
Ebin = Ebin.loc[(Ebin.index >= XLIM[0]) & (Ebin.index <= XLIM[1])]
Nbin = Nbin.loc[(Nbin.index >= XLIM[0]) & (Nbin.index <= XLIM[1])]
Wbin = Wbin.loc[(Wbin.index >= XLIM[0]) & (Wbin.index <= XLIM[1])]
Tbin = Tbin.loc[(Tbin.index >= XLIM[0]) & (Tbin.index <= XLIM[1])]


#### ------ filter timeseries ----- ####
from scipy.signal import butter, lfilter, freqz, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Filter requirements.
order = 8
cutoff_high = 1/cutoff_period_high  # desired cutoff frequency of the filter, Hz
cutoff_low = 1/cutoff_period_low

WW = np.nan_to_num(Wbin)

Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)


#### --------- plots ---------- ####
fig = plt.figure()
levels = np.linspace(0,10, 21)
ax = plt.subplot(1, 1, 1)
c = plt.contourf(Tbin.index, Tbin.columns, Tbin.T, levels, cmap=plt.cm.RdBu_r)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
ct = plt.contour(Wbin.index, Wbin.columns, Whigh, [0.0,], colors='w', linewidths=0.25)
#ct = plt.contour(Wbin.index, Wbin.columns, Whigh, [.0001,], colors='k', linewidths=0.25)
ct = plt.contourf(Wbin.index, Wbin.columns, Whigh, [-0.1, 0.0], cmap=plt.cm.binary_r, alpha=.3 )
plt.colorbar(c)
ax.set_xlim(XLIM[0], XLIM[1])
#ax.set_ylim(53, 83)
ax.set_ylim(55, 65)
ax.tick_params(labelbottom='on')
ax.set_ylabel(r'Depth (m)')
ax.set_xlabel(Wbin.index[1].strftime('%d-%b-%Y'))
ax.invert_yaxis()
ax.xaxis.set_major_locator(min15)
ax.xaxis.set_major_formatter(dfmt)
ax.xaxis.set_minor_locator(min1)

fig.set_size_inches(w=8, h=5)
fig.set_dpi(150)
fig.tight_layout()
fig.savefig(fig_name)
plt.show()
