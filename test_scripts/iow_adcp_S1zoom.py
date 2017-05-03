import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd

# If you don't use squeeze_me = True, then Pandas doesn't like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True fixes that.
#T_dict = loadmat('/media/cyrf0006/Seagate1TB/IOW/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
#ADCP_dict = loadmat('/media/cyrf0006/Seagate1TB/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)
T_dict = loadmat('/media/cyrf0006/Fred32/IOW/Titp_S1_p4std_1sAve.mat',squeeze_me=True)
ADCP_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)
mab_adcp = 1.58
mooring_depth = 83
adcpdir = 'up'

### --------- Using np arrays --------- ###
def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac

# Store temperature data
T = { k: T_dict[k] for k in ['habVec', 'Tbin']}
T['date_time'] = [matlab2datetime(tval) for tval in T_dict['timeVec']]
T['zVec'] = [mooring_depth-tval for tval in T_dict['habVec']]


# make a new dictionary with just dependent variables we want
# (we handle the time variable separately, below)
ADCP = { k: ADCP_dict[k] for k in ['SerNmmpersec', 'SerEmmpersec', 'SerVmmpersec']}

# To Pandas DataFrame
E = ADCP_dict['SerEmmpersec']/1000.0 # now in cm/s
N = ADCP_dict['SerNmmpersec']/1000.0
W = ADCP_dict['SerVmmpersec']/1000.0

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
# Pandas time
pandaTime = pd.date_range('2010-02-28 10:22:38', '2010-03-04 09:31:28', freq='2s')

# DataFrame with pandaTime as index
E = pd.DataFrame(E, index=pandaTime, columns=zVec) # now in m/s
N = pd.DataFrame(N, index=pandaTime, columns=zVec)
W = pd.DataFrame(W, index=pandaTime, columns=zVec)

## # Transpose and create DataFrame
## E = pd.DataFrame(E.T, columns=pandaTime, index=zVec) # now in m/s
## N = pd.DataFrame(N.T, columns=pandaTime, index=zVec)
## W = pd.DataFrame(W.T, columns=pandaTime, index=zVec)

# Cleaning
E[E<-32] = np.NaN
N[N<-32] = np.NaN
W[W<-32] = np.NaN

# time average
Ebin = E.resample('1min').mean()
Nbin = N.resample('1Min').mean()
Wbin = W.resample('1Min').mean()

# Remove "barotropic" currents (MATRIX IS TRANSPOSED!)
Ebin = Ebin.sub(Ebin.mean(axis=1), axis=0)
Nbin = Nbin.sub(Nbin.mean(axis=1), axis=0)

# plot
## fig = plt.figure()

days = dates.DayLocator()
hours1 = dates.HourLocator(interval=1)
dfmt = dates.DateFormatter('%H')
min5 = dates.MinuteLocator(interval=5)

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
fs = 1/2.0       # sample rate, Hz
fs_bin = 1/(60.0)       # sample rate, Hz1
cutoff_high = 1/(1800.0)  # desired cutoff frequency of the filter, Hz
cutoff_low = 1/(2.0*60*60)

# Get the filter coefficients so we can check its frequency response.
## CAREFULL ON W HICH DIMENSION THE FILTER IS APPLIED!!!
#b, a = butter_lowpass(cutoff_high, fs_bin, order)
EE = np.nan_to_num(Ebin)
NN = np.nan_to_num(Nbin)
WW = np.nan_to_num(Wbin)

Ehigh = butter_highpass_filter(EE.T, cutoff_high, fs_bin, order)
Nhigh = butter_highpass_filter(NN.T, cutoff_high, fs_bin, order)
Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)

# Demonstrate the use of the filter.
# Filter the data, and plot both the original and filtered signals.
fig = plt.figure()
XLIM = [pd.Timestamp('2010-03-01 12:00:00'), pd.Timestamp('2010-03-01 18:00:00')]
fig_name = 'S1_adcp_zoom.pdf'


ax3 = plt.subplot(3, 1, 1)
#levels = np.linspace(-.03, .03, 4)
levels = [-.03, -.01, -.005, .005, .01, .03]
#levels = np.linspace(-.06, .06, 6)
levelsT = np.linspace(0, 10, 11)
c = plt.contourf(Ebin.index, Ebin.columns, Ehigh, levels, cmap=plt.cm.RdBu, extend="both")
ct = plt.contour(T['date_time'], T['zVec'], T['Tbin'], levelsT, colors='k', linewidths=0.5)
plt.colorbar(c)
ax3.set_xlim(XLIM[0], XLIM[1])
ax3.set_ylim(0, 83)
ax3.tick_params(labelbottom='off')
ax3.set_ylabel(r'Depth (m)')
ax3.invert_yaxis()
ax3.xaxis.set_major_locator(hours1)
ax3.xaxis.set_major_formatter(dfmt)
ax3.xaxis.set_minor_locator(min5)


ax4 = plt.subplot(3, 1, 2)
#levels = np.linspace(-.04, .04, 5)
c = plt.contourf(Nbin.index, Nbin.columns, Nhigh, levels, cmap=plt.cm.RdBu, extend="both")
plt.colorbar(c)
ax4.set_xlim(XLIM[0], XLIM[1])
ax4.set_ylim(0, 83)
ax4.tick_params(labelbottom='off')
#ax4.set_ylabel(r'Depth (m)')
ax4.invert_yaxis()
ax4.xaxis.set_major_locator(hours1)
ax4.xaxis.set_major_formatter(dfmt)
ax4.xaxis.set_minor_locator(min5)

ax5 = plt.subplot(3, 1, 3)
#levels = np.linspace(-.004, .004, 5)
levels = [-.004, -.002, -.001, .001, .002, .004]
#levels = [-.003, -.001, -.0005, .0005, .001, .003]
c = plt.contourf(Wbin.index, Wbin.columns, Whigh, levels, cmap=plt.cm.RdBu, extend="both")
plt.colorbar(c)
ax5.set_xlim(XLIM[0], XLIM[1])
ax5.set_ylim(0, 83)
ax5.tick_params(labelbottom='on')
#ax5.set_ylabel(r'Depth (m)')
ax5.invert_yaxis()
ax5.xaxis.set_major_locator(hours1)
ax5.xaxis.set_major_formatter(dfmt)
ax5.xaxis.set_minor_locator(min5)



plt.subplots_adjust(hspace=0.35)
plt.show()
fig.savefig(fig_name)
