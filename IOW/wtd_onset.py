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
#T_dict = loadmat('/media/cyrf0006/Fred32/IOW/Titp_S1_p4std_1sAve.mat',squeeze_me=True)
#ADCP_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)
T_dict = loadmat('./data/Titp_S1_p4std_1sAve.mat',squeeze_me=True)
ADCP_dict = loadmat('./data/IOWp01.mat',squeeze_me=True)
mab_adcp = 1.58
mooring_depth = 83
adcpdir = 'up'
#XLIM = [pd.Timestamp('2010-03-01 23:00:00'), pd.Timestamp('2010-03-02 1:00:00')]
#XLIM = [pd.Timestamp('2010-03-02 3:00:00'), pd.Timestamp('2010-03-02 5:00:00')]
#XLIM = [pd.Timestamp('2010-03-01 7:00:00'), pd.Timestamp('2010-03-01 9:00:00')]
#XLIM = [pd.Timestamp('2010-03-01 11:00:00'), pd.Timestamp('2010-03-01 14:00:00')]
#XLIM = [pd.Timestamp('2010-03-01 14:00:00'), pd.Timestamp('2010-03-01 17:00:00')]
XLIM = [pd.Timestamp('2010-03-01 11:00:00'), pd.Timestamp('2010-03-01 17:00:00')]
fig_name = 'S1_onset.png'

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

#### ---- NIOZ thermostor chain ---- ####
T_dict = loadmat('./data/Titp_S1_p4std_1sAve.mat',squeeze_me=True)
T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])
pandaTimeT = pd.date_range('2010-02-28 11:02:00', '2010-03-04 07:40:23', freq='1s')
T = pd.DataFrame(T, index=pandaTimeT, columns=zVecT)
Tbin = T

# Cut timeseries
Tbin = Tbin.loc[(Tbin.index >= XLIM[0]) & (Tbin.index <= XLIM[1])]

### ---------- Using Pandas -------------- ###
# Pandas time
pandaTime = pd.date_range('2010-02-28 10:22:38', '2010-03-04 09:31:28', freq='2s')

# DataFrame with pandaTime as index
E = pd.DataFrame(E, index=pandaTime, columns=zVec) # now in m/s
N = pd.DataFrame(N, index=pandaTime, columns=zVec)
W = pd.DataFrame(W, index=pandaTime, columns=zVec)

# Cleaning
E[E<-32] = np.NaN
N[N<-32] = np.NaN
W[W<-32] = np.NaN

# time average
Ebin = E.resample('2min').mean()
Nbin = N.resample('2Min').mean()
Wbin = W.resample('2Min').mean()
fs_bin = 1/(60.0*2.0)       # sample rate, Hz1

# Remove "barotropic" currents (MATRIX IS TRANSPOSED!)
Ebin = Ebin.sub(Ebin.mean(axis=1), axis=0)
Nbin = Nbin.sub(Nbin.mean(axis=1), axis=0)

days = dates.DayLocator()
hours1 = dates.HourLocator(interval=1)
h1 = dates.HourLocator(interval=1)
dfmt = dates.DateFormatter('%H:%M')
dfmt2 = dates.DateFormatter('%H:%M')
min5 = dates.MinuteLocator(interval=5)
min15 = dates.MinuteLocator(interval=15)

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
order = 6
cutoff_high = 1/(1800.0)  # desired cutoff frequency of the filter, Hz

EE = np.nan_to_num(Ebin)
NN = np.nan_to_num(Nbin)
WW = np.nan_to_num(Wbin)
Ehigh = butter_highpass_filter(EE.T, cutoff_high, fs_bin, order)
Nhigh = butter_highpass_filter(NN.T, cutoff_high, fs_bin, order)
Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)



#### ---- plot ---- ####


fig = plt.figure(1)
# AX1 - Onset
ax1 = plt.subplot2grid((2, 1), (0, 0))
levels = np.linspace(-.005, .005, 11)
c = plt.contourf(Wbin.index, Wbin.columns, Whigh, levels, cmap=plt.cm.RdBu_r, extend="both")
levels2 = np.linspace(0,10, 11)
ct = plt.contour(Tbin.index, Tbin.columns, Tbin.T, levels2, colors='k', linewidth=0.1)
ax1.set_xlim(XLIM[0], XLIM[1])
ax1.set_ylim(20, 70)
ax1.tick_params(labelbottom='on')
ax1.set_ylabel(r'Depth (m)')
ax1.invert_yaxis()
ax1.xaxis.set_major_locator(hours1)
ax1.xaxis.set_major_formatter(dfmt)
ax1.xaxis.set_minor_locator(min15)
ax1.set_xlim(XLIM[0], XLIM[1])
#ax1.set_xlabel(XLIM[0].strftime('%d-%b-%Y'))
ax1.text(pd.Timestamp('2010-03-01 11:30:00'), 25, r"  $\rm w' ( m s^{-1})$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')

## # Add dashed rectangle for zoom
import matplotlib.dates as mdates # For zoomed rectangle
XLIM2 = [pd.Timestamp('2010-03-01 11:30:00'), pd.Timestamp('2010-03-01 14:00:00')]
start2 = mdates.date2num(XLIM2[0])
end2 = mdates.date2num(XLIM2[1])
rect_x = [start2, end2, end2, start2, start2]
rect_y = [69.8, 69.8, 20.2, 20.2, 69.8]
ax1.plot(rect_x, rect_y, '--k')


#XLIM2 = XLIM
# AX2 - Zoom
ax2 = plt.subplot2grid((2, 1), (1, 0))
c = plt.contourf(Wbin.index, Wbin.columns, Whigh, levels, cmap=plt.cm.RdBu_r, extend="both")
levels2 = np.linspace(0,10, 11)
ct = plt.contour(Tbin.index, Tbin.columns, Tbin.T, levels2, colors='k', linewidth=0.1)
ax2.set_xlim(XLIM2[0], XLIM2[1])
ax2.set_ylim(20, 70)
ax2.tick_params(labelbottom='on')
ax2.set_ylabel(r'Depth (m)', fontweight='bold')
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(h1)
ax2.xaxis.set_major_formatter(dfmt2)
ax2.xaxis.set_minor_locator(min5)
ax2.set_xlabel(XLIM2[0].strftime('%d-%b-%Y'))

# Langmuir propagation
propax = [pd.Timestamp('2010-03-01 12:20:00'), pd.Timestamp('2010-03-01 12:42:00')]
propay = [30, 60]
ax2.plot(propax, propay, '--k')

# Colorbar for both
cax = plt.axes([0.92,0.15,0.015,0.7])
plt.colorbar(c, cax=cax, ticks=[-.005, -.003, -.001, .001, .003, .005])



#### ---- Save Figure ---- ####
fig.set_size_inches(w=9, h=7)
fig.set_dpi(300)
#fig.tight_layout()
fig.savefig(fig_name)
#plt.show()

