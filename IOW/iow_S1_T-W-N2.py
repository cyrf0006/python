import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd
import seawater as sw

# Some infos:
mab_adcp = 1.58
mooring_depth = 83.0
adcpdir = 'up'

#XLIM = [pd.Timestamp('2010-03-01 15:30:00'), pd.Timestamp('2010-03-01 16:30:00')]

XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 05:00:00')]
fig_name = 'S1_T-W-N2.png'

#XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 06:00:00')]
#fig_name = 'S1_TW_zoom_IWs.pdf'


cutoff_period_high = 1800.0  # desired cutoff in seconds (high-pass)
cutoff_period_low = 2.0*60*60 

# If you don't use squeeze_me = True, then Pandas doesn't like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True fixes that.
#T_dict = loadmat('/media/cyrf0006/Seagate1TB/IOW/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
#ADCP_dict = loadmat('/media/cyrf0006/Seagate1TB/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)
T_dict = loadmat('./data/Titp_S1_p4std_1sAve.mat',squeeze_me=True)
ADCP_dict = loadmat('./data/IOWp01.mat',squeeze_me=True)
sbe_dict = loadmat('./iow_mooring/AL351_TSC2_S1_sbe_all.mat',squeeze_me=True)


### --------- Using np arrays --------- ###
def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac

# To Pandas DataFrame
E = ADCP_dict['SerEmmpersec']/1000.0 # now in cm/s
N = ADCP_dict['SerNmmpersec']/1000.0
W = ADCP_dict['SerVmmpersec']/1000.0 # now positive upward
T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])
Tmat = sbe_dict['Temperature']
Smat = sbe_dict['Salinity']
Dmat = sbe_dict['Density']
zVecSBE = sbe_dict['SBE_depth']
yearday = sbe_dict['SBE_decday']
pdTime = pd.to_datetime('2010') + pd.to_timedelta(yearday - 1, unit='d')

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
TT = pd.DataFrame(Tmat, index=pdTime, columns=zVecSBE) # now in m/s
S = pd.DataFrame(Smat, index=pdTime, columns=zVecSBE)
D = pd.DataFrame(Dmat, index=pdTime, columns=zVecSBE)

# Cleaning
E[E<-32] = np.NaN
N[N<-32] = np.NaN
W[W<-32] = np.NaN

# time average
Ebin = E.resample('60s').mean()
Nbin = N.resample('60s').mean()
Wbin = W.resample('60s').mean()
Tbin = T.resample('1s').mean()
TTbin = TT.resample('60s').mean()
Sbin = S.resample('60s').mean()
Dbin = D.resample('60s').mean()
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
TTbin = TTbin.loc[(TTbin.index >= XLIM[0]) & (TTbin.index <= XLIM[1])]
Sbin = Sbin.loc[(Sbin.index >= XLIM[0]) & (Sbin.index <= XLIM[1])]
Dbin = Dbin.loc[(Dbin.index >= XLIM[0]) & (Dbin.index <= XLIM[1])]



# Compute SW properties
N2 = sw.bfrq(Sbin.mean(), TTbin.mean(), TTbin.columns, 55)[0]
zVecN2 = (zVecSBE[1:] + zVecSBE[:-1]) / 2
import SW_extras as swe
N2_period_sec = 3600.0/swe.cph(N2)
N2_period_sec[-1] = np.nan # eliminate last (negative) value

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
fig = plt.figure(2)
ax0 = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=1)
#ax0.plot(Dbin.mean()-1000, Dbin.columns)
ax0.plot(N2_period_sec/60, zVecN2)
ax0.set_ylabel(r'Depth (m)')
ax0.invert_yaxis()
ax0.set_ylim(53, 83)
ax0.set_xlim(1, 5)
ax0.invert_yaxis()
ax0.grid()
## ax1 = ax0.twiny()
## ax1.plot(N2_period_sec, zVecN2)
## ax1.grid()
## ax0.set_xlabel(r'$\rm \sigma_0 (g kg^{-1})$')
ax0.set_xlabel(r'$\rm T_{N} (min)$')
## ax1.set_ylim(40, 83)
#ax0.xaxis.set_ticks([0, 150, 300, 450])
ax0.xaxis.set_ticks([0, 2, 4])

ax2 = plt.subplot2grid((1, 6), (0, 1), rowspan=1, colspan=5)
levels = np.linspace(0,10, 21)
c = plt.contourf(Tbin.index, Tbin.columns, Tbin.T, levels, cmap=plt.cm.RdBu_r)
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
ct = plt.contour(Wbin.index, Wbin.columns, Whigh, [0.0,], colors='w', linewidths=0.25)
#ct = plt.contour(Wbin.index, Wbin.columns, Whigh, [.0001,], colors='k', linewidths=0.25)
ct = plt.contourf(Wbin.index, Wbin.columns, Whigh, [-0.1, 0.0], cmap=plt.cm.binary_r, alpha=.3 )
plt.colorbar(c)
ax2.set_xlim(XLIM[0], XLIM[1])
ax2.set_ylim(53, 83)
ax2.tick_params(labelbottom='on')
ax2.tick_params(labelleft='off')
#ax2.set_ylabel(r'Depth (m)')
ax2.set_xlabel(Wbin.index[1].strftime('%d-%b-%Y'))
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(min15)
ax2.xaxis.set_major_formatter(dfmt)
ax2.xaxis.set_minor_locator(min1)
ax2.text(XLIM[0], 81, r'  T($^{\circ}$C)', horizontalalignment='left', verticalalignment='center', fontsize=14, color='w')


fig.set_size_inches(w=8, h=5)
fig.set_dpi(150)
fig.tight_layout()
fig.savefig(fig_name)
plt.show()





