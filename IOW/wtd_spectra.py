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


#XLIM = [pd.Timestamp('2010-03-01 21:00:00'), pd.Timestamp('2010-03-02 09:00:00')]
#XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 09:00:00')] # WTD (not used)

#XLIM = [pd.Timestamp('2010-03-01 21:00:00'), pd.Timestamp('2010-03-01 23:00:00')] # IWs 5/6min & 3/4min
#XLIM = [pd.Timestamp('2010-03-01 22:00:00'), pd.Timestamp('2010-03-02 00:00:00')] # IWs 6.67min
#XLIM = [pd.Timestamp('2010-03-02 00:00:00'), pd.Timestamp('2010-03-02 02:00:00')]
#XLIM = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 05:00:00')]
XLIM = [pd.Timestamp('2010-03-01 21:00:00'), pd.Timestamp('2010-03-02 01:00:00')]


fig_name = 'S1_TWspectra_toberename.png'

cutoff_period_high = 1800.0  # desired cutoff in seconds (high-pass)
cutoff_period_low = 2.0*60*60 
##### ----------------------------------------- ####


T_dict = loadmat('./data/Titp_S1_p4std_1sAve.mat',squeeze_me=True)
ADCP_dict = loadmat('./data/IOWp01.mat',squeeze_me=True)

# Dict2Pandas
T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])
pandaTimeT = pd.date_range('2010-02-28 11:02:00', '2010-03-04 07:40:23', freq='1s')

W = ADCP_dict['SerVmmpersec']/1000.0 # now positive upward
habADCP = mab_adcp + (ADCP_dict['SerBins']-ADCP_dict['SerBins'][0]) + ADCP_dict['RDIBin1Mid']
zVec = mooring_depth - habADCP
pandaTime = pd.date_range('2010-02-28 10:22:38', '2010-03-04 09:31:28', freq='2s')
  
# DataFrame with pandaTime as index
T = pd.DataFrame(T, index=pandaTimeT, columns=zVecT)
W = pd.DataFrame(W, index=pandaTime, columns=zVec)
W[W<-32] = np.NaN

# Smooth timeseries
## T = T.resample('30s').mean()
#W = W.resample('3s').mean()
## fs_W = 1/30.0
## fs_T = 1/30.0

fs_W = .5
fs_T = 1


#### --------------------------------- ###

# filter timeseries
from scipy.signal import butter, lfilter, freqz, filtfilt
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#### --------------------------------- ###


# Filter requirements.
order = 6
cutoff_high = 1/(1800.0)  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
## CAREFULL ON W HICH DIMENSION THE FILTER IS APPLIED!!!
WW = np.nan_to_num(W)
Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_W, order)

# Cut timeseries
T = T.loc[(T.index >= XLIM[0]) & (T.index <= XLIM[1])]
W = W.loc[(W.index >= XLIM[0]) & (W.index <= XLIM[1])]


# isolate the timeseries to transform
Tseries65 = np.array(T.iloc[:,60]) # iloc is the index (60 -> 65m)
Wseries70 = np.array(W.iloc[:,10])
Wseries70[np.isnan(Wseries70)] = 0
Wseries65 = np.array(W.iloc[:,12])
Wseries65[np.isnan(Wseries65)] = 0
Wseries30 = np.array(W.iloc[:,47])
Wseries30[np.isnan(Wseries30)] = 0
Whigh = Whigh[12,]
Whigh[np.isnan(Whigh)] = 0

#### ---- Welch method ---- ####
# **** Seems to work, but needs to be verified ********* ## 
# scipy.signal.welch(x, fs=1.0, window='hanning', nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)[source]
from scipy import signal
seriesize = np.size(Tseries65)
N = seriesize/2
f_T65, Pxx_den_T65 = signal.welch(Tseries65, fs_T, nperseg=N, noverlap=N/2, detrend='linear')
seriesize = np.size(Wseries70)
N = seriesize/2
f_W70, Pxx_den_W70 = signal.welch(Wseries70, fs_W, nperseg=N, noverlap=N/2) 
f_W65, Pxx_den_W65 = signal.welch(Wseries65, fs_W, nperseg=N, noverlap=N/2) 
f_W30, Pxx_den_W30 = signal.welch(Wseries30, fs_W, nperseg=N, noverlap=N/2)
#f_Whigh, Pxx_den_Whigh = signal.welch(Whigh, fs_W, nperseg=512, noverlap=256)

# resample spectrum
bins = np.geomspace(f_W30[1], f_W30[-1], 256)
digitized = np.digitize(f_W30, bins)
Pxx_den_W30s = np.array([Pxx_den_W30[digitized == i].mean() for i in range(0, len(bins))])
f_W30s = bins

       
xmin_IW1 = 2.6e-3
ymin_IW1 = [3e-2, 5e-1]
xmin_IW2 = 8.5e-4
ymin_IW2 = [6e-2, 5e-1]
xmin_IW3 = 1/85.0
ymin_IW3 = [1e-2, 2e-1]
xmin_surf = 1.35e-1
ymin_surf = [2e-2, 3e-1]


fig = plt.figure(2)
ax = plt.subplot(1,1,1)
#plt.loglog(f_W30, Pxx_den_W30)
#plt.loglog(f_W65, Pxx_den_W65, alpha=.8)

plt.loglog(f_T65, Pxx_den_T65, alpha=.8)
plt.loglog(f_W30, Pxx_den_W30, alpha=.8)
plt.loglog(f_W65, Pxx_den_W65, alpha=.8)

## plt.semilogx(f_T65, Pxx_den_T65, alpha=.8)
## plt.semilogx(f_W30, Pxx_den_W30, alpha=.8)
## plt.semilogx(f_W65, Pxx_den_W65, alpha=.8)

#plt.loglog(f_W30s, Pxx_den_W30s)
#plt.loglog(f_W70, Pxx_den_W70)
#plt.loglog(f_W65, Pxx_den_W65)
#plt.loglog(f_W30, Pxx_den_W30)
plt.plot([xmin_IW1, xmin_IW1],ymin_IW1, '--k')
plt.plot([xmin_IW2, xmin_IW2],ymin_IW2, '--k')
plt.plot([xmin_IW3, xmin_IW3],ymin_IW3, '--k')
plt.plot([xmin_surf, xmin_surf],ymin_surf, '--k')

#plt.loglog(2*np.pi/(60.0*f_Whigh), Pxx_den_Whigh)
#plt.xlim([4e0, 3e1])
plt.ylabel('PSD [V**2/Hz]')
plt.xlabel('f [Hz]')
ax.grid()
#plt.legend([r'$T_{65}$',r'$W_{70}$', r'$W_{65}$', r'$W_{30}$'])
#plt.legend([r'$T_{65m}$', r'$W_{65m}$', r'$W_{30m}$'])
plt.legend([r'$W_{30m}$', r'$W_{65m}$', r'$T_{65m}$'])
plt.text(1.4e-1, 3e-1, '7s')
plt.text(1/85.0, 2e-1, '85s')
plt.text(2e-3, 7e-1, '6.4min')
plt.text(5e-4, 7e-1, '20min')
#plt.show()
fig.savefig(fig_name)



#### ----  Other method ---- ####
# (https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python)

## p = 20*np.log10(np.abs(np.fft.rfft(Wseries65)))
## f = np.linspace(0, fs_W/2, len(p))
## plt.plot(f, p)

## plt.figure()
## xF = np.fft.fft(Wseries65)
## xF2 = np.fft.fft(Wseries30)
## xF3 = np.fft.fft(Tseries65)
## N = len(Wseries65)
## N2 = len(Wseries30)
## N3 = len(Tseries65)

## xF = xF[0:N/2]
## xF2 = xF2[0:N/2]
## xF3 = xF3[0:N/2]
## fr = np.linspace(0,fs_W/2,N/2)
## fr2 = np.linspace(0,fs_W/2,N2/2)
## fr3 = np.linspace(0,fs_T/2,N3/2)

## plt.ion()
## plt.plot(fr,abs(xF)**2)
## plt.plot(fr2,abs(xF2)**2)
## #plt.plot(fr3,abs(xF3)**2)
## plt.plot([.0025, .0025], [.28, .32], '--k')
## plt.plot([.00084, .00084], [.15, .25], '--k')
## plt.text(.0025, .33, '6.7min')
## plt.text(.00084, .26, '19.8min')
## plt.ylim([0, 0.32])

## plt.show()
