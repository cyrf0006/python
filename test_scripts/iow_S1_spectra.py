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
fig_name = 'S1_TWspectra_Mar02_3h.pdf'

cutoff_period_high = 1800.0  # desired cutoff in seconds (high-pass)
cutoff_period_low = 2.0*60*60 
##### ----------------------------------------- ####


# If you don't use squeeze_me = True, then Pandas doesn't like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True fixes that.
T_dict = loadmat('/media/cyrf0006/Fred32/IOW/Titp_S1_p4std_1sAve.mat',squeeze_me=True)
ADCP_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)

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

# Smooth timeseroes
T = T.resample('15s').mean()
W = W.resample('15s').mean()

# Cut timeseries
T = T.loc[(T.index >= XLIM[0]) & (T.index <= XLIM[1])]
W = W.loc[(W.index >= XLIM[0]) & (W.index <= XLIM[1])]

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


#### ---- Welch method ---- ####
# scipy.signal.welch(x, fs=1.0, window='hanning', nperseg=256, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)[source]
from scipy import signal
Tseries = np.array(T.iloc[:,35])
Wseries = np.array(W.iloc[:,17])
Wseries[np.isnan(Wseries)] = 0
fs = 1/15.0
f, Pxx_den = signal.welch(Tseries, fs, nperseg=256, detrend='linear')
f2,Pxx_den2 = signal.welch(Wseries, fs, nperseg=256) 

fig = plt.figure(2)
ax = plt.subplot(1,1,1)
plt.semilogy(f, Pxx_den)
plt.semilogy(f2, Pxx_den2)
#plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
ax.grid()
plt.legend(['T','w'])
plt.show()
fig.savefig(fig_name)


## ## ---- simple FFT ---- ##
## x = np.array(T.iloc[:,35])
## fs = 1
## N=1024
## fig = plt.figure()
## ax = fig.add_subplot(1,1,1)
## psd = np.fft.fft(x, N)
## f = np.fft.fftfreq(N, d=1)
## plt.semilogy(f, psd)
## plt.show()

## ## ---- Spectrogram ---- ##
## from scipy import signal
## x = np.array(T.iloc[:,35])
## fs = 1
## fig = plt.figure()
## ax = fig.add_subplot(1,1,1)
## f, t, Sxx = signal.spectrogram(x, fs)
## plt.pcolormesh(t, f, Sxx)
## plt.ylabel('Frequency [Hz]')
## plt.xlabel('Time [sec]')
## ax.set_yscale('log')
## plt.show()

## ## ---- Periodogram ---- ##
## # scipy.signal.periodogram(x, fs=1.0, window=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
## # scipy.signal.get_window(window, Nx, fftbins=True)[source]
## x = np.array(T.iloc[:,35])
## fs = 1
## f, Pxx_den = signal.periodogram(x, fs)
## f2, Pxx_den2 = signal.periodogram(x, fs, window=signal.get_window('blackman', 3600.0))
## plt.semilogy(f, Pxx_den)
## plt.ylim([1e-7, 1e2])
## plt.xlabel('frequency [Hz]')
## plt.ylabel('PSD [V**2/Hz]')
## plt.show()

