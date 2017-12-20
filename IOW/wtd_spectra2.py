import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd
from scipy import signal
from matplotlib.patches import Ellipse

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

#XLIM = [pd.Timestamp('2010-03-01 21:00:00'), pd.Timestamp('2010-03-02 01:00:00')]
XLIM = [pd.Timestamp('2010-03-01 17:00:00'), pd.Timestamp('2010-03-01 21:00:00')]
XLIM = [pd.Timestamp('2010-03-02 04:00:00'), pd.Timestamp('2010-03-02 08:00:00')]

#XLIM = [pd.Timestamp('2010-03-02 01:00:00'), pd.Timestamp('2010-03-02 05:00:00')]


fig_name = 'S1_spectral_contours.png'

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
W = W.resample('60s').mean()
## fs_W = 1/30.0
## fs_T = 1/30.0

fs_W = 1/60.0
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

Wspectra = []
for idx, z in enumerate(W.columns):

    Wseries = np.array(W.iloc[:,idx])
    Wseries[np.isnan(Wseries)] = 0
    
    #### ---- Welch method ---- ####
    seriesize = np.size(Wseries)
    N = seriesize/1
    f, Pxx = signal.welch(Wseries, fs_W, nperseg=N, noverlap=N/2) 

    Wspectra.append(Pxx)


levels = np.arange(-2.5, -.9, .1)
S = np.array(Wspectra)
S[0:1,:]=np.nan
#S[19:21,:]=np.nan


fig = plt.figure(2)
cf = plt.contourf(f[1:-1], zVec, np.log10(S[:, 1:-1]), levels, extend='both')   
plt.xscale('log')
plt.gca().invert_yaxis()
plt.xlabel('f [Hz]')
plt.ylabel('Depth [m]')

xmin_IW1 = 2.5e-3
xmin_IW2 = 8.5e-4
xmin_IW3 = 5.5e-4
plt.gca().annotate('~30 min.',
            xy=(xmin_IW3, 26), xycoords='data',
            xytext=(-120, -75), textcoords='offset points',
            size=16, color='w',
            # bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="simple",
                            fc="1", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

plt.gca().annotate('~20 min.',
            xy=(xmin_IW2, 35), xycoords='data',
            xytext=(-120, -75), textcoords='offset points',
            size=16, color='w',
            # bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="simple",
                            fc="1", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

plt.gca().annotate('~6.4 min.',
            xy=(xmin_IW1, 66), xycoords='data',
            xytext=(-170, -35), textcoords='offset points',
            size=16, color='w',
            # bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(arrowstyle="simple",
                            fc="1", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))


cax = plt.axes([0.91,0.12,0.01,0.68])
plt.colorbar(cf, cax=cax)
cax.text(-1, 1.12, r' [$\rm \frac{(ms^{-1})^2}{Hz}$]', horizontalalignment='left', verticalalignment='center', fontsize=11, color='k')


fig.set_size_inches(w=7, h=4)
#fig.tight_layout()
fig.set_dpi(300)
fig.savefig(fig_name)

