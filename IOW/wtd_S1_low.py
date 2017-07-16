import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib import colors, ticker, cm
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd

# Some info
fig_name = 'S1_sub_and_inertial.png'
XLIM = [pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00')]
time_zoom1 = pd.Timestamp('2010-03-02 03:00:00')
time_zoom2 = pd.Timestamp('2010-03-02 09:00:00')
mab_adcp = 1.58
mooring_depth = 83
adcpdir = 'up'

#### ---------- Wind data --------- ####
wave_dict = loadmat('SMHI_wave_33004_201002_201003.mat',squeeze_me=True)
meteo_dict = loadmat('SMHI_meteo_55570_201002_201003.mat',squeeze_me=True)
wind_dir = { k: meteo_dict[k] for k in ['wind_dir']}
wind_mag = { k: meteo_dict[k] for k in ['wind_mag']}
wave_height = { k: wave_dict[k] for k in ['waveheight_max','waveheight_signif']}
def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac

# convert Matlab time into list of python datetime objects and put in dictionary
wind_dir['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
wind_mag['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
wave_height['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]


# Wind data from R/V Alkor
#Alkor_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/AL351_DDL_wind.mat',squeeze_me=True, struct_as_record=False)
Alkor_dict = loadmat('./data/AL351_DDL_wind.mat',squeeze_me=True, struct_as_record=False)
Ualkor =  Alkor_dict['DDL'].wind.speed.data
yearday = Alkor_dict['DDL'].time.data+1
pdTimeAlkor = pd.to_datetime('2010') + pd.to_timedelta(yearday - 1, unit='d')
alkor = pd.DataFrame(Ualkor, index=pdTimeAlkor)
alkor = alkor.resample('30Min').mean()
# --------------------------------------# 

#### --------- NIOZ Tsensors --------- ####
#T_dict = loadmat('/media/cyrf0006/Fred32/IOW/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
T_dict = loadmat('./data/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])
pdTimeT = pd.date_range('2010-02-28 11:07:00', '2010-03-04 07:37:00', freq='10min')

# Dataframe
T = pd.DataFrame(T, index=pdTimeT, columns=zVecT)
# --------------------------------------#


### ---------ADCP data --------- ###
#ADCP_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)
ADCP_dict = loadmat('./data/IOWp01.mat',squeeze_me=True)

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
Ebin = E.resample('30min').mean()
Nbin = N.resample('30Min').mean()
Wbin = W.resample('30Min').mean()
fs_bin = 1/(60.0*30.0)       # sample rate, Hz1

# Remove "barotropic" currents (MATRIX IS TRANSPOSED!)
Ebin = Ebin.sub(Ebin.mean(axis=1), axis=0)
Nbin = Nbin.sub(Nbin.mean(axis=1), axis=0)
#### --------------------------------- ###

# filter timeseries
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
cutoff_low = 1/(2.0*60*60)

# Get the filter coefficients so we can check its frequency response.
## CAREFULL ON W HICH DIMENSION THE FILTER IS APPLIED!!!
#b, a = butter_lowpass(cutoff_high, fs_bin, order)
EE = np.nan_to_num(Ebin)
NN = np.nan_to_num(Nbin)
WW = np.nan_to_num(Wbin)
## Esub = butter_lowpass_filter(EE.T, cutoff_low, fs_bin, order)
## Nsub = butter_lowpass_filter(NN.T, cutoff_low, fs_bin, order)
## Wsub = butter_lowpass_filter(WW.T, cutoff_low, fs_bin, order)

cutoff_high = 1/(20.0*60*60)
Enear = butter_highpass_filter(EE.T, cutoff_high, fs_bin, order)
Nnear = butter_highpass_filter(NN.T, cutoff_high, fs_bin, order)
Wnear = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)

Esub = EE.T - Enear
Nsub = NN.T - Nnear


#### --------------------------------- ###



#### ----------- plot ------------ ####
days = dates.DayLocator()
hours6 = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%b %d')
hours1 = dates.HourLocator(interval=1)

rect_x = [time_zoom1, time_zoom2, time_zoom2, time_zoom1, time_zoom1]
rect_y = [83, 83, 1, 1, 83]
storm = pd.Timestamp('2010-03-01 6:00:00')
storm2 = pd.Timestamp('2010-03-01 12:00:00') # onset langmuir

fig = plt.figure(2)

# Wind
ax0 = plt.subplot2grid((9, 9), (0, 0), rowspan=1, colspan=8)
df = pd.DataFrame(wind_mag)
df = df.set_index('date_time')

ax0.plot(df.index, df.wind_mag, 'b')
#ax0.plot(alkor, 'b')
ax0.set_xlim(XLIM[0], XLIM[1])
ax0.tick_params(labelbottom='off')
ax0.xaxis.label.set_visible(False)
ax0.xaxis.set_major_locator(days)
ax0.xaxis.set_major_formatter(dfmt)
ax0.xaxis.set_minor_locator(hours6)
ax0.set_ylabel(r'$\rm \overline{u_w} (m s^{-1})$', color='b')
ax0.set_ylim(0,20)
ax0.tick_params('y', colors='b')
plt.grid()
#plt.plot([storm, storm], [0, 20], '--k')
plt.plot([storm2, storm2], [0, 20], '--k')
ax0.text(XLIM[1], 15, 'a  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')

ax01 = ax0.twinx()
df2 = pd.DataFrame(wave_height)
df2 = df2.set_index('date_time')
ax01.plot(df2.index, df2.waveheight_signif, 'r')
ax01.set_xlim(XLIM[0], XLIM[1])
ax01.tick_params(labelbottom='off')
ax01.xaxis.set_major_locator(days)
ax01.xaxis.set_major_formatter(dfmt)
ax01.xaxis.set_minor_locator(hours6)
ax01.set_ylabel(r'$\rm H_s (m)$', color='r')
ax01.set_ylim(0,5)
ax01.xaxis.label.set_visible(False)
ax01.tick_params('y', colors='r')
#ax0.plot([storm, storm], [18, 83], '--k')
plt.plot([storm2, storm2], [18, 83], '--k')


levels = np.linspace(-.15, .15, 9)
# E_near
ax1 = plt.subplot2grid((9, 9), (1, 0), rowspan=2, colspan=8)
c = plt.contourf(Ebin.index, Ebin.columns, Enear, levels, cmap=plt.cm.RdBu_r, extend="both")
ct = plt.contour(T.index, T.columns, T.T, [4,], colors='k', lw=0.5)
ax1.set_xlim(XLIM[0], XLIM[1])
ax1.set_ylim(18, 83)
ax1.tick_params(labelbottom='off')
ax1.set_ylabel(r'Depth (m)')
ax1.invert_yaxis()
ax1.xaxis.set_major_locator(days)
ax1.xaxis.set_major_formatter(dfmt)
ax1.xaxis.set_minor_locator(hours6)
ax1.text(XLIM[0], 75, r'  $\rm U_{near} ( m s^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
#ax1.plot([storm, storm], [18, 83], '--k')
ax1.plot([storm2, storm2], [18, 83], '--k')
ax1.text(XLIM[1], 25, 'b  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')

#N_near
ax2 = plt.subplot2grid((9, 9), (3, 0), rowspan=2, colspan=8)
c = plt.contourf(Nbin.index, Nbin.columns, Nnear, levels, cmap=plt.cm.RdBu_r, extend="both")
ct = plt.contour(T.index, T.columns, T.T, [4,], colors='k', lw=0.5)
ax2.set_xlim(XLIM[0], XLIM[1])
ax2.set_ylim(18, 83)
ax2.tick_params(labelbottom='off')
ax2.set_ylabel(r'Depth (m)')
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(days)
ax2.xaxis.set_major_formatter(dfmt)
ax2.xaxis.set_minor_locator(hours6)
ax2.text(XLIM[0], 75, r'  $\rm V_{near} ( m s^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
#ax2.plot([storm, storm], [18, 83], '--k')
ax2.plot([storm2, storm2], [18, 83], '--k')
ax2.text(XLIM[1], 25, 'c  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')

## cax = plt.axes([0.9,0.5,0.01,0.35]) # One colorbar for bpoth plots
## plt.colorbar(c, cax=cax, format='%.2f')


# E_sub
ax3 = plt.subplot2grid((9, 9), (5, 0), rowspan=2, colspan=8)
c = plt.contourf(Ebin.index, Ebin.columns, Esub, levels, cmap=plt.cm.RdBu_r, extend="both")
ct = plt.contour(T.index, T.columns, T.T, [4,], colors='k', lw=0.5)
ax3.set_xlim(XLIM[0], XLIM[1])
ax3.set_ylim(18, 83)
ax3.tick_params(labelbottom='off')
ax3.set_ylabel(r'Depth (m)')
ax3.invert_yaxis()
ax3.xaxis.set_major_locator(days)
ax3.xaxis.set_major_formatter(dfmt)
ax3.xaxis.set_minor_locator(hours6)
ax3.text(XLIM[0], 75, r'  $\rm U_{sub} ( m s^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
#ax3.plot([storm, storm], [18, 83], '--k')
ax3.plot([storm2, storm2], [18, 83], '--k')
ax3.text(XLIM[1], 25, 'd  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')

#N_sub
ax4 = plt.subplot2grid((9, 9), (7, 0), rowspan=2, colspan=8)
c = plt.contourf(Nbin.index, Nbin.columns, Nsub, levels, cmap=plt.cm.RdBu_r, extend="both")
ct = plt.contour(T.index, T.columns, T.T, [4,], colors='k', lw=0.5)
ax4.set_xlim(XLIM[0], XLIM[1])
ax4.set_ylim(18, 83)
ax4.tick_params(labelbottom='on')
ax4.set_ylabel(r'Depth (m)')
ax4.invert_yaxis()
ax4.xaxis.set_major_locator(days)
ax4.xaxis.set_major_formatter(dfmt)
ax4.xaxis.set_minor_locator(hours6)
ax4.text(XLIM[0], 75, r'  $\rm V_{sub} ( m s^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
#ax4.plot([storm, storm], [18, 83], '--k')
ax4.plot([storm2, storm2], [18, 83], '--k')
ax4.text(XLIM[1], 25, 'e  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')


cax = plt.axes([0.9,0.1,0.02,0.7]) # One colorbar for all plots
plt.colorbar(c, cax=cax, format='%.2f')


fig.set_size_inches(w=8, h=8)
fig.set_dpi(300)
fig.tight_layout()
plt.subplots_adjust(hspace=0.35)
fig.savefig(fig_name)
#plt.show()
