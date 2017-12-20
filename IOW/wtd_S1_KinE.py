import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib import colors, ticker, cm
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd
import gsw
import SW_extras as swe

# Some info
fig_name = 'S1_KE.png'
#XLIM = [pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00')]
XLIM = [pd.Timestamp('2010-03-01 06:00:00'), pd.Timestamp('2010-03-02 12:00:00')]
time_zoom1 = pd.Timestamp('2010-03-02 03:00:00')
time_zoom2 = pd.Timestamp('2010-03-02 09:00:00')
mab_adcp = 1.58
mooring_depth = 83
adcpdir = 'up'
lat = 55
lon = 16
Pbin = np.arange(0.5, 85, 1)

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

#### ---------- MSS cast --------- ####
lat = 55
lon = 16
MSS_S1_2_dict = loadmat('./data/MSS_DATA/S1_2.mat',squeeze_me=True, struct_as_record=False)
Zmss =  MSS_S1_2_dict['CTD'][0].P
Tmss =  MSS_S1_2_dict['CTD'][0].T
Smss =  MSS_S1_2_dict['CTD'][0].S
timemss2 = pd.Timestamp(MSS_S1_2_dict['STA'][0].date)
digitized = np.digitize(Zmss, Pbin) #<- this is awesome!
TTbin = np.array([Tmss[digitized == i].mean() for i in range(0, len(Pbin))])
SSbin = np.array([Smss[digitized == i].mean() for i in range(0, len(Pbin))])
SA_MSS_02 = gsw.SA_from_SP_Baltic(SSbin,lon,lat)
CT_MSS_02 = gsw.CT_from_t(SA_MSS_02,TTbin,Pbin)
SIG0_MSS_02 = gsw.sigma0(SA_MSS_02,CT_MSS_02)
idx_sort = SIG0_MSS_02.argsort() # sort density
SA_MSS_02 = SA_MSS_02[idx_sort]
CT_MSS_02 = CT_MSS_02[idx_sort]
SIG0_MSS_02 = SIG0_MSS_02[idx_sort]
N2_MSS_02 = gsw.Nsquared(SA_MSS_02,CT_MSS_02,Pbin,lat)
N2_02 = N2_MSS_02[0]
zN2_02 = N2_MSS_02[1]
N2_period_02 = 60.0/swe.cph(N2_02)
# ---------------------------------------- #

### ---------ADCP data --------- ###
#ADCP_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)
ADCP_dict = loadmat('./data/IOWp01.mat',squeeze_me=True)

ADCP = { k: ADCP_dict[k] for k in ['SerNmmpersec', 'SerEmmpersec', 'SerVmmpersec']}

# To Pandas DataFrame
E = ADCP_dict['SerEmmpersec']/1000.0 # now in cm/s
N = ADCP_dict['SerNmmpersec']/1000.0
W = ADCP_dict['SerVmmpersec']/1000.0
echo1 = ADCP_dict['SerEA1cnt'] # now in cm/s
echo2 = ADCP_dict['SerEA1cnt'] # now in cm/s
echo3 = ADCP_dict['SerEA1cnt'] # now in cm/s
echo4 = ADCP_dict['SerEA1cnt'] # now in cm/s


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
Ebin = E.resample('2min').mean()
Nbin = N.resample('2Min').mean()
Wbin = W.resample('2Min').mean()
fs_bin = 1/(60.0*2.0)       # sample rate, Hz1

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
cutoff_high = 1/(1800.0)  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
## CAREFULL ON W HICH DIMENSION THE FILTER IS APPLIED!!!
EE = np.nan_to_num(Ebin)
NN = np.nan_to_num(Nbin)
WW = np.nan_to_num(Wbin)

# Flag some data (because mooring floatation)
plt.figure()
plt.plot(EE.mean(axis=0), zVec)
EE[:,0:5]=np.nan
EE[:,25:30]=np.nan
EE[:,-1]=np.nan
NN[:,0:5]=np.nan
NN[:,25:30]=np.nan
NN[:,-1]=np.nan
WW[:,0:5]=np.nan
WW[:,25:30]=np.nan
WW[:,-1]=np.nan

# Check flaggin results
plt.plot(EE.mean(axis=0), zVec, 'r')
plt.title('check effect of flagging')
#plt.show() # <--------- Uncomment to check result

# Filtering
Elow = butter_lowpass_filter(EE.T, cutoff_low, fs_bin, order)
Nlow = butter_lowpass_filter(NN.T, cutoff_low, fs_bin, order)
Wlow = butter_lowpass_filter(WW.T, cutoff_low, fs_bin, order)
Ehigh = butter_highpass_filter(EE.T, cutoff_high, fs_bin, order)
Nhigh = butter_highpass_filter(NN.T, cutoff_high, fs_bin, order)
Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)
# --------------------------------- #

#### -------- Calculate Kinetic Energy --------- ####
K = .5*1010*(Ehigh**2 + Nhigh**2 + Whigh**2)
#K = .5*1010*(Ehigh**2 + Nhigh**2)
#K = .5*1010*(Whigh**2)
K_df = pd.DataFrame(K.T, index=Ebin.index, columns=zVec)
K_df = K_df.resample('30min').mean()


#### ----------- plot ------------ ####
days = dates.DayLocator()
hours6 = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%H %M')
hours1 = dates.HourLocator(interval=1)

storm = pd.Timestamp('2010-03-01 6:00:00')
storm2 = pd.Timestamp('2010-03-01 12:00:00') # onset langmuir

fig = plt.figure(2)

# Wind
ax0 = plt.subplot2grid((4, 9), (0, 0), rowspan=1, colspan=8)
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
#plt.plot([storm2, storm2], [0, 20], '--k')
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
#plt.plot([storm2, storm2], [18, 83], '--k')
# add patch
import matplotlib.dates as mdates
zoomx = [pd.Timestamp('2010-03-01 11:00:00'), pd.Timestamp('2010-03-02 1:45:00')]
start = mdates.date2num(zoomx[0])
end = mdates.date2num(zoomx[1])
rect_x = [start, end, end, start, start]
zoomy = [0, 20]
rect_y = [zoomy[0], zoomy[0], zoomy[1], zoomy[1], zoomy[0]]
rect = zip(rect_x, rect_y)
Rgon = plt.Polygon(rect,color='gray', alpha=0.3)
ax01.add_patch(Rgon)

# KE
ax3 = plt.subplot2grid((4, 9), (1, 0), rowspan=3, colspan=8)
#levels = np.linspace(-2.4, 1, 21)
#levels = np.linspace(-2.4, 1, 18)
levels = np.linspace(-2.4, .8, 17)
#levels = np.linspace(.3, 4, 61)
#levels = 21
c = plt.contourf(K_df.index, K_df.columns, np.log10(K_df.T), levels, extend="both", color='k')
#c = plt.contourf(K_df.index, K_df.columns, K_df.T, levels, extend="both", color='k')
ct = plt.contour(T.index, T.columns, T.T, [4,], colors='k', lw=0.5)
#c = plt.contourf(Ebin.index, shr[1], np.log10(shr[0]), levels, cmap=plt.cm.Reds, extend="both")
cax = plt.axes([0.89,0.1,0.02,0.6])
plt.colorbar(c, cax=cax)
ax3.set_xlim(XLIM[0], XLIM[1])
ax3.set_ylim(19, 70)
ax3.tick_params(labelbottom='on')
ax3.set_ylabel(r'Depth (m)')
ax3.invert_yaxis()
ax3.xaxis.set_major_locator(hours6)
ax3.xaxis.set_major_formatter(dfmt)
ax3.xaxis.set_minor_locator(hours1)
ax3.text(XLIM[0], 23, r'  $\rm K_{IW}\,(J\,Kg^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='w')
ax3.text(pd.Timestamp('2010-03-01 12:00:00'), 73, '1 March', horizontalalignment='center', verticalalignment='center', fontsize=12, color='k')
ax3.text(pd.Timestamp('2010-03-02 06:00:00'), 73, '2 March', horizontalalignment='center', verticalalignment='center', fontsize=12, color='k')
ax3.text(XLIM[1], 22, 'b  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='w')
ax3.grid('on')


## Annotations
ann_text_x = pd.Timestamp('2010-03-02 03:00:00')
ann1_x = pd.Timestamp('2010-03-01 19:00:00')
ann2_x = pd.Timestamp('2010-03-01 21:00:00')
ann3_x = pd.Timestamp('2010-03-01 22:00:00')
ann4_x = pd.Timestamp('2010-03-01 12:00:00')
ann1_y = 25
ann2_y = 40
ann3_y = 55
ann_text4_x = pd.Timestamp('2010-03-01 07:00:00')
ann4_y = 60

ax3.annotate('Deepening\n IW energy', xy=(ann1_x, ann1_y), xytext=(ann_text_x, 25),
            arrowprops=dict(facecolor='black', width=1, shrink=0.05), fontsize=14
            )
ax3.annotate('', xy=(ann2_x, ann2_y), xytext=(ann_text_x, 25),
            arrowprops=dict(facecolor='black', width=1, shrink=0.05),
            )
ax3.annotate('', xy=(ann3_x, ann3_y), xytext=(ann_text_x, 25),
            arrowprops=dict(facecolor='black', width=1, shrink=0.05),
            )
ax3.annotate('Large IW (Fig.6f)', xy=(ann4_x, ann4_y), xytext=(ann_text4_x, 52),
            arrowprops=dict(facecolor='black', width=1, shrink=0.05), fontsize=14
            )





fig.set_size_inches(w=8, h=8)
fig.set_dpi(300)
fig.tight_layout()
plt.subplots_adjust(hspace=0.35)
fig.savefig(fig_name)
#plt.show()

