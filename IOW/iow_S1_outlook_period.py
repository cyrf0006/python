import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd
#import seawater as sw
import gsw

# Some infos:
mab_adcp = 1.58
mooring_depth = 83.0
adcpdir = 'up'

#XLIM = [pd.Timestamp('2010-03-01 15:30:00'), pd.Timestamp('2010-03-01 16:30:00')]
XLIM = [pd.Timestamp('2010-03-01 06:00:00'), pd.Timestamp('2010-03-02 12:00:00')]
fig_name = 'S1_outlook_period.png'

#### ---------- Wind data --------- ####
meteo_dict = loadmat('SMHI_meteo_55570_201002_201003.mat',squeeze_me=True)
wind_dir = { k: meteo_dict[k] for k in ['wind_dir']}
wind_mag = { k: meteo_dict[k] for k in ['wind_mag']}
def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac

# convert Matlab time into list of python datetime objects and put in dictionary
wind_dir['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
wind_mag['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]

# Wind data from R/V Alkor
#Alkor_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/AL351_DDL_wind.mat',squeeze_me=True, struct_as_record=False)
Alkor_dict = loadmat('./data/AL351_DDL_wind.mat',squeeze_me=True, struct_as_record=False)
Ualkor =  Alkor_dict['DDL'].wind.speed.data
yearday = Alkor_dict['DDL'].time.data+1
pdTimeAlkor = pd.to_datetime('2010') + pd.to_timedelta(yearday - 1, unit='d')
alkor = pd.DataFrame(Ualkor, index=pdTimeAlkor)
alkor = alkor.resample('30Min').mean()
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
Ebin = E.resample('1min').mean()
Nbin = N.resample('1Min').mean()
Wbin = W.resample('1Min').mean()
fs_bin = 1/(60.0)       # sample rate, Hz1

# Remove "barotropic" currents (MATRIX IS TRANSPOSED!)
Ebin = Ebin.sub(Ebin.mean(axis=1), axis=0)
Nbin = Nbin.sub(Nbin.mean(axis=1), axis=0)
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

def testGauss(x, y, s, npts):
	b = gaussian(39, 10)
	ga = filters.convolve1d(y, b/b.sum())
	plt.plot(x, ga)
	print "gaerr", ssqe(ga, s, npts)
	return ga

# Filter requirements.
order = 8
cutoff_low = 1/(2.0*60*60)
cutoff_high = 1/(1800.0)  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
## CAREFULL ON W HICH DIMENSION THE FILTER IS APPLIED!!!
#b, a = butter_lowpass(cutoff_high, fs_bin, order)
EE = np.nan_to_num(Ebin)
NN = np.nan_to_num(Nbin)
WW = np.nan_to_num(Wbin)
Elow = butter_lowpass_filter(EE.T, cutoff_low, fs_bin, order)
Nlow = butter_lowpass_filter(NN.T, cutoff_low, fs_bin, order)
Wlow = butter_lowpass_filter(WW.T, cutoff_low, fs_bin, order)
Ehigh = butter_highpass_filter(EE.T, cutoff_high, fs_bin, order)
Nhigh = butter_highpass_filter(NN.T, cutoff_high, fs_bin, order)
Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)
#### --------------------------------- ###

#### --------- NIOZ Tsensors --------- ####
T_dict = loadmat('./data/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])
pdTimeT = pd.date_range('2010-02-28 11:07:00', '2010-03-04 07:37:00', freq='10min')

# Dataframe
T = pd.DataFrame(T, index=pdTimeT, columns=zVecT)
# --------------------------------------# 

#### ---------- MSS cast --------- ####
# CTD casts before storm (pd is powerful!))
lat = 55
lon = 16
Pbin = np.arange(0.5, 85, 1)
MSS_S1_1_dict = loadmat('./data/MSS_DATA/S1_1.mat',squeeze_me=True, struct_as_record=False)
Zmss =  MSS_S1_1_dict['CTD'][2].P
Tmss =  MSS_S1_1_dict['CTD'][2].T
Smss =  MSS_S1_1_dict['CTD'][2].S
digitized = np.digitize(Zmss, Pbin) #<- this is awesome!
TTbin = np.array([Tmss[digitized == i].mean() for i in range(0, len(Pbin))])
SSbin = np.array([Smss[digitized == i].mean() for i in range(0, len(Pbin))])
SA_MSS_01 = gsw.SA_from_SP_Baltic(SSbin,lon,lat)
CT_MSS_01 = gsw.CT_from_t(SA_MSS_01,TTbin,Pbin)
SIG0_MSS_01 = gsw.sigma0(SA_MSS_01,CT_MSS_01)
N2_MSS_01 = gsw.Nsquared(SA_MSS_01,CT_MSS_01,Pbin,lat)
import SW_extras as swe
N2_01 = N2_MSS_01[0]
zN2_01 = N2_MSS_01[1]
N2_period_01 = 60.0/swe.cph(N2_01)

MSS_S1_2_dict = loadmat('./data/MSS_DATA/S1_2.mat',squeeze_me=True, struct_as_record=False)
Zmss =  MSS_S1_2_dict['CTD'][2].P
Tmss =  MSS_S1_2_dict['CTD'][2].T
Smss =  MSS_S1_2_dict['CTD'][2].S
digitized = np.digitize(Zmss, Pbin) #<- this is awesome!
TTbin = np.array([Tmss[digitized == i].mean() for i in range(0, len(Pbin))])
SSbin = np.array([Smss[digitized == i].mean() for i in range(0, len(Pbin))])
SA_MSS_02 = gsw.SA_from_SP_Baltic(SSbin,lon,lat)
CT_MSS_02 = gsw.CT_from_t(SA_MSS_02,TTbin,Pbin)
SIG0_MSS_02 = gsw.sigma0(SA_MSS_02,CT_MSS_02)
N2_MSS_02 = gsw.Nsquared(SA_MSS_02,CT_MSS_02,Pbin,lat)
N2_02 = N2_MSS_02[0]
zN2_02 = N2_MSS_02[1]
N2_period_02 = 60.0/swe.cph(N2_02)

MSS_S1_5_dict = loadmat('./data/MSS_DATA/S1_5.mat',squeeze_me=True, struct_as_record=False)
Zmss =  MSS_S1_5_dict['CTD'][-1].P
Tmss =  MSS_S1_5_dict['CTD'][-1].T
Smss =  MSS_S1_5_dict['CTD'][-1].S
digitized = np.digitize(Zmss, Pbin) #<- this is awesome!
TTbin = np.array([Tmss[digitized == i].mean() for i in range(0, len(Pbin))])
SSbin = np.array([Smss[digitized == i].mean() for i in range(0, len(Pbin))])
SA_MSS_05 = gsw.SA_from_SP_Baltic(SSbin,lon,lat)
CT_MSS_05 = gsw.CT_from_t(SA_MSS_05,TTbin,Pbin)
SIG0_MSS_05 = gsw.sigma0(SA_MSS_05,CT_MSS_05)
N2_MSS_05 = gsw.Nsquared(SA_MSS_05,CT_MSS_05,Pbin,lat)
N2_05 = N2_MSS_05[0]
zN2_05 = N2_MSS_05[1]
N2_period_05 = 60.0/swe.cph(N2_05)
## # ------------------------------------# 


#### --------- plots ---------- ####
days = dates.DayLocator()
hours = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%d')

fig = plt.figure(2)
ax0 = plt.subplot2grid((4, 9), (0, 2), rowspan=1, colspan=6)
ax0.plot(alkor) # WIND
ax0.set_xlim(XLIM[0], XLIM[1])
ax0.tick_params(labelbottom='off')
ax0.xaxis.label.set_visible(False)
ax0.xaxis.set_major_locator(days)
ax0.xaxis.set_major_formatter(dfmt)
ax0.xaxis.set_minor_locator(hours)
ax0.set_ylabel(r'$\overline{U} (m s^{-1})$')
ax0.set_ylim(0,20)
plt.grid()


ax1 = plt.subplot2grid((4, 9), (1, 0), rowspan=3, colspan=2)
# N2_period
#ax1.semilogx(N2_period_01, zN2_01)
ax1.semilogx(N2_period_02, zN2_02)
#ax1.semilogx(N2_period_05, zN2_05)
ax1.set_ylabel(r'Depth (m)')
ax1.invert_yaxis()
ax1.set_ylim(1, 83)
#ax1.set_xlim(0, 30)
ax1.invert_yaxis()
ax1.grid()
ax1.legend(['28/02', '02/03', '04/03'])
ax1.set_xlabel(r'$\rm T_N (min)$')
ax1.set_xticks([1e0, 1e1, 1e2])

#W_high
ax2 = plt.subplot2grid((4, 9), (1, 2), rowspan=3, colspan=6)
levels = [-.004, -.002, -.001, .001, .002, .004]
c = plt.contourf(Wbin.index, Wbin.columns, Whigh, levels, cmap=plt.cm.RdBu_r, extend="both")
ct = plt.contour(T.index, T.columns, T.T, [4,], colors='k', lw=0.5)
cax = plt.axes([0.9,0.2,0.02,0.5])
plt.colorbar(c, cax=cax)
ax2.set_xlim(XLIM[0], XLIM[1])
ax2.set_ylim(0, 83)
ax2.tick_params(labelbottom='on')
ax2.tick_params(labelleft='off')
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(days)
ax2.xaxis.set_major_formatter(dfmt)
ax2.xaxis.set_minor_locator(hours)
ax2.text(XLIM[0], 25, r"  $\rm w' ( m s^{-1})$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='w')

# add rectangle
import matplotlib.dates as mdates
start = mdates.date2num(XLIM[0])
end = mdates.date2num(XLIM[1])
width = end - start
rect_x = [start, end, end, start, start]
rect_y = [0,0,19,19,0]
rect = zip(rect_x, rect_y)
Rgon = plt.Polygon(rect,color=np.multiply([1.0,1.0,1.0],.7), alpha=0.0, hatch='/')
#Rgon = plt.Polygon(rect,color='none', edgecolor='red', facecolor="red", hatch='/')
ax2.add_patch(Rgon)

## # add zoomed rectangle
## start = mdates.date2num(pd.Timestamp('2010-03-02 03:00:00'))
## end = mdates.date2num(pd.Timestamp('2010-03-02 09:00:00'))
## rect_x = [start, end, end, start, start]
## rect_y = [83, 83, 53, 53, 83]
## rect = zip(rect_x, rect_y)
## Rgon2 = plt.Polygon(rect, color='k', ls='--', lw=1, alpha=1, fill=False)
## #Rgon = plt.Polygon(rect,color='none', edgecolor='red', facecolor="red", hatch='/')
## ax2.add_patch(Rgon2)


fig.set_size_inches(w=8, h=5)
fig.set_dpi(150)
fig.tight_layout()
fig.savefig(fig_name)
plt.show()





