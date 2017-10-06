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
fig_name = 'S1_highfreq.png'
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
Elow = butter_lowpass_filter(EE.T, cutoff_low, fs_bin, order)
Nlow = butter_lowpass_filter(NN.T, cutoff_low, fs_bin, order)
Wlow = butter_lowpass_filter(WW.T, cutoff_low, fs_bin, order)
Ehigh = butter_highpass_filter(EE.T, cutoff_high, fs_bin, order)
Nhigh = butter_highpass_filter(NN.T, cutoff_high, fs_bin, order)
Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)
#### --------------------------------- ###

#### ----- shear calculation --------- ####
def shear(z, u, v=None):
    """
    Shear calculation
    z: numpy array of length n
    u,v: matrices of size nxm or mxn

    shr[0] = shear (S[s^-1])
    shr[1] = corresponding nx1 depth vector
    """
    
    if v is None: # fill with zeros if no 'v'
        v = np.zeros(u.shape)
                     
    if u.shape[0] != z.size:
        u = u.T # transpose if needed
        v = v.T
    
    m = z.size
    iup = np.arange(0, m - 1)
    ilo = np.arange(1, m)
    z_ave = (z[iup] + z[ilo]) / 2.

    if u.size != z.size: # multi-dimensional   
        z = np.tile(np.matrix(z).T, (1, u.shape[1]))
        
    du = np.diff(u, axis=0)
    dv = np.diff(u, axis=0)
    dz = np.diff(z, axis=0)
    shr = np.abs(du/dz) + np.abs(dv/dz)
    return shr, z_ave

## def shear(z, u, v=0):
##     """
##     Shear calculation
##     z: numpy array of length n
##     u,v: matrices of size nxm or mxn

##     shr[0] = shear (S[s^-1])
##     shr[1] = corresponding nx1 depth vector
##     """

##     if u.shape[0] != z.size:
##         u = u.T # transpose if needed
##         v = v.T

##     m = z.size
##     iup = np.arange(0, m - 1)
##     ilo = np.arange(1, m)
##     z_ave = (z[iup] + z[ilo]) / 2.

##     z = np.tile(np.matrix(z).T, (1, u.shape[1]))
##     du = np.diff(u, axis=0)   
##     dv = np.diff(v, axis=0)   
##     dz = np.diff(z, axis=0)

##     shr = np.abs(du/dz) + np.abs(dv/dz)
##     return shr, z_ave

UU = np.array(EE) # total shear
VV = np.array(NN)
## UU = np.array(Ehigh) # low freq. shear
## VV = np.array(Nhigh)
Z = np.array(Ebin.columns)

shr = shear(Z, UU, VV)
#shr = shear(Z, Elow, Nlow)
#### --------------------------------- ###


#### ----------- plot ------------ ####
days = dates.DayLocator()
hours6 = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%b %d')
hours1 = dates.HourLocator(interval=1)

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


# W_high
ax1 = plt.subplot2grid((9, 9), (1, 0), rowspan=2, colspan=8)
levels = np.linspace(-.005, .005, 11)
c = plt.contourf(Wbin.index, Wbin.columns, Whigh, levels, cmap=plt.cm.RdBu_r, extend="both")
ct = plt.contour(T.index, T.columns, T.T, [4,], colors='k', lw=0.5)
cax = plt.axes([0.89,0.7,0.01,0.15])
plt.colorbar(c, cax=cax, ticks=[-.005, -.003, -.001, .001, .003, .005])
ax1.set_xlim(XLIM[0], XLIM[1])
ax1.set_ylim(0, 88)
ax1.tick_params(labelbottom='off')
ax1.set_ylabel(r'Depth (m)')
ax1.invert_yaxis()
ax1.xaxis.set_major_locator(days)
ax1.xaxis.set_major_formatter(dfmt)
ax1.xaxis.set_minor_locator(hours6)
ax1.text(XLIM[0], 10, r"  $\rm W' ( m s^{-1})$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
#ax1.plot([storm, storm], [18, 83], '--k')
#ax1.plot([storm2, storm2], [0, 88], '--k')
ax1.text(XLIM[1], 10, 'b  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')
# add patch
zoomy = [19,79]
rect_y = [zoomy[0], zoomy[0], zoomy[1], zoomy[1], zoomy[0]]
rect = zip(rect_x, rect_y)
Rgon = plt.Polygon(rect,color='gray', alpha=0.4)
ax1.add_patch(Rgon)


# W_high average
ax2 = plt.subplot2grid((9, 9), (3, 0), rowspan=2, colspan=8)
idx = np.where(np.logical_and(Z>=20, Z<=40))
w_ave = np.mean(Whigh[idx],axis=0)
w_ave = w_ave**2

# runmean
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N
N=30
w_filt = running_mean(w_ave, N)
time_wfilt = Ebin.index[N/2:-N/2+1]

plt.plot(Ebin.index, w_ave*10000)
plt.plot(time_wfilt, w_filt*10000)
ax2.set_xlim(XLIM[0], XLIM[1])
ax2.tick_params(labelbottom='off')
ax2.xaxis.set_major_locator(days)
ax2.xaxis.set_major_formatter(dfmt)
ax2.xaxis.set_minor_locator(hours6)
ax2.set_ylabel(r'$(m^2 s^{-2})\times 10^4$')
ax2.set_ylim(0, 1)
ax2.text(XLIM[0], 0.9, r"  $\rm <W'>_{(20-40m)}^2$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
plt.grid()
ax2.text(XLIM[1], .9, 'c  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')
#ax2.plot([storm, storm], [18, 83], '--k')
#ax2.plot([storm2, storm2], [0, 88], '--k')

# S2
ax3 = plt.subplot2grid((9, 9), (5, 0), rowspan=2, colspan=8)
levels = np.linspace(-1.8, -0.4, 8)
#levels = np.linspace(-1.6, -0.8, 5)
c = plt.contourf(Ebin.index, shr[1], np.log10(shr[0]), levels, cmap=plt.cm.RdBu_r, extend="both", color='k')
ct = plt.contour(T.index, T.columns, T.T, [4,], colors='k', lw=0.5)
#c = plt.contourf(Ebin.index, shr[1], np.log10(shr[0]), levels, cmap=plt.cm.Reds, extend="both")
cax = plt.axes([0.89,0.27,0.01,0.15])
plt.colorbar(c, cax=cax)
ax3.set_xlim(XLIM[0], XLIM[1])
ax3.set_ylim(0, 88)
ax3.tick_params(labelbottom='off')
ax3.set_ylabel(r'Depth (m)')
ax3.invert_yaxis()
ax3.xaxis.set_major_locator(days)
ax3.xaxis.set_major_formatter(dfmt)
ax3.xaxis.set_minor_locator(hours6)
ax3.text(XLIM[0], 10, r'  $\rm log_{10}(S / s^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
#ax3.plot([storm, storm], [18, 83], '--k')
#ax3.plot([storm2, storm2], [0, 88], '--k')
ax3.text(XLIM[1], 10, 'd  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')

# epsilon
EPS = pd.read_pickle('MSS_S1.pkl')

# Casts identifyer
casts = np.ones(len(EPS.index))
castsID = pd.DataFrame(casts, index=EPS.index) 

# resample data 
EPS = EPS.resample('60Min').mean()

ax4 = plt.subplot2grid((9, 9), (7, 0), rowspan=2, colspan=8)
levels = np.linspace(-9.5,-6.5, 20)
c = plt.contourf(EPS.index, EPS.columns, np.log10(np.transpose(EPS)), levels, cmap=plt.cm.RdBu_r, extend="both")
cax = plt.axes([0.89,0.06,0.01,0.15])
#plt.colorbar(c, cax=cax, ticks=[-10, -9, -8, -7, -6])
plt.colorbar(c, cax=cax, ticks=[np.linspace(-9.5, -6.5, 7)])
ax4.text(pd.Timestamp('2010-03-01 00:00:00'), 10, r'$\rm log_{10}(\epsilon / W Kg^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
ax4.set_ylabel(r'Depth (m)')
ax4.set_xlim(XLIM)
ax4.xaxis.set_major_locator(days)
ax4.xaxis.set_major_formatter(dfmt)
ax4.xaxis.set_minor_locator(hours6)
ax4.set_ylim(0, 88)
ax4.invert_yaxis()
#ax4.plot([storm2, storm2], [0, 88], '--k')

# Mark casts
ax4.plot(castsID, '|', color=[.3,.3,.3])
ax4.text(XLIM[1], 15, 'e  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='w')



fig.set_size_inches(w=8, h=8)
fig.set_dpi(300)
fig.tight_layout()
plt.subplots_adjust(hspace=0.35)
fig.savefig(fig_name)
#plt.show()
