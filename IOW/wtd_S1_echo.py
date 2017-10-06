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
XLIM = [pd.Timestamp('2010-03-01 6:00:00'), pd.Timestamp('2010-03-02 23:00:00')]
XLIM = [pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-03 23:00:00')]
XLIM_zoom = [pd.Timestamp('2010-03-01 21:00:00'), pd.Timestamp('2010-03-02 05:00:00')]
YLIM = [0, 80]
YLIM2 = [18, 60]

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

### ---------ADCP data --------- ###
#ADCP_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/adcp/IOWp01.mat',squeeze_me=True)
ADCP_dict = loadmat('./data/IOWp01.mat',squeeze_me=True)

ADCP = { k: ADCP_dict[k] for k in ['SerNmmpersec', 'SerEmmpersec', 'SerVmmpersec']}

# To Pandas DataFrame
E = ADCP_dict['SerEmmpersec']/1000.0 # now in cm/s
N = ADCP_dict['SerNmmpersec']/1000.0
W = ADCP_dict['SerVmmpersec']/1000.0
echo1 = ADCP_dict['SerEA1cnt']
echo2 = ADCP_dict['SerEA2cnt']
echo3 = ADCP_dict['SerEA3cnt']
echo4 = ADCP_dict['SerEA4cnt']


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
EA1 = pd.DataFrame(echo1, index=pandaTime, columns=zVec)
EA2 = pd.DataFrame(echo2, index=pandaTime, columns=zVec)
EA3 = pd.DataFrame(echo3, index=pandaTime, columns=zVec)
EA4 = pd.DataFrame(echo4, index=pandaTime, columns=zVec)

# reduce timeseries
E = E[(E.index > XLIM[0]) & (E.index <= XLIM[1])]
N = N[(N.index > XLIM[0]) & (N.index <= XLIM[1])]
W = W[(W.index > XLIM[0]) & (W.index <= XLIM[1])]
EA1 = EA1[(EA1.index > XLIM[0]) & (EA1.index <= XLIM[1])]
EA2 = EA2[(EA2.index > XLIM[0]) & (EA2.index <= XLIM[1])]
EA3 = EA3[(EA3.index > XLIM[0]) & (EA3.index <= XLIM[1])]
EA4 = EA4[(EA4.index > XLIM[0]) & (EA4.index <= XLIM[1])]

# time averaging
E[E<-32] = np.NaN
N[N<-32] = np.NaN
W[W<-32] = np.NaN

Ebin = E.resample('2min').mean()
Nbin = N.resample('2Min').mean()
Wbin = W.resample('2Min').mean()
fs_bin = 1/(60.0*2.0)       # sample rate, Hz1
## EA1bin = EA1.resample('1min').mean()
## EA2bin = EA2.resample('1min').mean()
## EA3bin = EA3.resample('1min').mean()
## EA4bin = EA4.resample('1min').mean()
EA1bin = EA1.resample('30s').mean()
EA2bin = EA2.resample('30s').mean()
EA3bin = EA3.resample('30s').mean()
EA4bin = EA4.resample('30s').mean()

# Composite echo
EAbin = EA1bin.add(EA2bin, fill_value=0)
EAbin = EAbin.add(EA3bin, fill_value=0)
EAbin = EAbin.add(EA4bin, fill_value=0)
EAbin = EAbin/4
EAbin = EAbin - EAbin.mean(axis=0)

# filter W 
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

order = 6
cutoff_high = 1/(1800.0)  # desired cutoff frequency of the filter, Hz
WW = np.nan_to_num(Wbin)
Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)
#### --------------------------------- ###


#### ---- Function to plot zooming boxes ---- #####
from matplotlib.patches import Rectangle

def zoomingBox(ax1, roi, ax2, color='red', linewidth=2, roiKwargs={}, arrowKwargs={}):
    '''
    **Notes (for reasons unknown to me)**
    1. Sometimes the zorder of the axes need to be adjusted manually...
    2. The figure fraction is accurate only with qt backend but not inline...
    '''
    roiKwargs = dict([('fill',False), ('linestyle','dashed'), ('color',color), ('linewidth',linewidth)] + roiKwargs.items())
    ax1.add_patch(Rectangle([roi[0],roi[2]], roi[1]-roi[0], roi[3]-roi[2], **roiKwargs))
    arrowKwargs = dict([('arrowstyle','-'), ('color',color), ('linewidth',linewidth)] + arrowKwargs.items())
    srcCorners = [[roi[0],roi[2]], [roi[0],roi[3]], [roi[1],roi[2]], [roi[1],roi[3]]]
    dstCorners = ax2.get_position().corners()
    srcBB = ax1.get_position()
    dstBB = ax2.get_position()
    if (dstBB.min[0]>srcBB.max[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.max[0]<srcBB.min[0] and dstBB.min[1]>srcBB.max[1]):
        src = [0, 3]; dst = [0, 3]
    elif (dstBB.max[0]<srcBB.min[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.min[0]>srcBB.max[0] and dstBB.min[1]>srcBB.max[1]):
        src = [1, 2]; dst = [1, 2]
    elif dstBB.max[1] < srcBB.min[1]:
        src = [0, 2]; dst = [1, 3]
    elif dstBB.min[1] > srcBB.max[1]:
        src = [1, 3]; dst = [0, 2]
    elif dstBB.max[0] < srcBB.min[0]:
        src = [0, 1]; dst = [2, 3]
    elif dstBB.min[0] > srcBB.max[0]:
        src = [2, 3]; dst = [0, 1]
    for k in range(2):
        ax1.annotate('', xy=dstCorners[dst[k]], xytext=srcCorners[src[k]], xycoords='figure fraction', textcoords='data', arrowprops=arrowKwargs)


## --------- plot Composite ------- ##
days = dates.DayLocator()
hours6 = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%b %d')
hours1 = dates.HourLocator(interval=1)
min15 = dates.MinuteLocator(interval=15)
dfmt2 = dates.DateFormatter('%H:%M')

levels = np.linspace(-30, 60, 10)
fig = plt.figure(1)

# Wind
ax0 = plt.subplot2grid((7, 1), (0, 0), rowspan=1, colspan=1)
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



ax1 = plt.subplot2grid((7, 1), (1, 0), rowspan=3, colspan=1)
im1 = ax1.contourf(EAbin.index, EAbin.columns, EAbin.T, levels, aspect='auto', cmap=plt.cm.jet)
ax1.set_xlim(XLIM[0], XLIM[1])
ax1.set_ylim(YLIM[0], YLIM[1])
ax1.set_ylabel(r'Depth (m)')
ax01.xaxis.label.set_visible(False)
ax1.invert_yaxis()
ax1.xaxis.set_major_locator(days)
ax1.xaxis.set_major_formatter(dfmt)
ax1.xaxis.set_minor_locator(hours6)
## # Add dashed rectangle for zoom
start1 = mdates.date2num(XLIM_zoom[0])
end1 = mdates.date2num(XLIM_zoom[1])
rect1_x = [start1, end1, end1, start1, start1]
rect1_y = [YLIM[1], YLIM[1], YLIM[0], YLIM[0], YLIM[1]]
ax1.plot(rect1_x, rect1_y, '--', color='deeppink', linewidth=2)
plt.grid()

ax2 = plt.subplot2grid((7, 1), (5, 0), rowspan=3, colspan=1)
im1 = ax2.contourf(EAbin.index, EAbin.columns, EAbin.T, levels, aspect='auto', cmap=plt.cm.jet)
c1 = plt.contour(Wbin.index, Wbin.columns, Whigh, [0.0,], colors='w', linewidths=0.25)
c2 = plt.contourf(Wbin.index, Wbin.columns, Whigh, [-0.1, 0.0], cmap=plt.cm.binary_r, alpha=.3 )
#plt.colorbar(im1, ticks=levels)
cax = plt.axes([0.91,0.15,0.02,0.5])
plt.colorbar(im1, cax=cax)
#plt.colorbar(im1)
ax2.set_xlim(XLIM_zoom[0], XLIM_zoom[1])
ax2.set_ylim(YLIM[0], YLIM[1])
ax2.set_ylabel(r'Depth (m)')
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(hours1)
ax2.xaxis.set_major_formatter(dfmt2)
ax2.xaxis.set_minor_locator(min15)

# Zoom boxes
zoomingBox(ax1, [rect1_x[0], rect1_x[1], rect1_y[0], rect1_y[1]], ax2, color='deeppink')
plt.show()




## --------- plot Single Beam ------- ##
fig = plt.figure(2)
levels = np.linspace(30, 140, 12)
levels = np.linspace(-30, 60, 10)

# Wind
ax0 = plt.subplot2grid((7, 1), (0, 0), rowspan=1, colspan=1)
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

# choose beam here
EAbeam = EA4bin - EA4bin.mean(axis=0)

ax1 = plt.subplot2grid((7, 1), (1, 0), rowspan=3, colspan=1)
im1 = ax1.contourf(EAbeam.index, EAbeam.columns, EAbeam.T, levels, aspect='auto', cmap=plt.cm.jet)
ax1.set_xlim(XLIM[0], XLIM[1])
ax1.set_ylim(YLIM[0], YLIM[1])
ax1.set_ylabel(r'Depth (m)')
ax01.xaxis.label.set_visible(False)
ax1.invert_yaxis()
ax1.xaxis.set_major_locator(days)
ax1.xaxis.set_major_formatter(dfmt)
ax1.xaxis.set_minor_locator(hours6)
## # Add dashed rectangle for zoom
start1 = mdates.date2num(XLIM_zoom[0])
end1 = mdates.date2num(XLIM_zoom[1])
rect1_x = [start1, end1, end1, start1, start1]
rect1_y = [YLIM2[1], YLIM2[1], YLIM2[0], YLIM2[0], YLIM2[1]]
ax1.plot(rect1_x, rect1_y, '--', color='deeppink', linewidth=2)
plt.grid()

ax2 = plt.subplot2grid((7, 1), (5, 0), rowspan=3, colspan=1)
im1 = ax2.contourf(EAbeam.index, EAbeam.columns, EAbeam.T, levels, aspect='auto', cmap=plt.cm.jet)
c1 = plt.contour(Wbin.index, Wbin.columns, Whigh, [0.0,], colors='w', linewidths=0.25)
c2 = plt.contourf(Wbin.index, Wbin.columns, Whigh, [-0.1, 0.0], cmap=plt.cm.binary_r, alpha=.3 )
#plt.colorbar(im1, ticks=levels)
cax = plt.axes([0.91,0.15,0.02,0.5])
plt.colorbar(im1, cax=cax)
#plt.colorbar(im1)
ax2.set_xlim(XLIM_zoom[0], XLIM_zoom[1])
ax2.set_ylim(YLIM2[0], YLIM2[1])
ax2.set_ylabel(r'Depth (m)')
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(hours1)
ax2.xaxis.set_major_formatter(dfmt2)
ax2.xaxis.set_minor_locator(min15)

# Zoom boxes
zoomingBox(ax1, [rect1_x[0], rect1_x[1], rect1_y[0], rect1_y[1]], ax2, color='deeppink')
plt.show()


## fig.set_size_inches(w=8, h=8)
## fig.set_dpi(300)
## fig.tight_layout()
## plt.subplots_adjust(hspace=0.35)
## fig.savefig(fig_name)

