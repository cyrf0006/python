import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd
import matplotlib.dates as mdates # For zoomed rectangle


#### --------- Sequence Selection --------- #### 
XLIM = [pd.Timestamp('2010-03-01 11:00:00'), pd.Timestamp('2010-03-02 1:45:00')]
XLIM1 = [pd.Timestamp('2010-03-01 11:15:00'), pd.Timestamp('2010-03-01 15:00:00')]
XLIM2 = [pd.Timestamp('2010-03-01 21:00:00'), pd.Timestamp('2010-03-02 01:00:00')]
XLIM3 = [pd.Timestamp('2010-03-01 11:30:00'), pd.Timestamp('2010-03-01 14:00:00')]
XLIM4 = [pd.Timestamp('2010-03-01 21:15:00'), pd.Timestamp('2010-03-02 00:30:00')]
#XLIM21 = [pd.Timestamp('2010-03-02 03:00:00'), pd.Timestamp('2010-03-02 06:00:00')]
#XLIM4 = [pd.Timestamp('2010-03-02 03:45:00'), pd.Timestamp('2010-03-02 05:30:00')]

fig_name = 'S1_onset3.png'
#3XLIM2 = [pd.Timestamp('2010-03-01 11:30:00'), pd.Timestamp('2010-03-01 14:00:00')]
YLIM = [19, 78]
YLIM1 = [20, 77]
YLIM2 = [20, 77]
#YLIM3 = [56, 76]
YLIM3 = [53, 63]
YLIM4 = [56, 76]

#### ---------- Meto data --------- ####
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
# --------------------------------------# 


### --------------- ADCP ------------ ###
ADCP_dict = loadmat('./data/IOWp01.mat',squeeze_me=True)
mab_adcp = 1.58
mooring_depth = 83
adcpdir = 'up'
ADCP = { k: ADCP_dict[k] for k in ['SerNmmpersec', 'SerEmmpersec', 'SerVmmpersec']}

E = ADCP_dict['SerEmmpersec']/1000.0 # now in cm/s
N = ADCP_dict['SerNmmpersec']/1000.0
W = ADCP_dict['SerVmmpersec']/1000.0

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
# Cut timeseries
Ebin = Ebin.loc[(Ebin.index >= XLIM[0]) & (Ebin.index <= XLIM[1])]
Nbin = Nbin.loc[(Nbin.index >= XLIM[0]) & (Nbin.index <= XLIM[1])]
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

#### ------ filter timeseries ----- ####
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

# Filter requirements.
order = 6
cutoff_high = 1/(1800.0)  # desired cutoff frequency of the filter, Hz

EE = np.nan_to_num(Ebin)
NN = np.nan_to_num(Nbin)
WW = np.nan_to_num(Wbin)
Ehigh = butter_highpass_filter(EE.T, cutoff_high, fs_bin, order)
Nhigh = butter_highpass_filter(NN.T, cutoff_high, fs_bin, order)
Whigh = butter_highpass_filter(WW.T, cutoff_high, fs_bin, order)


## Ticks managements
days = dates.DayLocator() # AX0 - AX1
hours6 = dates.HourLocator(interval=6)
hours1 = dates.HourLocator(interval=1)
hours2 = dates.HourLocator(interval=2)
dfmt = dates.DateFormatter('%H:%M')

hours1_1 = dates.HourLocator(interval=1)
min15_1 = dates.MinuteLocator(interval=15)
dfmt1 = dates.DateFormatter('%H:%M')

hours1_3 = dates.HourLocator(interval=1)
min15_3 = dates.MinuteLocator(interval=15)
dfmt3 = dates.DateFormatter('%H:%M')

hours1_2 = dates.HourLocator(interval=1)
min15_2 = dates.MinuteLocator(interval=15)
dfmt2 = dates.DateFormatter('%H:%M')

hours1_4 = dates.HourLocator(interval=1)
min15_4 = dates.MinuteLocator(interval=15)
dfmt4 = dates.DateFormatter('%H:%M')

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

#### ------- plot figure -------- ####
fig = plt.figure(1)
fig.subplots_adjust(hspace=.5)
## AX0 - Wind
ax0 = plt.subplot2grid((10,2), (0, 0), rowspan=1, colspan=2)
df = pd.DataFrame(wind_mag)
df = df.set_index('date_time')

ax0.plot(df.index, df.wind_mag, 'b')
ax0.set_xlim(XLIM[0], XLIM[1])
ax0.tick_params(labelbottom='off')
ax0.xaxis.label.set_visible(False)
ax0.xaxis.set_major_locator(hours2)
ax0.xaxis.set_major_formatter(dfmt)
ax0.set_ylabel(r'$\rm \overline{u_w} (m s^{-1})$', color='b')
ax0.set_ylim(0,20)
ax0.tick_params('y', colors='b')
plt.grid()
ax0.text(XLIM[1], 15, 'a  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')

ax01 = ax0.twinx()
df2 = pd.DataFrame(wave_height)
df2 = df2.set_index('date_time')
ax01.plot(df2.index, df2.waveheight_signif, 'r')
ax01.set_xlim(XLIM[0], XLIM[1])
ax01.tick_params(labelbottom='off')
ax01.xaxis.set_major_locator(hours2)
ax01.xaxis.set_major_formatter(dfmt)
ax01.set_ylabel(r'$\rm H_s (m)$', color='r')
ax01.set_ylim(0,5)
ax01.xaxis.label.set_visible(False)
ax01.tick_params('y', colors='r')


## AX1 - Velocity full
ax1 = plt.subplot2grid((10, 2), (1, 0), rowspan=3, colspan=2)
levels = np.linspace(-.005, .005, 11)
c = plt.contourf(Wbin.index, Wbin.columns, Whigh, levels, cmap=plt.cm.RdBu_r, extend="both")
levels2 = np.linspace(0,10, 5)
ct = plt.contour(Tbin.index, Tbin.columns, Tbin.T, levels2, colors='k', linewidth=0.05)
ax1.set_xlim(XLIM[0], XLIM[1])
ax1.set_ylim(YLIM[0], YLIM[1])
ax1.tick_params(labelbottom='on')
ax1.set_ylabel(r'Depth (m)')
ax1.invert_yaxis()
ax1.xaxis.set_major_locator(hours2)
ax1.xaxis.set_major_formatter(dfmt)
ax1.set_xlim(XLIM[0], XLIM[1])
#ax1.set_xlabel(XLIM[0].strftime('%d-%b-%Y'))
#ax1.text(XLIM[0], 25, r"  $\rm w' ( m s^{-1})$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='k', fontweight='bold')
## # Add dashed rectangle for zoom
start1 = mdates.date2num(XLIM1[0])
end1 = mdates.date2num(XLIM1[1])
rect1_x = [start1, end1, end1, start1, start1]
rect1_y = [YLIM1[1], YLIM1[1], YLIM1[0], YLIM1[0], YLIM1[1]]
ax1.plot(rect1_x, rect1_y, '--', color='deeppink', linewidth=2)
start2 = mdates.date2num(XLIM2[0])
end2 = mdates.date2num(XLIM2[1])
rect2_x = [start2, end2, end2, start2, start2]
rect2_y = [YLIM2[1], YLIM2[1], YLIM2[0], YLIM2[0], YLIM2[1]]
ax1.plot(rect2_x, rect2_y, '--', color='deeppink', linewidth=2)

## AX11 - Velocity ZOOM 1
ax11 = plt.subplot2grid((10, 2), (4, 0), rowspan=3, colspan=1)
levels = np.linspace(-.005, .005, 11)
c = plt.contourf(Wbin.index, Wbin.columns, Whigh, levels, cmap=plt.cm.RdBu_r, extend="both")
levels2 = np.linspace(0,10, 5)
ct = plt.contour(Tbin.index, Tbin.columns, Tbin.T, levels2, colors='k', linewidth=0.05)
ax11.set_xlim(XLIM1[0], XLIM1[1])
ax11.set_ylim(YLIM1[0], YLIM1[1])
ax11.tick_params(labelbottom='on')
ax11.set_ylabel(r'Depth (m)')
ax11.invert_yaxis()
ax11.xaxis.set_major_locator(hours1_1)
ax11.xaxis.set_major_formatter(dfmt1)
ax11.xaxis.set_minor_locator(min15_1)
ax11.set_xlim(XLIM1[0], XLIM1[1])
#ax11.text(XLIM1[0], 25, r"  $\rm w' ( m s^{-1})$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='k', fontweight='bold')
## # Add dashed rectangle for zoom
start3 = mdates.date2num(XLIM3[0])
end3 = mdates.date2num(XLIM3[1])
rect3_x = [start3, end3, end3, start3, start3]
rect3_y = [YLIM3[1], YLIM3[1], YLIM3[0], YLIM3[0], YLIM3[1]]
ax11.plot(rect3_x, rect3_y, '--', color='deeppink', linewidth=2)
# Langmuir propagation
propax = [pd.Timestamp('2010-03-01 12:16:00'), pd.Timestamp('2010-03-01 12:37:00')]
propay = [30, 60]
ax11.plot(propax, propay, '--', color='deeppink', linewidth=2)


## AX12 - Temperature Zoom
ax12 = plt.subplot2grid((10, 2), (7, 0), rowspan=3, colspan=1)
levels3 = np.linspace(0,10, 21)
c = plt.contourf(Tbin.index, Tbin.columns, Tbin.T, levels3, cmap=plt.cm.RdBu_r)
ax12.set_xlim(XLIM3[0], XLIM3[1])
ax12.set_ylim(YLIM3[0], YLIM3[1])
ax12.tick_params(labelbottom='on')
ax12.set_ylabel(r'Depth (m)')
#ax12.text(XLIM3[0], 54, r"  $\rm T(^{circ}C)$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='k', fontweight='bold')
ax12.invert_yaxis()
ax12.xaxis.set_major_locator(hours1_3)
ax12.xaxis.set_major_formatter(dfmt3)
ax12.xaxis.set_minor_locator(min15_3)


## AX21 - Velocity ZOOM 2
ax21 = plt.subplot2grid((10, 2), (4, 1), rowspan=3, colspan=1)
levels = np.linspace(-.005, .005, 11)
c = plt.contourf(Wbin.index, Wbin.columns, Whigh, levels, cmap=plt.cm.RdBu_r, extend="both")
ct = plt.contour(Tbin.index, Tbin.columns, Tbin.T, levels2, colors='k', linewidth=0.05)
ax21.set_xlim(XLIM2[0], XLIM2[1])
ax21.set_ylim(YLIM2[0], YLIM2[1])
ax21.tick_params(labelleft='off')
ax21.tick_params(labelbottom='on')
#ax21.set_ylabel(r'Depth (m)')
ax21.invert_yaxis()
ax21.xaxis.set_major_locator(hours1_2)
ax21.xaxis.set_major_formatter(dfmt2)
ax21.xaxis.set_minor_locator(min15_2)
ax21.set_xlim(XLIM2[0], XLIM2[1])
## # Add dashed rectangle for zoom
start4 = mdates.date2num(XLIM4[0])
end4 = mdates.date2num(XLIM4[1])
rect4_x = [start4, end4, end4, start4, start4]
rect4_y = [YLIM4[1], YLIM4[1], YLIM4[0], YLIM4[0], YLIM4[1]]
ax21.plot(rect4_x, rect4_y, '--', color='deeppink', linewidth=2)

# Colorbar for velocity
ax21.text(XLIM2[1], -25.5, r" $\rm w' ( m s^{-1})$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='k', fontweight='bold')
cax = plt.axes([0.91,0.38,0.014,0.34])
plt.colorbar(c, cax=cax, ticks=[-.005, -.003, -.001, .001, .003, .005])


## AX22 - Temperature Zoom
ax22 = plt.subplot2grid((10, 2), (7, 1), rowspan=3, colspan=1)
c = plt.contourf(Tbin.index, Tbin.columns, Tbin.T, levels3, cmap=plt.cm.RdBu_r)
ax22.set_xlim(XLIM4[0], XLIM4[1])
ax22.set_ylim(YLIM4[0], YLIM4[1])
ax22.tick_params(labelbottom='on')
ax21.tick_params(labelleft='off')
ax22.invert_yaxis()
ax22.xaxis.set_major_locator(hours1_4)
ax22.xaxis.set_major_formatter(dfmt4)
ax22.xaxis.set_minor_locator(min15_4)



# Colorbar for temperature
ax22.text(XLIM4[1], 57, r" $\rm T(^{\circ}C)$", horizontalalignment='left', verticalalignment='center', fontsize=14, color='k', fontweight='bold')
cax = plt.axes([0.91,0.11,0.014,0.18])
plt.colorbar(c, cax=cax, ticks=np.linspace(0,10, 6))

zoomingBox(ax1, [rect1_x[0], rect1_x[1], rect1_y[0], rect1_y[1]], ax11, color='deeppink')
zoomingBox(ax1, [rect2_x[0], rect2_x[1], rect2_y[0], rect2_y[1]], ax21, color='deeppink')

zoomingBox(ax11, [rect3_x[0], rect3_x[1], rect3_y[0], rect3_y[1]], ax12, color='deeppink')
zoomingBox(ax21, [rect4_x[0], rect4_x[1], rect4_y[0], rect4_y[1]], ax22, color='deeppink')


#### ---- Save Figure ---- ####
fig.set_size_inches(w=9, h=10)
fig.set_dpi(300)
#fig.tight_layout()
fig.savefig(fig_name)
#plt.show()

