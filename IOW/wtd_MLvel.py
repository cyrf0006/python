import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd
import math

# Load data
ADCP_dict = loadmat('./data/IOWp01.mat',squeeze_me=True)
mab_adcp = 1.58
mooring_depth = 83
adcpdir = 'up'
XLIM = [pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 00:00:00')]
XLIM_cut = XLIM

fig_name = 'ML_vel.png'

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

### --------- ADCP data --------- ###
ADCP = { k: ADCP_dict[k] for k in ['SerNmmpersec', 'SerEmmpersec', 'SerVmmpersec']}

# To Pandas DataFrame
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

# Cut timeseries
E = E.loc[(E.index >= XLIM_cut[0]) & (E.index <= XLIM_cut[1])]
N = N.loc[(N.index >= XLIM_cut[0]) & (N.index <= XLIM_cut[1])]
W = W.loc[(W.index >= XLIM_cut[0]) & (W.index <= XLIM_cut[1])]

# Cleaning
E[E<-32] = np.NaN
N[N<-32] = np.NaN
W[W<-32] = np.NaN

# time average
Ebin = E.resample('30min').mean()
Nbin = N.resample('30min').mean()
Wbin = W.resample('30min').mean()
days = dates.DayLocator()
hours6 = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%b %d')


#### ---- Compute ML velocity ---- ####

# Average 20-30m
u_ML = Ebin.iloc[:, -13:-2].mean(axis=1)
v_ML = Nbin.iloc[:, -13:-2].mean(axis=1)
u_IW = Ebin.iloc[:, 3:4].mean(axis=1)
v_IW = Nbin.iloc[:, 3:4].mean(axis=1)
                                 
U_ML = np.sqrt(u_ML**2 + v_ML**2) 
U_ML_dir = np.arctan2(u_ML,v_ML)*180/np.pi
U_ML_dir[U_ML_dir.values<0] = U_ML_dir[U_ML_dir.values<0] + 360.0
U_IW = np.sqrt(u_IW**2 + v_IW**2) 
U_IW_dir = np.arctan2(u_IW,v_IW)*180/np.pi
U_IW_dir[U_IW_dir.values<0] = U_IW_dir[U_IW_dir.values<0] + 360.0

#### ---- PLot ---- ####

# plot 1

#### ----------- plot ------------ ####
days = dates.DayLocator()
hours6 = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%b %d')
hours1 = dates.HourLocator(interval=1)

fig = plt.figure()

# Wind
ax0 = plt.subplot(2, 1, 1)
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
ax0.set_ylim(0,15)
ax0.tick_params('y', colors='b')
plt.grid()
#plt.plot([storm, storm], [0, 20], '--k')
#plt.plot([storm2, storm2], [0, 20], '--k')
ax0.text(XLIM[0], 13, ' a', horizontalalignment='left', verticalalignment='center', fontsize=15, color='k')

ax01 = ax0.twinx()
df2 = pd.DataFrame(wind_dir)
df2 = df2.set_index('date_time')
#df2 = df.set_index('date_time')
#df2 = pd.DataFrame(wave_height)
#ax01.plot(df2.index, df2.waveheight_signif, 'r')
ax01.plot(df2.index, df2.wind_dir, 'r')
ax01.set_xlim(XLIM[0], XLIM[1])
ax01.tick_params(labelbottom='off')
ax01.xaxis.set_major_locator(days)
ax01.xaxis.set_major_formatter(dfmt)
ax01.xaxis.set_minor_locator(hours6)
ax01.set_ylabel(r'$\rm \theta$', color='r')
#ax01.set_ylim(0,5)
ax01.xaxis.label.set_visible(False)
ax01.tick_params('y', colors='r')
#ax01.legend([r'$\rm \theta_{wind}$', r'$\rm \theta_{wave}$'], bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.yticks([0, 90, 180, 270, 360], ['N', 'E', 'S', 'W', 'N'], rotation='horizontal')


ax1 = plt.subplot(2, 1, 2)
ax1.plot(U_ML, 'b')
ax1.plot(U_IW, '--b')
#ax0.plot(alkor, 'b')
ax1.set_xlim(XLIM[0], XLIM[1])
## ax1.xaxis.set_major_locator(days)
## ax1.xaxis.set_major_formatter(dfmt)
## ax1.xaxis.set_minor_locator(hours6)
ax1.set_ylabel(r'$\rm \overline{U} (m s^{-1})$', color='b')
#ax1.set_ylim(0,20)
ax1.tick_params('y', colors='b')
plt.grid()
#plt.plot([storm, storm], [0, 20], '--k')
#plt.plot([storm2, storm2], [0, 20], '--k')
#ax1.text(XLIM[1], 15, 'a  ', horizontalalignment='right', verticalalignment='center', fontsize=15, color='k')
ax1.text(XLIM[0], .33, ' b', horizontalalignment='left', verticalalignment='center', fontsize=15, color='k')

ax11 = ax1.twinx()
ax11.plot(U_ML_dir, 'r')
ax11.plot(U_IW_dir, '--r')
ax11.set_xlim(XLIM[0], XLIM[1])
ax11.xaxis.set_major_locator(days)
ax11.xaxis.set_major_formatter(dfmt)
ax11.xaxis.set_minor_locator(hours6)
ax11.set_ylabel(r'$\rm \theta$', color='r')
ax11.tick_params('y', colors='r')
plt.yticks([0, 90, 180, 270, 360], ['N','E', 'S', 'W', 'N'], rotation='horizontal')

plt.show()


fig.set_size_inches(w=6, h=6)
fig.set_dpi(150)
#fig.tight_layout()
fig.savefig(fig_name)
#plt.show()
