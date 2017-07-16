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
XLIM = [pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00')]
#fig_name = 'S1_outlook_wind.png'
fig_name = 'S1_outlook_meteo.png'

#### ---------- Meteo data --------- ####
meteo_dict = loadmat('SMHI_meteo_55570_201002_201003.mat',squeeze_me=True)
wave_dict = loadmat('SMHI_wave_33004_201002_201003.mat',squeeze_me=True)
wind_dir = { k: meteo_dict[k] for k in ['wind_dir']}
wind_mag = { k: meteo_dict[k] for k in ['wind_mag']}
wave_height = { k: wave_dict[k] for k in ['waveheight_max','waveheight_signif']}
wave_dir = { k: wave_dict[k] for k in ['wavedir_maxenergy']}
wave_period= { k: wave_dict[k] for k in ['waveperiod_mean']}
def matlab2datetime(matlab_datenum):
    day = datetime.datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac

# convert Matlab time into list of python datetime objects and put in dictionary
wind_dir['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
wind_mag['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
wave_height['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]
wave_dir['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]
wave_period['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]

# Wind data from R/V Alkor
#Alkor_dict = loadmat('/media/cyrf0006/Fred32/IOW/IOWdata/AL351_DDL_wind.mat',squeeze_me=True, struct_as_record=False)
Alkor_dict = loadmat('./data/AL351_DDL_wind.mat',squeeze_me=True, struct_as_record=False)
Ualkor =  Alkor_dict['DDL'].wind.speed.data
yearday = Alkor_dict['DDL'].time.data+1
pdTimeAlkor = pd.to_datetime('2010') + pd.to_timedelta(yearday - 1, unit='d')
alkor = pd.DataFrame(Ualkor, index=pdTimeAlkor)
alkor = alkor.resample('30Min').mean()
# --------------------------------------# 


#### ---------- IOW SBE Mcats --------- ####
sbe_dict = loadmat('/home/cyrf0006/research/IOW/iow_mooring/AL351_TSC2_S1_sbe_all.mat',squeeze_me=True)
Tmat = sbe_dict['Temperature']
Smat = sbe_dict['Salinity']
Dmat = sbe_dict['Density']
zVecSBE = sbe_dict['SBE_depth']
yearday = sbe_dict['SBE_decday']
pdTimeCats = pd.to_datetime('2010') + pd.to_timedelta(yearday - 1, unit='d')
TT = pd.DataFrame(Tmat, index=pdTimeCats, columns=zVecSBE) # now in m/s
S = pd.DataFrame(Smat, index=pdTimeCats, columns=zVecSBE)
D = pd.DataFrame(Dmat, index=pdTimeCats, columns=zVecSBE)

## NOTE: Use CTD cast from MSS to get profile after storm!!!
# --------------------------------------# 

#### ---------- CTD cast --------- ####
# CTD casts before storm (pd is powerful!))
lat = 55
lon = 16
ctd_fname = './data/ctd/0001_01_bav_buo.cnv'
data = pd.read_csv(ctd_fname, header=None, delim_whitespace=True)
Zctd = data[:][7]
Tctd = data[:][1]
Sctd = data[:][5]


Ibtm = np.argmax(Zctd)    
Pbin = np.arange(0.5, Zctd[Ibtm], 1)
digitized = np.digitize(Zctd[0:Ibtm], Pbin) #<- this is awesome!
Tbin = np.array([Tctd[0:Ibtm][digitized == i].mean() for i in range(0, len(Pbin))])
Sbin = np.array([Sctd[0:Ibtm][digitized == i].mean() for i in range(0, len(Pbin))])
Z = np.abs(gsw.z_from_p(Pbin,lat))
SA = gsw.SA_from_SP_Baltic(Sbin,lon,lat)
CT = gsw.CT_from_t(SA,Tbin,Pbin)
SIG0 = gsw.sigma0(SA,CT)
N2 = gsw.Nsquared(SA,CT,Pbin,lat)
# ------------------------------------# 

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
MSS_S1_1_dict = loadmat('./data/MSS_DATA/S1_1.mat',squeeze_me=True, struct_as_record=False)
Zmss =  MSS_S1_1_dict['CTD'][2].P
Tmss =  MSS_S1_1_dict['CTD'][2].T
Smss =  MSS_S1_1_dict['CTD'][2].S
timemss1 = pd.Timestamp(MSS_S1_1_dict['STA'][2].date)
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
N2_MSS_02 = gsw.Nsquared(SA_MSS_02,CT_MSS_02,Pbin,lat)
N2_02 = N2_MSS_02[0]
zN2_02 = N2_MSS_02[1]
N2_period_02 = 60.0/swe.cph(N2_02)

MSS_S1_5_dict = loadmat('./data/MSS_DATA/S1_5.mat',squeeze_me=True, struct_as_record=False)
Zmss =  MSS_S1_5_dict['CTD'][-10].P
Tmss =  MSS_S1_5_dict['CTD'][-10].T
Smss =  MSS_S1_5_dict['CTD'][-10].S
timemss5 = pd.Timestamp(MSS_S1_5_dict['STA'][-10].date)
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
storm2 = pd.Timestamp('2010-03-01 12:00:00') # onset langmuir

fig = plt.figure(2)


# AX1 - Wind mag.
ax1 = plt.subplot2grid((7, 9), (0, 2), rowspan=1, colspan=6)
df = pd.DataFrame(wind_mag)
df = df.set_index('date_time')
df.plot(ax=ax1, grid='on', legend=False)
alkor.plot(ax=ax1, grid='on', legend=False)
#ax1.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00 '))
ax1.set_xlim(XLIM)
ax1.set_ylim(0,20)
ax1.tick_params(labelbottom='off')
ax1.xaxis.label.set_visible(False)
ax1.xaxis.set_major_locator(days)
ax1.xaxis.set_major_formatter(dfmt)
ax1.xaxis.set_minor_locator(hours)
ax1.set_ylabel(r'$\overline{U} (m/s)$')
ax1.legend(['SMHI','ship'], bbox_to_anchor=(1., 1.), loc=2)
ax1.text(XLIM[0], 15, '  a', horizontalalignment='left', verticalalignment='center', fontsize=15, color='k')
plt.plot([storm2, storm2], [0, 20], '--k')

# AX2 - Wave heights
ax2 = plt.subplot2grid((7, 9), (1, 2), rowspan=1, colspan=6)
df = pd.DataFrame(wave_height)
df = df.set_index('date_time')
df.plot(ax=ax2, grid='on')
#XTICKS = axes[0].get_xticks()
#ax2.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00'))
ax2.set_xlim(XLIM)
ax2.set_ylim(0,7.5)
ax2.tick_params(labelbottom='off')
ax2.xaxis.label.set_visible(False)
ax2.set_ylabel('H(m)')
ax2.xaxis.set_major_locator(days)
ax2.xaxis.set_major_formatter(dfmt)
ax2.xaxis.set_minor_locator(hours)
ax2.legend([r'$\rm H_{max}$', r'$\rm H_{s}$'], bbox_to_anchor=(1., 1.), loc=2)
# bbox_to_anchor is origin; loc=1 is inside; bordaxesoad=0 means directly on the axis
ax2.text(XLIM[0], 6, '  b', horizontalalignment='left', verticalalignment='center', fontsize=15, color='k')
plt.plot([storm2, storm2], [0, 7.5], '--k')


# AX3 - Wind/wave direction
ax3 = plt.subplot2grid((7, 9), (2, 2), rowspan=1, colspan=6)
df1 = pd.DataFrame(wind_dir)
df2= pd.DataFrame(wave_dir)
df1 = df1.set_index('date_time')
df2 = df2.set_index('date_time')
df1.plot(ax=ax3, grid='on', label="winddir", legend=True)
df2.plot(secondary_y=False, ax=ax3, grid='on', label="wavedir", legend=True)
#ax3.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00'))
ax3.set_xlim(XLIM)
ax3.set_ylim(85,365)
ax3.set_ylabel(r'$\rm \theta$')
ax3.xaxis.label.set_visible(False)
ax3.tick_params(labelbottom='off')
#ax3.legend([r'$\rm \theta_{wind}$', r'$\rm \theta_{wave}$'], bbox_to_anchor=(-0.18, 0.8), loc=1, borderaxespad=0)
ax3.legend([r'$\rm wind$', r'$\rm wave$'], bbox_to_anchor=(1., 1.), loc=2)
ax3.xaxis.set_major_locator(days)
ax3.xaxis.set_major_formatter(dfmt)
ax3.xaxis.set_minor_locator(hours)
plt.yticks([90, 180, 270, 360], ['E', 'S', 'W', 'N'], rotation='horizontal')
ax3.text(XLIM[0], 340, '  c', horizontalalignment='left', verticalalignment='center', fontsize=15, color='k')
plt.plot([storm2, storm2], [85, 365], '--k')

# AX4 - Wave period
ax4 = plt.subplot2grid((7, 9), (3, 2), rowspan=1, colspan=6)
df = pd.DataFrame(wave_period)
df = df.set_index('date_time')
df.plot(ax=ax4, grid='on', legend=False)
#ax4.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00'))
ax4.set_xlim(XLIM)
ax4.set_ylim(2,7)
ax4.xaxis.label.set_visible(False)
ax4.tick_params(labelbottom='off')
ax4.set_ylabel(r'$\rm \overline{T}_s (s)$')
ax4.xaxis.set_major_locator(days)
ax4.xaxis.set_major_formatter(dfmt)
ax4.xaxis.set_minor_locator(hours)
ax4.text(XLIM[0], 6, '  d', horizontalalignment='left', verticalalignment='center', fontsize=15, color='k')
plt.plot([storm2, storm2], [2, 7], '--k')

# AX5 - Temperature
ax5 = plt.subplot2grid((7, 9), (4, 2), rowspan=3, colspan=6)
levels = np.linspace(0,10, 21)
levels2 = np.linspace(0,10, 11)
c = plt.contourf(T.index, T.columns, T.T, levels, cmap=plt.cm.RdBu_r)
#plt.plot([storm, storm], [53, 83], '--k')
plt.plot([timemss1, timemss1], [30, 83], '--')
plt.plot([timemss2, timemss2], [30, 83], '--')
plt.plot([timemss5, timemss5], [30, 83], '--')
plt.plot([storm2, storm2], [30, 83], '--k')
cax = plt.axes([0.9,0.08,0.02,0.3])
plt.colorbar(c, cax=cax)
ax5.set_xlim(XLIM[0], XLIM[1])
#ax5.set_ylim(1, 83)
ax5.set_ylim(30, 83)
ax5.tick_params(labelbottom='on')
ax5.tick_params(labelleft='off')
ax5.invert_yaxis()
ax5.xaxis.set_major_locator(days)
ax5.xaxis.set_major_formatter(dfmt)
ax5.xaxis.set_minor_locator(hours)
ax5.text(XLIM[1], 35, r'  T($^{\circ}$C)', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
ax5.text(XLIM[0], 33, '  e', horizontalalignment='left', verticalalignment='center', fontsize=15, color='k')
#ax5.grid()

# add rectangle
## import matplotlib.dates as mdates
## start = mdates.date2num(XLIM[0])
## end = mdates.date2num(XLIM[1])
## width = end - start
## rect_x = [start, end, end, start, start]
## rect_y = [0,0,52,52,0]
## rect = zip(rect_x, rect_y)
## Rgon = plt.Polygon(rect,color=np.multiply([1.0,1.0,1.0],.7), alpha=0.0, hatch='/')
## ax2.add_patch(Rgon)

# add zoomed rectangle
## start = mdates.date2num(pd.Timestamp('2010-03-02 03:00:00'))
## end = mdates.date2num(pd.Timestamp('2010-03-02 09:00:00'))
## rect_x = [start, end, end, start, start]
## rect_y = [83, 83, 53, 53, 83]
## rect = zip(rect_x, rect_y)
## Rgon2 = plt.Polygon(rect, color='k', ls='--', lw=1, alpha=1, fill=False)
## #Rgon = plt.Polygon(rect,color='none', edgecolor='red', facecolor="red", hatch='/')
## ax2.add_patch(Rgon2)


# AX6 - N2
ax6 = plt.subplot2grid((7, 9), (4, 0), rowspan=3, colspan=2)
ax6.semilogx(np.sqrt(N2_01), zN2_01)
ax6.semilogx(np.sqrt(N2_02), zN2_02)
ax6.semilogx(np.sqrt(N2_05), zN2_05)
ax6.set_ylabel(r'Depth (m)')
ax6.invert_yaxis()
#ax6.set_ylim(1, 83)
ax6.set_ylim(30, 83)
ax6.invert_yaxis()
ax6.grid()
ax6.legend(['28/02', '02/03', '04/03'], bbox_to_anchor=(0., 1.0), loc=3)
ax6.set_xlabel(r'$\rm N (s^{-1})$')
#ax6.set_xticks([1e0, 1e1, 1e2])
ax6.set_xlim(3e-3, 1e-1)
ax6.text(5e-2, 33, 'f', horizontalalignment='left', verticalalignment='center', fontsize=15, color='k')

#### --------- Save Figure ------------- ####
fig.set_size_inches(w=8, h=10)
fig.set_dpi(300)
fig.tight_layout()
fig.savefig(fig_name)

#plt.show()





