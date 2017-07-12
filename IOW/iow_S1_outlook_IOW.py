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
fig_name = 'S1_outlook_IOW.png'

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
# --------------------------------------# 

#### ---------- CTD cast --------- ####
# CTD casts before storm (pd is powerful!))
lat = 55
lon = 16
ctd_fname = '/media/cyrf0006/Fred32/IOW/ctd/0001_01_bav_buo.cnv'
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

#### ---------- IOW SBE Mcats --------- ####
sbe_dict = loadmat('/home/cyrf0006/research/IOW/iow_mooring/AL351_TSC2_S1_sbe_all.mat',squeeze_me=True)
Tmat = sbe_dict['Temperature']
#Smat = sbe_dict['Salinity']
#Dmat = sbe_dict['Density']
zVecSBE = sbe_dict['SBE_depth']
yearday = sbe_dict['SBE_decday']+1 # !! NOTE: error in .mat, yearday too short by 1
pdTimeCats = pd.to_datetime('2010') + pd.to_timedelta(yearday - 1, unit='d')

# add fake temperature above mooring (comment to ignore)
zVecSBE = np.append(Pbin[39],zVecSBE)
zVecSBE = np.append(Pbin[1],zVecSBE)
T1 = np.empty(pdTimeCats.size)
T1.fill(Tbin[1])
Tmat = np.column_stack((T1,T1,Tmat))

# Dataframe
TT = pd.DataFrame(Tmat, index=pdTimeCats, columns=zVecSBE) # now in m/s
#S = pd.DataFrame(Smat, index=pdTimeCats, columns=zVecSBE)
#D = pd.DataFrame(Dmat, index=pdTimeCats, columns=zVecSBE)
# --------------------------------------# 


#### --------- NIOZ Tsensors --------- ####
T_dict = loadmat('/media/cyrf0006/Fred32/IOW/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
T = T_dict['Tbin'].T
zVecT = mooring_depth - np.array(T_dict['habVec'])
pdTimeT = pd.date_range('2010-02-28 11:07:00', '2010-03-04 07:37:00', freq='10min')

# add fake temperature above mooring (comment to ignore)
zVecT = np.append(Pbin[52],zVecT)
zVecT = np.append(Pbin[1],zVecT)
#T0 = T[:,0]
T1 = np.empty(pdTimeT.size)
T1.fill(Tbin[1])
T52 = np.empty(pdTimeT.size)
T52.fill(Tbin[52])
T = np.column_stack((T1,T1,T))

# Dataframe
T = pd.DataFrame(T, index=pdTimeT, columns=zVecT)
# --------------------------------------# 

#### ---------- MSS cast --------- ####
# CTD casts before storm (pd is powerful!))
lat = 55
lon = 16
MSS_S1_1_dict = loadmat('/media/cyrf0006/Fred32/IOW/MSS_DATA/S1_1.mat',squeeze_me=True, struct_as_record=False)
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

MSS_S1_2_dict = loadmat('/media/cyrf0006/Fred32/IOW/MSS_DATA/S1_2.mat',squeeze_me=True, struct_as_record=False)
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

MSS_S1_5_dict = loadmat('/media/cyrf0006/Fred32/IOW/MSS_DATA/S1_5.mat',squeeze_me=True, struct_as_record=False)
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
## # ------------------------------------# 


#### --------- plots ---------- ####
days = dates.DayLocator()
hours = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%H:%M')

fig = plt.figure(2)
ax0 = plt.subplot2grid((4, 9), (0, 2), rowspan=1, colspan=6)
df = pd.DataFrame(wind_mag)
df = df.set_index('date_time')
df.plot(ax=ax0, grid='on', legend=False)
ax0.set_xlim(XLIM[0], XLIM[1])
ax0.tick_params(labelbottom='off')
ax0.xaxis.label.set_visible(False)
ax0.set_ylabel(r'$\overline{U} (m/s)$')
ax0.xaxis.set_major_locator(days)
ax0.xaxis.set_major_formatter(dfmt)
ax0.xaxis.set_minor_locator(hours)

ax1 = plt.subplot2grid((4, 9), (1, 0), rowspan=3, colspan=2)
#ax1.plot(SIG0, Z)
ax1.plot(SIG0_MSS_01, Z)
#ax1.plot(SIG0_MSS_02, Z)
#ax1.plot(SIG0_MSS_05, Z)
ax1.set_ylabel(r'Depth (m)')
ax1.invert_yaxis()
ax1.set_ylim(1, 83)
ax1.set_xlim(5, 14)
ax1.invert_yaxis()
ax1.grid()
ax1.legend(['28/02', '02/03', '04/03'])
## ax1 = ax1.twiny()
## ax1.plot(N2_period_sec, zVecN2)
## ax1.grid()
ax1.set_xlabel(r'$\rm \sigma_0 (g kg^{-1})$')
#ax1.set_xlabel(r'$\rm T_{N} (s)$')
## ax1.set_ylim(40, 83)
#ax1.xaxis.set_ticks([0, 150, 300, 450])

ax2 = plt.subplot2grid((4, 9), (1, 2), rowspan=3, colspan=6)
levels = np.linspace(0,10, 21)
c = plt.contourf(TT.index, TT.columns, TT.T, levels, cmap=plt.cm.RdBu_r)
cax = plt.axes([0.9,0.2,0.02,0.5])
plt.colorbar(c, cax=cax)
#plt.colorbar(c)
ax2.set_xlim(XLIM[0], XLIM[1])
ax2.set_ylim(1, 83)
ax2.tick_params(labelbottom='on')
ax2.tick_params(labelleft='off')
#ax2.set_ylabel(r'Depth (m)')
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(days)
ax2.xaxis.set_major_formatter(dfmt)
ax2.xaxis.set_minor_locator(hours)
ax2.text(XLIM[0], 75, r'  T($^{\circ}$C)', horizontalalignment='left', verticalalignment='center', fontsize=14, color='w')
#ax2.grid()

# add rectangle
import matplotlib.dates as mdates
start = mdates.date2num(XLIM[0])
end = mdates.date2num(XLIM[1])
#mid = mdates.date2num(datetime.datetime(2010,03,03,06,30,01))
rect_x = [start, end, end, start, start]
rect_y = [0,0,40,40,0]
rect = zip(rect_x, rect_y)
Rgon = plt.Polygon(rect,color=np.multiply([1.0,1.0,1.0],.7), alpha=0.0, hatch='/')
#Rgon = plt.Polygon(rect,color='none', edgecolor='red', facecolor="red", hatch='/')
ax2.add_patch(Rgon)

fig.set_size_inches(w=8, h=5)
fig.set_dpi(150)
fig.tight_layout()
fig.savefig(fig_name)
plt.show()





