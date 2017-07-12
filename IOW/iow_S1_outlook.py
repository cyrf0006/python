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
fig_name = 'S1_outlook.png'


#### ---------- IOW SBE Mcats --------- ####
sbe_dict = loadmat('/home/cyrf0006/research/IOW/iow_mooring/AL351_TSC2_S1_sbe_all.mat',squeeze_me=True)
Tmat = sbe_dict['Temperature']
Smat = sbe_dict['Salinity']
Dmat = sbe_dict['Density']
zVecSBE = sbe_dict['SBE_depth']
yearday = sbe_dict['SBE_decday']-1
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
hours = dates.HourLocator(interval=12)
dfmt = dates.DateFormatter('%d')


fig = plt.figure(2)
ax0 = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=1)
#ax0.plot(SIG0, Z)
ax0.plot(SIG0_MSS_01, Z)
#ax0.plot(SIG0_MSS_02, Z)
#ax0.plot(SIG0_MSS_05, Z)
ax0.set_ylabel(r'Depth (m)')
ax0.invert_yaxis()
ax0.set_ylim(1, 83)
ax0.set_xlim(5, 14)
ax0.invert_yaxis()
ax0.grid()
ax0.legend(['28/02', '02/03', '04/03'])
## ax1 = ax0.twiny()
## ax1.plot(N2_period_sec, zVecN2)
## ax1.grid()
ax0.set_xlabel(r'$\rm \sigma_0 (g kg^{-1})$')
#ax0.set_xlabel(r'$\rm T_{N} (s)$')
## ax1.set_ylim(40, 83)
#ax0.xaxis.set_ticks([0, 150, 300, 450])

ax2 = plt.subplot2grid((1, 6), (0, 1), rowspan=1, colspan=5)
levels = np.linspace(0,10, 21)
c = plt.contourf(T.index, T.columns, T.T, levels, cmap=plt.cm.RdBu_r)
plt.colorbar(c)
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

# add rectangle
import matplotlib.dates as mdates
start = mdates.date2num(XLIM[0])
end = mdates.date2num(XLIM[1])
width = end - start
rect_x = [start, end, end, start, start]
rect_y = [0,0,52,52,0]
rect = zip(rect_x, rect_y)
Rgon = plt.Polygon(rect,color=np.multiply([1.0,1.0,1.0],.7), alpha=0.0, hatch='/')
ax2.add_patch(Rgon)

fig.set_size_inches(w=8, h=5)
fig.set_dpi(150)
fig.tight_layout()
fig.savefig(fig_name)
plt.show()





