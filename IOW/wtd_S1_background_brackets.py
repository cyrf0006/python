# This function aims to present backgound confitions for 3 MSS casts (before "during" and after storm)

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
import SW_extras as swe

# Some infos:
fig_name = 'S1_background.png'
lat = 55
lon = 16
Pbin = np.arange(0.5, 85, 1)

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
SA = gsw.SA_from_SP_Baltic(S,lon,lat)
CT = gsw.CT_from_t(SA,TT,zVecSBE)
SIG0 = gsw.sigma0(SA,CT)

# To DataFrame
SIG0_SBE = pd.DataFrame(SIG0, index=pdTimeCats, columns=zVecSBE)
SA_SBE = pd.DataFrame(SA, index=pdTimeCats, columns=zVecSBE)
CT_SBE = pd.DataFrame(CT, index=pdTimeCats, columns=zVecSBE)

# time average MicroCats
SIG0_SBEbin = SIG0_SBE.resample('30min').mean()
SA_SBEbin = SA_SBE.resample('30min').mean()
CT_SBEbin = CT_SBE.resample('30min').mean()

# profile during the storm
storm_time = pd.Timestamp('2010-03-02 00:00:00')
idx_storm = np.argmin(np.abs(SIG0_SBEbin.index - storm_time)) # <----- This is pretty cool!
SIG0_SBE_prof = SIG0_SBEbin.iloc[idx_storm]
SA_SBE_prof = SA_SBEbin.iloc[idx_storm]
CT_SBE_prof = CT_SBEbin.iloc[idx_storm]
[N2_SBE, zN2_SBE] = gsw.Nsquared(SA_SBE_prof,CT_SBE_prof,zVecSBE,lat)
N2period_SBE = 60.0/swe.cph(N2_SBE)

# --------------------------------------# 

#### ---------- MSS cast --------- ####
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
idx_sort = SIG0_MSS_01.argsort() # sort density
SA_MSS_01 = SA_MSS_01[idx_sort]
CT_MSS_01 = CT_MSS_01[idx_sort]
SIG0_MSS_01 = SIG0_MSS_01[idx_sort]
N2_MSS_01 = gsw.Nsquared(SA_MSS_01,CT_MSS_01,Pbin,lat)
N2_01 = N2_MSS_01[0]
zN2_01 = N2_MSS_01[1]
N2_period_01 = 60.0/swe.cph(N2_01)

# save sigma for Daniel
from scipy import signal
from astropy.io import ascii
def average(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)
SIG0_smooth = average(SIG0_MSS_01, 4)
P_smooth = average(Pbin, 4)

ascii.write([P_smooth, SIG0_smooth], 'sig0_4m.txt', names=['P', 'sig0'])
ascii.write([Pbin, SIG0_MSS_01], 'sig0_1m.txt', names=['P', 'sig0'])


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
idx_sort = SIG0_MSS_05.argsort() # sort density
SA_MSS_05 = SA_MSS_05[idx_sort]
CT_MSS_05 = CT_MSS_05[idx_sort]
SIG0_MSS_05 = SIG0_MSS_05[idx_sort]
N2_MSS_05 = gsw.Nsquared(SA_MSS_05,CT_MSS_05,Pbin,lat)
N2_05 = N2_MSS_05[0]
zN2_05 = N2_MSS_05[1]
N2_period_05 = 60.0/swe.cph(N2_05)
## # ------------------------------------# 


#### ---------- plots --------- ####


fig = plt.figure(2)


# AX1 - T
ax1 = plt.subplot2grid((1, 11), (0, 0), rowspan=1, colspan=2)
ax1.plot(CT_MSS_01, Pbin)
ax1.plot(CT_MSS_02, Pbin)
ax1.plot(CT_MSS_05, Pbin)
ax1.grid()
ax1.set_ylabel(r'Depth (m)')
ax1.set_ylim(0, 85)
ax1.invert_yaxis()
ax1.set_xlabel(r'$\rm T (^{\circ}C)$')
ax1.legend(['28/02 MSS', '02/03 MSS', '04/03 MSS', '02/03 SBE'])
plt.text(1, 83, 'a', fontsize=14, fontweight='bold')

# AX2 - S
ax2 = plt.subplot2grid((1, 11), (0, 2), rowspan=1, colspan=2)
ax2.plot(SA_MSS_01, Pbin)
ax2.plot(SA_MSS_02, Pbin)
ax2.plot(SA_MSS_05, Pbin)
ax2.grid()
ax2.set_ylim(0, 85)
ax2.set_yticklabels([])
ax2.invert_yaxis()
ax2.set_xlabel(r'$\rm S_A (g Kg^{-1})$')
plt.text(7, 83, 'b', fontsize=14, fontweight='bold')

# AX3 - Sigma0
ax3 = plt.subplot2grid((1, 11), (0, 4), rowspan=1, colspan=2)
ax3.plot(SIG0_MSS_01, Pbin)
ax3.plot(SIG0_MSS_02, Pbin)
ax3.plot(SIG0_MSS_05, Pbin)
ax3.plot(SIG0_SBE_prof, zVecSBE, '-o')
ax3.grid()
ax3.set_ylim(0, 85)
ax3.set_yticklabels([])
ax3.invert_yaxis()
ax3.set_xlabel(r'$\rm \sigma_0 (Kg m^{-3})$')
#ax3.legend(['28/02 MSS', '02/03 MSS', '04/03 MSS', '02/03 SBE'])
plt.text(6, 83, 'c', fontsize=14, fontweight='bold')


# AX6 - Annotation
ax6 = plt.subplot2grid((1, 11), (0, 6), rowspan=1, colspan=1)
ax6.annotate('Mixed layer',
            xy=(0.0, .94),  # theta, radius
            xytext=(0.4, .94),    # fraction, fraction
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-[, widthB=2.7, lengthB=.8', lw=2),
            horizontalalignment='left',
            verticalalignment='center',
            rotation=90
            )

ax6.annotate('Transition layer',
            xy=(0.0, .68),  # theta, radius
            xytext=(0.4, .68),    # fraction, fraction
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-[, widthB=9.5, lengthB=.8', lw=2),
            horizontalalignment='left',
            verticalalignment='center',
            rotation=90
            )

ax6.annotate('Halocline',
            xy=(0.0, .35),  # theta, radius
            xytext=(0.4, .35),    # fraction, fraction
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-[, widthB=5.6, lengthB=.8', lw=2),
            horizontalalignment='left',
            verticalalignment='center',
            rotation=90
            )


ax6.annotate('Bottom layer',
            xy=(0.0, .1),  # theta, radius
            xytext=(0.4, .1),    # fraction, fraction
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-[, widthB=5, lengthB=.8', lw=2),
            horizontalalignment='left',
            verticalalignment='center',
            rotation=90
            )
ax6.axis('off')


# AX4 - N2
ax4 = plt.subplot2grid((1, 11), (0, 7), rowspan=1, colspan=2)
## ax4.semilogx(N2_01, zN2_01)
## ax4.semilogx(N2_02, zN2_02)
## ax4.semilogx(N2_05, zN2_05)
## ax4.semilogx(N2_SBE, zN2_SBE, '-o')
ax4.semilogx(np.sqrt(N2_01), zN2_01)
ax4.semilogx(np.sqrt(N2_02), zN2_02)
ax4.semilogx(np.sqrt(N2_05), zN2_05)
ax4.semilogx(np.sqrt(N2_SBE), zN2_SBE, '-o')
ax4.invert_yaxis()
ax4.grid()
ax4.set_xlabel(r'$\rm N (s^{-1})$')
ax4.set_ylim(0, 85)
#ax4.set_xlim(1e-7, 1e-2)
ax4.set_xlim(8e-4, 1e-1)
ax4.set_yticklabels([])
ax4.invert_yaxis()
#plt.text(2e-7, 83, 'd', fontsize=14, fontweight='bold')
plt.text(1e-3, 83, 'd', fontsize=14, fontweight='bold')


# AX5 - N2period
ax5 = plt.subplot2grid((1, 11), (0, 9), rowspan=1, colspan=2)
ax5.plot(N2_period_01, zN2_01)
ax5.plot(N2_period_02, zN2_02)
ax5.plot(N2_period_05, zN2_05)
ax5.semilogx(N2period_SBE, zN2_SBE, '-o')
ax5.plot([6.4,6.4], [10,85], '--k', linewidth=0.5)
ax5.plot([20, 20], [8,85], '--k', linewidth=0.5)
ax5.plot([1.4, 1.4], [35,85], '--k', linewidth=0.5)
plt.text(3, 9, '6.4min')
plt.text(18, 7, '20min')
plt.text(1.5, 34, '85s')
#ax5.semilogx(N2_period_01, zN2_01)
#ax5.semilogx(N2_period_02, zN2_02)
#ax5.semilogx(N2_period_05, zN2_05)
ax5.grid()
ax5.set_xlabel(r'$\rm T_N (min)$')
ax5.yaxis.label.set_visible(False)
ax5.set_ylim(0, 85)
ax5.set_xlim(0, 60)
ax5.set_yticklabels([])
ax5.invert_yaxis()
plt.text(1.5, 83, 'e', fontsize=14, fontweight='bold')


#### --------- Save Figure ------------- ####
fig.set_size_inches(w=11, h=8)
fig.set_dpi(300)
fig.tight_layout()
fig.savefig(fig_name)



## #### ---- Check SBE casts ---- ######
## fig = plt.figure(3)
## ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
## c = ax1.contourf(D.index, D.columns, D.T)
## ax1.invert_yaxis()
## plt.colorbar(c)
## plt.show()



