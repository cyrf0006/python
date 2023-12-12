'''
Last iteration for the comparison plots for the entire Catalonian coast.

This is the modification of icatmar_PC_catches_corr_seine_regional.py made while working on the paper.


Frederic.Cyr@dfo-mpo.gc.ca
December 2023
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import seaborn as sn
import cmocean as cmo
from matplotlib.colors import from_levels_and_colors

months = [1, 12] # months to keep

# For map limits
lllon = -60.
urlon = 80.
lllat = -50
urlat = 80.

# For study area
lat1 = 26
lat2 = 45
lon1 =  -8
lon2 = 42

v = np.arange(995, 1025) # SLP values

# Load SLP data from NOAA ESRL
ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mon.mean.nc')
# Selection of a subset region
ds = ds.where((ds.lon>=lon1) & (ds.lon<=lon2), drop=True) # original one
ds = ds.where((ds.lat>=lat1) & (ds.lat<=lat2), drop=True)
# spatial average to Series
ds_series = ds.mean(dim=['lat', 'lon'])
df = ds_series['slp'].to_series()
# restrict time
df = df[df.index.year>=2000]
df = df.resample('As').mean()
df = (df - df.mean()) / df.std()

# all Med
pc_zos = pd.read_pickle('./EOFs/scores_zos_annual_med.pkl')
pc_so = pd.read_pickle('./EOFs/scores_so_annual_med.pkl')
pc_so200 = pd.read_pickle('./EOFs/scores_so_200_annual_med.pkl')
pc_so400 = pd.read_pickle('./EOFs/scores_so_400_annual_med.pkl')
pc_botT = pd.read_pickle('./EOFs/scores_bottomT_annual_med.pkl')
pc_sst = pd.read_pickle('./EOFs/scores_thetao_annual_med.pkl')
pc_sst200 = pd.read_pickle('./EOFs/scores_thetao_200_annual_med.pkl')
pc_sst400 = pd.read_pickle('./EOFs/scores_thetao_400_annual_med.pkl')
# NW Med
pc_zos_NW = pd.read_pickle('./EOFs/scores_zos_annual_NW.pkl')
pc_so_NW = pd.read_pickle('./EOFs/scores_so_annual_NW.pkl')
pc_so200_NW = pd.read_pickle('./EOFs/scores_so_200_annual_NW.pkl')
pc_so400_NW = pd.read_pickle('./EOFs/scores_so_400_annual_NW.pkl')
pc_botT_NW = pd.read_pickle('./EOFs/scores_bottomT_annual_NW.pkl')
pc_sst_NW = pd.read_pickle('./EOFs/scores_thetao_annual_NW.pkl')
pc_sst200_NW = pd.read_pickle('./EOFs/scores_thetao_200_annual_NW.pkl')
pc_sst400_NW = pd.read_pickle('./EOFs/scores_thetao_400_annual_NW.pkl')
# NNW Med
pc_zos_NNW = pd.read_pickle('./EOFs/scores_zos_annual_NNW.pkl')
pc_so_NNW = pd.read_pickle('./EOFs/scores_so_annual_NNW.pkl')
pc_so075_NNW = pd.read_pickle('./EOFs/scores_so_075_annual_NNW.pkl')
pc_so200_NNW = pd.read_pickle('./EOFs/scores_so_200_annual_NNW.pkl')
pc_so400_NNW = pd.read_pickle('./EOFs/scores_so_400_annual_NNW.pkl')
pc_botT_NNW = pd.read_pickle('./EOFs/scores_bottomT_annual_NNW.pkl')
pc_sst_NNW = pd.read_pickle('./EOFs/scores_thetao_annual_NNW.pkl')
pc_sst075_NNW = pd.read_pickle('./EOFs/scores_thetao_075_annual_NNW.pkl')
pc_sst200_NNW = pd.read_pickle('./EOFs/scores_thetao_200_annual_NNW.pkl')
pc_sst400_NNW = pd.read_pickle('./EOFs/scores_thetao_400_annual_NNW.pkl')                   
pc_eke_NNW = pd.read_pickle('./EOFs/scores_EKE_annual_NNW.pkl')
pc_ke_NNW = pd.read_pickle('./EOFs/scores_KE_annual_NNW.pkl')
pc_wmld_NNW = pd.read_pickle('./EOFs/scores_wMLD_annual_NNW.pkl')
pc_chla_NNW = pd.read_pickle('./EOFs/scores_chla_annual_NNW.pkl')
pc_chla75_NNW = pd.read_pickle('./EOFs/scores_chla75_annual_NNW.pkl')


## ---- Catch data ---- ##
# Load data
df_landings = pd.read_csv('20231017_Landings_PS_00-22.csv')

VAR = 'Average weight (Kg / day * vessel)' # CPUE
#VAR = 'SumWeight_Kg'

# Average all catalunya
df_cpue = df_landings.groupby(['Date', 'Subgroup', ]).mean()[VAR]
df_cpue = df_cpue.unstack()
df_cpue.index = pd.to_datetime(df_cpue.index)

# Check seasonality of Auxis
auxis_monthly = df_cpue[['Auxis spp.']]
auxis_monthly = auxis_monthly.groupby(auxis_monthly.index.month).mean()

# Annual averages
df_cpue = df_cpue.resample('As').mean()

## Anomalies
df_anom = (df_cpue - df_cpue.mean()) / df_cpue.std()
df_anom.index = df_anom.index.year

## ---- Individual speciess---- ##
sardine = df_anom[['Sardina pilchardus']].mean(axis=1)
anchoa = df_anom[['Engraulis encrasicolus']].mean(axis=1) 
sardinella = df_anom[['Sardinella aurita']].mean(axis=1) 
auxis = df_anom[['Auxis spp.']].mean(axis=1) 
seabream = df_anom[['Spondyliosoma cantharus']].mean(axis=1) 

## ----  Comparision plots ---- ##

## Auxis vs EKE(2)
plt.close('all')
fig = plt.figure(3)
p = pc_eke_NNW.iloc[1]*-1
ax = p.plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
auxis.plot(ax=ax2, linewidth=1, color='tab:blue')
plt.title('EKE (-PC2); r=0.61')
ax.legend(['EKE'], loc='upper left')
ax2.legend(['Auxis spp.'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_EKE_2_auxis.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Anchoas vs T400(2)
plt.close('all')
fig = plt.figure(3)
ax = pc_sst400_NNW.iloc[1].plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
anchoa.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('T 400m (PC2); r=0.66')
ax.legend(['T 400m'], loc='upper left')
ax2.legend(['Anchoa'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_T400_2_anchoa.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Anchoas, Sardines vs T400(2)
plt.close('all')
fig = plt.figure(3)
ax = pc_sst400_NNW.iloc[1].plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
anchoa.plot(ax=ax2, linewidth=1, color='tab:blue')
sardine.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('T 400m - PC2')
ax.legend(['T 400m'], loc='upper left')
ax2.legend(['Anchoa', 'Sardine'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_T400_2_anchoa_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Sardines vs Sal 75m (3)
plt.close('all')
fig = plt.figure(3)
ax = pc_so075_NNW.iloc[2].plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('S 75m - PC3')
ax.legend(['T 400m'], loc='upper left')
ax2.legend(['Sardine'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_S075_3_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Sardinella vs Surf Sal (1)
plt.close('all')
fig = plt.figure(3)
ax = pc_so_NNW.iloc[0].plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
sardinella.plot(ax=ax2, linewidth=1, color='tab:red')
plt.title('Surface salinity (PC1); r=-0.62')
ax.legend(['S surf.'], loc='upper left')
ax2.legend(['Sardinella'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_S0_1_sardinella.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Dorada vs KE (1)
plt.close('all')
fig = plt.figure(3)
ax = pc_ke_NNW.iloc[0].plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
seabream.plot(ax=ax2, linewidth=1, color='tab:red')
plt.title('Kinetic Energy (PC1); r=0.76')
ax.legend(['KE'], loc='upper left')
ax2.legend(['Spondyliosoma'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_KE_1_seabream.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



# ------------------ EXTRAS - Combinations of PCs -------------- #


## Sardinella vs Surf Sal (1+3)
plt.close('all')
fig = plt.figure(3)
P = pc_so_NNW.iloc[2] - pc_so_NNW.iloc[0]
ax = P.plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
sardinella.plot(ax=ax2, linewidth=1, color='tab:red')
plt.title('Surface salinity  (PC3 - PC1); r=0.81')
ax.legend(['S surf.'], loc='upper left')
ax2.legend(['Sardinella'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_S0_1-3_sardinella.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Sardinella vs Surf Sal (1+3) + S75 (1+3)
plt.close('all')
fig = plt.figure(3)
P = pc_so_NNW.iloc[2] - pc_so_NNW.iloc[0] + pc_so075_NNW.iloc[2] - pc_so075_NNW.iloc[0] 
ax = P.plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
sardinella.plot(ax=ax2, linewidth=1, color='tab:red')
plt.title('Salinity surf (PC3-PC1) & 75m (PC3-PC1); r=0.82')
ax.legend(['S surf.'], loc='upper left')
ax2.legend(['Sardinella'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_S0_1-3_S75_1-3_sardinella.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Sardine vs S75m (3) - T400 (2)
plt.close('all')
fig = plt.figure(3)
P = pc_so075_NNW.iloc[2] - pc_sst400_NNW.iloc[1] 
ax = P.plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='tab:red')
plt.title('75m salinity (PC3) & 400m temp (-PC2); r=0.93')
#ax.legend(['PC'], loc='upper left')
ax2.legend(['Sardine'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_S075_T400_2_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Anchoa vs T400m(2) & botT(3)
plt.close('all')
fig = plt.figure(3)
P = pc_sst400_NNW.iloc[1] - pc_botT_NNW.iloc[2] 
ax = P.plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
anchoa.plot(ax=ax2, linewidth=1, color='tab:blue')
plt.title('400m temperature (PC2) & bot.T (-PC3); r=070')
#ax.legend(['PC'], loc='upper left')
ax2.legend(['Anchoa'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_T400_2_botT_3_anchoa.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Anchoa vs T400m(1-2)
plt.close('all')
fig = plt.figure(3)
P = pc_sst400_NNW.iloc[0] + pc_sst400_NNW.iloc[1] 
ax = P.plot(linewidth=2, color='tab:gray')
plt.ylabel('PC')
ax2 = ax.twinx()
anchoa.plot(ax=ax2, linewidth=1, color='tab:blue')
plt.title('400m temperature (PC1&2); r=070')
#ax.legend(['PC'], loc='upper left')
ax2.legend(['Anchoa'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'cat_NNW_T400_1-2_anchoa.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


# ------------------ EXTRAS - Other plots -------------- #
## Auxis spp. seasonal distrib
plt.close('all')
fig = plt.figure(3)
auxis_monthly.plot.bar(zorder=10)
plt.title('Monthly distribution of Auxis spp. (2000-2021)')
plt.ylabel(r'CPUE (Kg/day/vessel')
plt.xlabel('month')
plt.grid()

## Pinche modo2
pc_sst400 = pd.read_pickle('./EOFs/scores_so_400_deseasonNNW.pkl')                   
pc_sst400 = pc_sst400.T
pc_sst400 = pc_sst400.resample('M').mean()
modo2 = pc_sst400[[2]]*-1
modo2 = modo2.groupby(modo2.index.month).max()
modo1 = pc_sst400[[1]]
modo1 = modo1.groupby(modo1.index.month).max()
df = pd.concat([modo1, modo2], axis=1)
plt.close('all')
fig = plt.figure(3)
ax = df.plot.bar(rot=0)
#modo2.plot.bar(zorder=10)
plt.title('T400 monthly variation')
plt.xlabel('month')
plt.grid()

