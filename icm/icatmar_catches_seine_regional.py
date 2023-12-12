# data source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=71800&vid=676

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
pc_wmld_NNW = pd.read_pickle('./EOFs/scores_wMLD_annual_NNW.pkl')
pc_chla_NNW = pd.read_pickle('./EOFs/scores_chla_annual_NNW.pkl')
pc_chla75_NNW = pd.read_pickle('./EOFs/scores_chla75_annual_NNW.pkl')



## ---- Catch data ---- ##
# Load data
df_landings = pd.read_csv('20231017_Landings_PS_00-22.csv')

# Separate by region
North = df_landings[(df_landings.PortName=='BLANES') | (df_landings.PortName=='ROSES') | (df_landings.PortName=='PALAMÓS') | (df_landings.PortName=='ARENYS DE MAR') | (df_landings.PortName=='SANT FELIU DE GUÍXOLS') | (df_landings.PortName=="L'ESCALA") | (df_landings.PortName=='PORT DE LA SELVA') | (df_landings.PortName=='LLANÇÀ')]
Center = df_landings[(df_landings.PortName=='BARCELONA') | (df_landings.PortName=='VILANOVA I LA GELTRÚ') | (df_landings.PortName=='TARRAGONA') | (df_landings.PortName=='MATARÓ') | (df_landings.PortName=='CAMBRILS')]
South = df_landings[(df_landings.PortName=='LA RÀPITA') | (df_landings.PortName=="L'AMETLLA DE MAR") | (df_landings.PortName=='DELTEBRE') | (df_landings.PortName=="LES CASES D'ALCANAR")]

# CPUEs
VAR = 'Average weight (Kg / day * vessel)' # CPUE
#VAR = 'SumWeight_Kg'
cpue_north = North.groupby(['Date', 'Subgroup', ]).mean()[VAR]
cpue_north = cpue_north.unstack()
cpue_north.index = pd.to_datetime(cpue_north.index)
cpue_north = cpue_north.resample('As').mean()
cpue_center = Center.groupby(['Date', 'Subgroup', ]).mean()[VAR]
cpue_center = cpue_center.unstack()
cpue_center.index = pd.to_datetime(cpue_center.index)
cpue_center = cpue_center.resample('As').mean()
cpue_south = South.groupby(['Date', 'Subgroup', ]).mean()[VAR]
cpue_south = cpue_south.unstack()
cpue_south.index = pd.to_datetime(cpue_south.index)
cpue_south = cpue_south.resample('As').mean()

# Average all catalunya (deprecated)
# *see -> icatmar_PC_catches_corr_seine_catalunya.py) 
cpue_cat = pd.concat([cpue_north, cpue_center, cpue_south], axis=0)
cpue_cat = cpue_cat.groupby(['Date']).mean()

# Seclect most important species based on value (all regions)
df_value = df_landings.groupby(['Date', 'Subgroup']).mean()['SumAmount_Euros']
df_value = df_value.unstack()
df_value.index = pd.to_datetime(df_value.index)
df_value = df_value.resample('As').mean()
value10 = df_value.sum()[df_value.sum()>=df_value.sum().sort_values().iloc[-10]].index
value8 = df_value.sum()[df_value.sum()>=df_value.sum().sort_values().iloc[-8]].index

cpue_north = cpue_north[value8.values]
cpue_center = cpue_center[value8.values]
cpue_south = cpue_south[value8.values]
cpue_cat = cpue_cat[value10.values]

# Rename columns
cpue_north.columns = cpue_north.keys()+' N'
cpue_center.columns = cpue_center.keys()+' C'
cpue_south.columns = cpue_south.keys()+' S'
cpue_cat.columns = cpue_cat.keys()+' Cat'

# Concat regions
df_cpue = pd.concat([cpue_north, cpue_center, cpue_south, cpue_cat], axis=1)

## Anomalies
df_anom = (df_cpue - df_cpue.mean()) / df_cpue.std()
df_anom.index = df_anom.index.year


## ---- Individual speciess---- ##
sardineN = df_anom[['Sardina pilchardus N']].mean(axis=1)
sardineC = df_anom[['Sardina pilchardus C']].mean(axis=1)
sardineS = df_anom[['Sardina pilchardus S']].mean(axis=1)
sardineCat = df_anom[['Sardina pilchardus Cat']].mean(axis=1)
anchoaN = df_anom[['Engraulis encrasicolus N']].mean(axis=1) 
anchoaC = df_anom[['Engraulis encrasicolus C']].mean(axis=1) 
anchoaS = df_anom[['Engraulis encrasicolus S']].mean(axis=1) 
anchoaCat = df_anom[['Engraulis encrasicolus Cat']].mean(axis=1) 
sardinellaCat = df_anom[['Sardinella aurita Cat']].mean(axis=1) 
auxisN = df_anom[['Auxis spp. N']].mean(axis=1) 
auxisC = df_anom[['Auxis spp. C']].mean(axis=1) 
sardaC = df_anom[['Sarda sarda C']].mean(axis=1) 


#### ---- NNW Med ---- ####

## Compa Surface Sal vs Sardines
plt.close('all')
fig = plt.figure(3)
ax = pc_so_NNW.iloc[2].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardineN.plot(ax=ax2, linewidth=1, color='magenta')
sardineC.plot(ax=ax2, linewidth=1, color='tab:green')
sardineS.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('NNW Surface Salinity - PC3')
ax2.legend(['Sardine N', 'Sardine C', 'Sardine S'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_Sal_3_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa 75m Sal vs Sardines
plt.close('all')
fig = plt.figure(3)
ax = pc_so075_NNW.iloc[2].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardineN.plot(ax=ax2, linewidth=1, color='magenta')
sardineC.plot(ax=ax2, linewidth=1, color='tab:green')
sardineS.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('NNW 75m Salinity - PC3')
ax2.legend(['Sardine N', 'Sardine C', 'Sardine S'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_Sal075_3_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa 75m Sal vs Sardines S/Cat
plt.close('all')
fig = plt.figure(3)
ax = pc_so075_NNW.iloc[2].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardineS.plot(ax=ax2, linewidth=1, color='tab:orange')
sardineCat.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('NNW 75m Salinity - PC3')
ax2.legend(['Sardine S', 'Sardine Cat'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_Sal075_3_sardineCat.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Surface Sal vs Sardinella
plt.close('all')
fig = plt.figure(3)
ax = pc_so_NNW.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardinellaCat.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('NNW Surface Salinity - PC1')
ax2.legend(['Sardinella Cat'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_Sal_1_sardinella.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Compa convection vs Anchoas
plt.close('all')
fig = plt.figure(3)
ax = pc_botT_NNW.iloc[0].plot(linewidth=2)
pc_wmld_NNW.iloc[0].plot(linewidth=2, ax=ax)
plt.ylabel('PC')
ax2 = ax.twinx()
anchoaS.plot(ax=ax2, linewidth=1, color='tab:green')
ax.legend(['Bottom T(1)', 'wMLD(1)'], loc='upper left')
ax2.legend(['Anchoa S'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_convection_anchoa.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa EKE vs Sarda sarda
plt.close('all')
fig = plt.figure(3)
ax = pc_eke_NNW.iloc[4].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardaC.plot(ax=ax2, linewidth=1, color='tab:green')
ax.legend(['EKE(5)'], loc='upper left')
ax2.legend(['Sarda sarda C'], loc='upper right')
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_EKE_sarda_kg.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



## Compa Temp 400m vs Anchoas
plt.close('all')
fig = plt.figure(3)
ax = pc_sst400_NNW.iloc[1].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
anchoaN.plot(ax=ax2, linewidth=1, color='magenta')
anchoaC.plot(ax=ax2, linewidth=1, color='tab:green')
anchoaS.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('NNW 400m Temp - PC2')
ax2.legend(['Anchoa N', 'Anchoa C', 'Anchoa S'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_Temp4_s_anchoa.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Compa Bottom T vs Sardines
plt.close('all')
fig = plt.figure(3)
ax = pc_botT_NNW.iloc[4].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardineN.plot(ax=ax2, linewidth=1, color='magenta')
sardineC.plot(ax=ax2, linewidth=1, color='tab:green')
sardineS.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('NNW Bottom Temperature - PC5')
ax2.legend(['Sardine N', 'Sardine C', 'Sardine S'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_BotT_5_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



## Compa Bottom T vs Auxis spp.
plt.close('all')
fig = plt.figure(3)
ax = pc_botT_NNW.iloc[2].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
auxisN.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('NNW Bottom Temperature - PC2')
ax2.legend(['Auxis spp.'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_BotT_2_auxis.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa 400m Sal vs Sardines
plt.close('all')
fig = plt.figure(3)
ax = pc_so400_NNW.iloc[4].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardineN.plot(ax=ax2, linewidth=1, color='magenta')
sardineC.plot(ax=ax2, linewidth=1, color='tab:green')
sardineS.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('NNW 400m Salinity - PC5')
ax2.legend(['Sardine N', 'Sardine C', 'Sardine S'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_Sal400_5_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Compa EKE (4) vs CHL75 (2)
plt.close('all')
fig = plt.figure(3)
ax = pc_eke_NNW.iloc[3].plot(linewidth=2)
pc_chla75_NNW.iloc[1].plot(linewidth=2, ax=ax)
ax.legend(['EKE (4)', 'Chl-a 75m (2)'])
plt.grid()
#plt.xlim([1987, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'regional_NNW_EKE_chl.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)
