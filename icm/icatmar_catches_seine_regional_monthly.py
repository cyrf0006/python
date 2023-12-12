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

# NNW Med - Monthly
pc_zos_NNW = pd.read_pickle('./EOFs/scores_zos_deseasonNNW.pkl')
pc_so_NNW = pd.read_pickle('./EOFs/scores_so_deseasonNNW.pkl')
pc_so200_NNW = pd.read_pickle('./EOFs/scores_so_200_deseasonNNW.pkl')
pc_so400_NNW = pd.read_pickle('./EOFs/scores_so_400_deseasonNNW.pkl')
pc_botT_NNW = pd.read_pickle('./EOFs/scores_bottomT_deseasonNNW.pkl')
pc_sst_NNW = pd.read_pickle('./EOFs/scores_thetao_deseasonNNW.pkl')
pc_sst200_NNW = pd.read_pickle('./EOFs/scores_thetao_200_deseasonNNW.pkl')
pc_sst400_NNW = pd.read_pickle('./EOFs/scores_thetao_400_deseasonNNW.pkl')           

## ---- Catch data ---- ##
# Load data
df_landings = pd.read_csv('20231017_Landings_PS_00-22.csv')

# Separate by region
North = df_landings[(df_landings.PortName=='BLANES') | (df_landings.PortName=='ARENYS DE MAR') | (df_landings.PortName=='SANT FELIU DE GUÍXOLS') | (df_landings.PortName=="L'ESCALA")]
Center = df_landings[(df_landings.PortName=='BARCELONA') | (df_landings.PortName=='VILANOVA I LA GELTRÚ') | (df_landings.PortName=='TARRAGONA') | (df_landings.PortName=='MATARÓ') | (df_landings.PortName=='CAMBRILS') | (df_landings.PortName=='ROSES') | (df_landings.PortName=='PORT DE LA SELVA')  | (df_landings.PortName=='LLANÇÀ') | (df_landings.PortName=='PALAMÓS')]
South = df_landings[(df_landings.PortName=='LA RÀPITA') | (df_landings.PortName=="L'AMETLLA DE MAR") | (df_landings.PortName=='DELTEBRE') | (df_landings.PortName=="LES CASES D'ALCANAR")]

# CPUEs
cpue_north = North.groupby(['Date', 'Subgroup', ]).mean()['Average weight (Kg / day * vessel)']
cpue_north = cpue_north.unstack()
cpue_north.index = pd.to_datetime(cpue_north.index)
cpue_north = cpue_north.resample('Q').mean()
cpue_center = Center.groupby(['Date', 'Subgroup', ]).mean()['Average weight (Kg / day * vessel)']
cpue_center = cpue_center.unstack()
cpue_center.index = pd.to_datetime(cpue_center.index)
cpue_center = cpue_center.resample('Q').mean()
cpue_south = South.groupby(['Date', 'Subgroup', ]).mean()['Average weight (Kg / day * vessel)']
cpue_south = cpue_south.unstack()
cpue_south.index = pd.to_datetime(cpue_south.index)
cpue_south = cpue_south.resample('Q').mean()

# Seclect most important species based on value (all regions)
df_value = df_landings.groupby(['Date', 'Subgroup']).mean()['SumAmount_Euros']
df_value = df_value.unstack()
df_value.index = pd.to_datetime(df_value.index)
df_value = df_value.resample('Q').mean()
value20 = df_value.sum()[df_value.sum()>=df_value.sum().sort_values().iloc[-20]].index
value8 = df_value.sum()[df_value.sum()>=df_value.sum().sort_values().iloc[-8]].index

cpue_north = cpue_north[value8.values]
cpue_center = cpue_center[value8.values]
cpue_south = cpue_south[value8.values]

# Rename columns
cpue_north.columns = cpue_north.keys()+' N'
cpue_center.columns = cpue_center.keys()+' C'
cpue_south.columns = cpue_south.keys()+' S'

# Concat regions
df_cpue = pd.concat([cpue_north, cpue_center, cpue_south], axis=1)

## Anomalies
df_anom = (df_cpue - df_cpue.mean()) / df_cpue.std()
#df_anom.index = df_anom.index.year


## ---- Individual speciess---- ##
sardineN = df_anom[['Sardina pilchardus N']].mean(axis=1)
sardineC = df_anom[['Sardina pilchardus C']].mean(axis=1)
sardineS = df_anom[['Sardina pilchardus S']].mean(axis=1)
anchoaN = df_anom[['Engraulis encrasicolus N']].mean(axis=1) 
anchoaC = df_anom[['Engraulis encrasicolus C']].mean(axis=1) 
anchoaS = df_anom[['Engraulis encrasicolus S']].mean(axis=1) 
sardinellaN = df_anom[['Sardinella aurita N']].mean(axis=1) 
#auxisN = df_anom[['Auxis spp. N']].mean(axis=1) 
#auxisC = df_anom[['Auxis spp. C']].mean(axis=1) 
sardine = pd.concat([sardineN, sardineC, sardineS], axis=1).mean(axis=1)

#### ---- NNW Med ---- ####

## Compa Surface Sal vs Sardines
plt.close('all')
fig = plt.figure(3)
ax = pc_so_NNW.iloc[2].resample('Q').mean().plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='magenta')
#sardineC.plot(ax=ax2, linewidth=1, color='tab:green')
#sardineS.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('NNW Surface Salinity - PC3')
ax2.legend(['Sardine'])
plt.ylabel('CPUE anomaly')
plt.grid()
#plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'season_NNW_Sal_3_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Compa Temp 400m vs Sardines
plt.close('all')
fig = plt.figure(3)
ax = pc_sst400_NNW.iloc[1].resample('Q').mean().plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('NNW 400m Temp - PC2')
ax2.legend(['Sardine'])
plt.ylabel('CPUE anomaly')
plt.grid()
#plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'season_NNW_Temp4_2_sardine.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



## Compa Temp 400m vs Anchoa
plt.close('all')
fig = plt.figure(3)
ax = pc_sst400_NNW.iloc[1].resample('Q').mean().plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
anchoaC.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('NNW 400m Temp - PC2')
ax2.legend(['Anchoa C'])
plt.ylabel('CPUE anomaly')
plt.grid()
#plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'season_NNW_Temp4_2_anchoa.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)






