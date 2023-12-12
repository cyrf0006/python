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
pc_so200_NNW = pd.read_pickle('./EOFs/scores_so_200_annual_NNW.pkl')
pc_so400_NNW = pd.read_pickle('./EOFs/scores_so_400_annual_NNW.pkl')
pc_botT_NNW = pd.read_pickle('./EOFs/scores_bottomT_annual_NNW.pkl')
pc_sst_NNW = pd.read_pickle('./EOFs/scores_thetao_annual_NNW.pkl')
pc_sst200_NNW = pd.read_pickle('./EOFs/scores_thetao_200_annual_NNW.pkl')
pc_sst400_NNW = pd.read_pickle('./EOFs/scores_thetao_400_annual_NNW.pkl')                   
## --- Catch data --- ##
# Load data
df_catch = pd.read_csv('Catch composition (season)_ICATMAR.csv')
df_landings = pd.read_csv('20231017_Landings_PS_00-22.csv')

# Subgroups (N=129)
#df_weight = df_landings.groupby(['Date', 'Subgroup']).mean()['Effort (day * vessel)']
df_weight = df_landings.groupby(['Date', 'Subgroup']).mean()['SumWeight_Kg']
df_weight = df_weight.unstack()
df_weight.index = pd.to_datetime(df_weight.index)
df_weight = df_weight.resample('As').mean()

df_value = df_landings.groupby(['Date', 'Subgroup']).mean()['SumAmount_Euros']
df_value = df_value.unstack()
df_value.index = pd.to_datetime(df_value.index)
df_value = df_value.resample('As').mean()

df_cpue = df_landings.groupby(['Date', 'Subgroup']).mean()['Average weight (Kg / day * vessel)']
df_cpue = df_cpue.unstack()
df_cpue.index = pd.to_datetime(df_cpue.index)
df_cpue = df_cpue.resample('As').mean()

# Keep 30 most important
#value30 = df_value.sum()[df_value.sum()>=df_value.sum().sort_values().iloc[-30]].index
#weight30 = df_weight.sum()[df_weight.sum()>=df_weight.sum().sort_values().iloc[-30]].index

#df_value = df_value[value30.values]
#df_weight = df_weight[weight30.values]
# index to year
df_value.index = df_value.index.year
df_weight.index = df_weight.index.year
df_cpue.index = df_cpue.index.year
df.index = df.index.year

## ---- Group species ---- ##
#df_anom = (df_weight - df_weight.mean()) / df_weight.std()
df_anom = (df_cpue - df_cpue.mean()) / df_cpue.std()

## ---- Group species ---- ##
pelagics = df_anom[['Scomber scombrus', 'Sardina pilchardus', 'Micromesistius poutassou', 'Trachurus spp.']].mean(axis=1)
pelagics2 = df_anom[['Mugilidae spp.','Gobidae']].mean(axis=1)
pelagics3 = df_anom[['Diplodus sargus', 'Lepidopus caudatus', 'Scomber scombrus', 'Sardina pilchardus', 'Micromesistius poutassou', 'Trachurus spp.','Trisopterus spp. ']].mean(axis=1)
ground = df_anom[['Lophius spp.', 'Trisopterus spp. ', 'Micromesistius poutassou', 'Eledone spp.']].mean(axis=1)
langoust = df_anom[['Nephrops norvegicus']].mean(axis=1)
shrimp = df_anom[['Parapenaeus longirostris']].mean(axis=1)
sardine = df_anom[['Sardina pilchardus']].mean(axis=1)
mackrel = df_anom[['Scomber scombrus']].mean(axis=1)
whiting = df_anom[['Micromesistius poutassou']].mean(axis=1)
pagellusB = df_anom[['Pagellus bogaraveo']].mean(axis=1)
pagellusE = df_anom[['Pagellus erythrinus']].mean(axis=1)
sardasarda = df_anom[['Sarda sarda']].mean(axis=1)
eledone = df_anom[['Eledone spp.']].mean(axis=1)
sphyraena = df_anom[['Sphyraena sphyraena']].mean(axis=1) #baracuda
sar = df_anom[['Diplodus sargus']].mean(axis=1)
bonito = df_anom[['Sarda sarda']].mean(axis=1)
squilla = df_anom[['Squilla mantis']].mean(axis=1)
rascasse = df_anom[['Helicolenus dactylopterus']].mean(axis=1)
seabream = df_anom[['Lithognathus mormyrus']].mean(axis=1)
trisopterus = df_anom[['Trisopterus spp. ']].mean(axis=1) #small gadidae
lepidopus = df_anom[['Lepidopus caudatus']].mean(axis=1) # Sabre
mullus = df_anom[['Mullus spp.']].mean(axis=1) # Mullet
mugilidae = df_anom[['Mugilidae spp.']].mean(axis=1) # 
anchoa = df_anom[['Engraulis encrasicolus']].mean(axis=1) 
sardinella = df_anom[['Sardinella aurita']].mean(axis=1) 
belone = df_anom[['Belone belone']].mean(axis=1) 
auxis = df_anom[['Auxis spp.']].mean(axis=1) 
calamar = df_anom[['Ommastrephidae spp.']].mean(axis=1) 
dorada = df_anom[['Sparus aurata']].mean(axis=1) # Sabre

#### ---- Whole Med ---- ####

## Compa Med Sal200 vs Diplodus sargus, etc.
plt.close('all')
fig = plt.figure(3)
ax = pc_so200.iloc[1].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sar.plot(ax=ax2, linewidth=1, color='magenta')
calamar.plot(ax=ax2, linewidth=1, color='tab:green')
dorada.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('Med Salinity 200m - PC2')
ax2.legend(['Sar', 'Calamar', 'Dorada'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_medSal200_2_calamar.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Med Sal400 vs Diplodus sargus, etc.
plt.close('all')
fig = plt.figure(3)
ax = pc_so400.iloc[1].plot(linewidth=2)
pc_so400.iloc[2].plot(linewidth=2, ax=ax)
plt.ylabel('PC')
ax2 = ax.twinx()
sar.plot(ax=ax2, linewidth=1, color='magenta')
calamar.plot(ax=ax2, linewidth=1, color='tab:green')
dorada.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('Med Salinity 400m - PC2&3')
ax.legend(['PC2', 'PC3'], loc='upper left')
ax2.legend(['Sar', 'Calamar', 'Dorada'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_MedSal400_2-3_calamar.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Med Temp200 vs Diplodus sargus, etc.
plt.close('all')
fig = plt.figure(3)
ax = pc_sst200.iloc[4].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sar.plot(ax=ax2, linewidth=1, color='magenta')
calamar.plot(ax=ax2, linewidth=1, color='tab:green')
dorada.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('Med Temperature 400m - PC5')
ax2.legend(['Sar', 'Calamar', 'Dorada'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_MedTem200_5_calamar.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Med Temp200 vs Diplodus sargus, etc.
plt.close('all')
fig = plt.figure(3)
ax = pc_botT.iloc[3].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('Med Bottom Temperature - PC4')
ax2.legend(['Sardina'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_MedBotT_4_sardina.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Med Temp200 vs Sardina y  Caballa
plt.close('all')
fig = plt.figure(3)
ax = pc_sst200.iloc[3].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='magenta')
mackrel.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('Med Temperature 200m - PC4')
ax2.legend(['Sardina', 'Caballa'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_MedTem200_4_sardina.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)
#### ---- NW Med ---- ####

## Compa NW BotT (3) vs Anchoa
plt.close('all')
fig = plt.figure(3)
ax = pc_botT_NW.iloc[2].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
anchoa.plot(ax=ax2, linewidth=1, color='magenta')
mackrel.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('NW Bottom T - PC3')
ax2.legend(['Anchoa', 'Caballa'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_NWbotT_anchoa.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa NW Surface S vs Sardinella
plt.close('all')
fig = plt.figure(3)
ax = pc_so_NW.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardinella.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('Surface Salinity - PC1')
ax2.legend(['Sardinella'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_NWsal_1_sardinella.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Compa NW Surface T (3) vs Belone
plt.close('all')
fig = plt.figure(3)
ax = pc_sst_NW.iloc[2].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
belone.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('Surface Temeprature - PC3')
ax2.legend(['Belone belone'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_NWsst_3_belone.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



## Compa Bottom T (2) vs Auxis
plt.close('all')
fig = plt.figure(3)
ax = pc_botT_NW.iloc[1].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
auxis.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('Bottom Temperature - PC2')
ax2.legend(['Auxis spp.'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_NWbotT_2_auxis.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Sal 200m vs Sardina and co.
plt.close('all')
fig = plt.figure(3)
ax = pc_so200_NW.iloc[1].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='magenta')
mackrel.plot(ax=ax2, linewidth=1, color='tab:green')
whiting.plot(ax=ax2, linewidth=1, color='tab:red')
auxis.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('Salinity 200m - PC2')
ax2.legend(['Sardina', 'Caballa', 'Whiting', 'Auxis spp.'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_NWSal2_2_sadina.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Sal 200m vs Sardina y caballa
plt.close('all')
fig = plt.figure(3)
ax = pc_so200_NW.iloc[1].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='magenta')
mackrel.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('Salinity 200m - PC2')
ax2.legend(['Sardina', 'Caballa'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_NWSal2_2_sadina-caballa.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa NNW surface Sal vs Sardina
plt.close('all')
fig = plt.figure(3)
ax = pc_so_NNW.iloc[2].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardine.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('Salinity surface - PC3')
ax2.legend(['Sardina'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_NNWSal_3_sadina.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)





keyboard

