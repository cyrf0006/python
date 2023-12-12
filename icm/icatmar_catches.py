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

pc_zos = pd.read_pickle('./EOFs/scores_zos_annual_NNW.pkl')
pc_so = pd.read_pickle('./EOFs/scores_so_annual_NNW.pkl')
pc_so200 = pd.read_pickle('./EOFs/scores_so_200_annual_NNW.pkl')
pc_so400 = pd.read_pickle('./EOFs/scores_so_400_annual_NNW.pkl')
pc_botT = pd.read_pickle('./EOFs/scores_bottomT_annual_NNW.pkl')
pc_sst = pd.read_pickle('./EOFs/scores_thetao_annual_NNW.pkl')
pc_sst200 = pd.read_pickle('./EOFs/scores_thetao_200_annual_NNW.pkl')
pc_sst400 = pd.read_pickle('./EOFs/scores_thetao_400_annual_NNW.pkl')

pc_eke = pd.read_pickle('./EOFs/scores_EKE_annual_NNW.pkl')

                       
## --- Catch data --- ##
# Load data
df_catch = pd.read_csv('Catch composition (season)_ICATMAR.csv')
df_landings = pd.read_csv('20230921_landingsCatalunya_00-22.csv')

df_hake = pd.read_csv('Landings HKE 1988 2022.csv', sep=';')
df_hake.set_index(' timeC', inplace=True)
df_hake.rename_axis('year', inplace=True)
df_hake.rename(columns={'obsC':'hake'}, inplace=True)

# Groups (N=9)
#df_weight = df_landings.groupby(['Date', 'Group']).mean()['SumWeight_Kg']
#df_weight = df_weight.unstack()
#df_weight.index = pd.to_datetime(df_weight.index)
#df_weight = df_weight.resample('As').mean()

# Subgroups (N=174)
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
mullus = df_anom[['Mullus spp.']].mean(axis=1) # Sabre
mugilidae = df_anom[['Mugilidae spp.']].mean(axis=1) # Sabre


# group by classes
df_class = pd.read_csv('Classification.csv', sep=';')
Actiniidae = df_anom[df_class[df_class.Class=='Actiniidae'].Name.to_list()].mean(axis=1)
Ascidiacea = df_anom[df_class[df_class.Class=='Ascidiacea'].Name.to_list()].mean(axis=1)
Bivalvia = df_anom[df_class[df_class.Class=='Bivalvia'].Name.to_list()].mean(axis=1)
Cephalopoda = df_anom[df_class[df_class.Class=='Cephalopoda'].Name.to_list()].mean(axis=1)
Chondrichthyes = df_anom[df_class[df_class.Class=='Chondrichthyes'].Name.to_list()].mean(axis=1)
Echinoidea = df_anom[df_class[df_class.Class=='Echinoidea'].Name.to_list()].mean(axis=1)
Elasmobranchii = df_anom[df_class[df_class.Class=='Elasmobranchii'].Name.to_list()].mean(axis=1)
Gastropoda = df_anom[df_class[df_class.Class=='Gastropoda'].Name.to_list()].mean(axis=1)
Holocephali = df_anom[df_class[df_class.Class=='Holocephali'].Name.to_list()].mean(axis=1)
Holothuroidea = df_anom[df_class[df_class.Class=='Holothuroidea'].Name.to_list()].mean(axis=1)
Malacostraca = df_anom[df_class[df_class.Class=='Malacostraca'].Name.to_list()].mean(axis=1)
Teleostei = df_anom[df_class[df_class.Class=='Teleostei'].Name.to_list()].mean(axis=1)

# Concat classes
df_class_anom = pd.concat([Actiniidae, Ascidiacea, Bivalvia, Cephalopoda, Chondrichthyes, Echinoidea, Elasmobranchii, Gastropoda, Holocephali, Holothuroidea, Malacostraca, Teleostei], axis=1)
# Rename
df_class_anom.columns=['Actiniidae', 'Ascidiacea', 'Bivalvia', 'Cephalopoda','Chondrichthyes', 'Echinoidea', 'Elasmobranchii', 'Gastropoda','Holocephali', 'Holothuroidea', 'Malacostraca', 'Teleostei']

## --- plot --- ##
plt.close('all')
ax = df.plot(linewidth=3)
#df.rolling(24, center=True).mean().plot()
pelagics.plot(ax=ax)
pelagics2.plot(ax=ax)
ground.plot(ax=ax)
langoust.plot(ax=ax)
shrimp.plot(ax=ax)
plt.legend(['SLP', 'Pelagics1','Pelagics2','Groundfish','Langoust','Shrimp'])
plt.grid()
plt.show()

plt.close('all')
ax = df.plot(linewidth=3)
sardine.plot(ax=ax)
mackrel.plot(ax=ax)
whiting.plot(ax=ax)
plt.legend(['SLP', 'Sardine', 'Mackrel', 'Whiting'])
plt.grid()
plt.show()


# Set new time index 
time = df_catch.Season
time.replace('Winter', '02-15', inplace=True)
time.replace('Spring', '05-15', inplace=True)
time.replace('Summer', '08-15', inplace=True)
time.replace('Autumn', '11-15', inplace=True)
Date = pd.to_datetime(df_catch.Year.astype('str') + '-' + time)
df_catch['Date'] = Date
df_catch.set_index('Date', inplace=True)
df_catch.drop(columns=['Year', 'Season'], inplace=True)

landed = df_catch[df_catch.Classification == 'Landed']
landed.groupby('Date').mean().plot(ax=ax)

plt.show()



## Compa Sal-1 vs some pelagics
plt.close('all')
fig = plt.figure(3)
ax = pc_so.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardasarda.plot(ax=ax2, linewidth=1, color='magenta')
pagellusB.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('Surface Salinity - PC1')
ax2.legend(['Sarda sarda', 'Pagellus B.'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_sal1_sardasarda.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Sal200-1 vs some species
plt.close('all')
fig = plt.figure(3)
ax = pc_so200.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
lepidopus.plot(ax=ax2, linewidth=1, color='magenta')
rascasse.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('200m Salinity - PC1')
ax2.legend(['Lepidopus', 'Scorpiofish'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_sal200_1_scorpio.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Sal200-1 vs some species
plt.close('all')
fig = plt.figure(3)
ax = pc_so200.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
mugilidae.plot(ax=ax2, linewidth=1, color='magenta')
mullus.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('200m Salinity - PC1')
ax2.legend(['Mugilidae spp.', 'Mullus spp.'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_sal200_1_mullus.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



## Compa Sal400-1 vs some squilla mantis...
plt.close('all')
fig = plt.figure(3)
ax = pc_so400.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
squilla.plot(ax=ax2, linewidth=1, color='magenta')
eledone.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('400m Salinity - PC1')
ax2.legend(['Squilla mantis', 'Eledone spp.'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_sal400_1_squilla.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Sal400-1 vs some species
plt.close('all')
fig = plt.figure(3)
ax = pc_so400.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
rascasse.plot(ax=ax2, linewidth=1, color='magenta')
seabream.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('400m Salinity - PC1')
ax2.legend(['Scorpiofish', 'Seabream'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_sal400_1_scorpio.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Sal400-2 vs Norvegicus
plt.close('all')
fig = plt.figure(3)
ax = pc_so400.iloc[1].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
langoust.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('400m Salinity - PC2')
ax2.legend(['Nephrops norvegicus'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_sal400_1_norvegicus_NNW.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Tem200-1 vs Norvegicus
plt.close('all')
fig = plt.figure(3)
ax = pc_sst200.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
langoust.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('200m Temperature - PC1')
ax2.legend(['Nephrops norvegicus'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_tem200_1_norvegicus.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Tem400-1 vs Norvegicus
plt.close('all')
fig = plt.figure(3)
ax = pc_sst400.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
langoust.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('400m Temperature - PC1')
ax2.legend(['Nephrops norvegicus'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_tem400_1_norvegicus.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa Sal400-2 vs Norvegicus
plt.close('all')
fig = plt.figure(3)
ax = pc_so400.iloc[1].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
langoust.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('400m Salinity - PC2')
ax2.legend(['Nephrops norvegicus'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_sal400_1_norvegicus.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa EKE vs Norvegicus
plt.close('all')
fig = plt.figure(3)
ax = pc_eke.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
langoust.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('Surface EKE - PC1')
ax2.legend(['Nephrops norvegicus'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_EKE_1_norvegicus.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## Compa EKE vs Norvegicus + Rose srhimp
plt.close('all')
fig = plt.figure(3)
ax = pc_eke.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
langoust.plot(ax=ax2, linewidth=1, color='magenta')
shrimp.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('Surface EKE - PC1')
ax2.legend(['Nephrops', 'Parapenaeus'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_EKE_1_norvegicus_shrimp.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)


## Compa Bottom T vs some pelagics and barracuda
plt.close('all')
fig = plt.figure(3)
ax = pc_botT.iloc[3].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
pelagics3.plot(ax=ax2, linewidth=1, color='magenta')
sphyraena.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('Bottom T - PC4')
ax2.legend(['Pelagics3', 'Sphyraena'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_botT_4_pelagics3.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



## Compa Temp400-1 and Sal400-1 vs Hake
plt.close('all')
fig = plt.figure(3)
ax = pc_sst400.iloc[0].plot(linewidth=2)
pc_so400.iloc[0].plot(linewidth=2, ax=ax)
ax2 = ax.twinx()
df_hake.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('T/S 400m - PC1')
ax2.legend(['Hake'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1987, 2022])
fig.set_size_inches(w=6,h=4)
fig_name = 'PCcompa_TS400_1_hake.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)






## Compa Sal-1 vs some pelagics
plt.close('all')
ax = pc_so.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardasarda.plot(ax=ax2, linewidth=1, color='magenta')
pagellusB.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('Surface Salinity - PC1')
ax2.legend(['Sarda sarda', 'Pagellus B.'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
plt.show()

pel = pd.concat([pc_so.iloc[0].T, pelagics], axis=1)
plt.close('all')
plt.scatter(pel[[1]], pel[[0]], c=pel.index)
plt.xlabel('SSS - PC1')
plt.ylabel('Pelagics CPUE')
plt.grid()
plt.show()


## Compa Sal-1 vs some pelagics
plt.close('all')
ax = pc_so.iloc[0].plot(linewidth=2)
plt.ylabel('PC')
ax2 = ax.twinx()
sardasarda.plot(ax=ax2, linewidth=1, color='magenta')
pagellusB.plot(ax=ax2, linewidth=1, color='tab:green')
plt.title('Surface Salinity - PC1')
ax2.legend(['Sarda sarda', 'Pagellus B.'])
plt.ylabel('CPUE anomaly')
plt.grid()
plt.xlim([1998, 2022])
plt.show()

## Compa Temp400-1 and Sal400-1 vs Hake
plt.close('all')
ax = pc_sst400.iloc[0].plot(linewidth=2)
pc_so400.iloc[0].plot(linewidth=2, ax=ax)
ax2 = ax.twinx()
hake.plot(ax=ax2, linewidth=1, color='tab:green')
ax.legend(['T400', 'S400'])
ax2.legend(['Hake'])
plt.grid()
plt.show()

## Compa SST1 Eledone
plt.close('all')
ax = pc_SST.iloc[0].plot(linewidth=2)
ax2 = ax.twinx()
eledone.plot(ax=ax2, linewidth=1, color='tab:orange')
plt.title('SST - PC 1')
ax2.legend('Eledone')
plt.grid()
plt.show()

## Compa BotT1 
plt.close('all')
ax = pc_botT.iloc[0].plot(linewidth=2)
ax2 = ax.twinx()
eledone.plot(ax=ax2, linewidth=1, color='tab:orange')
sar.plot(ax=ax2, linewidth=1, color='tab:green')
sphyraena.plot(ax=ax2, linewidth=1, color='magenta')
plt.title('SST - PC 1')
ax2.legend(['Eledone', 'Diplodus sargus', 'Sphyraena'])
plt.grid()
plt.show()
