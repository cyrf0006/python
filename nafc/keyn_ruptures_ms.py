'''
Script to test break points in NLCI timeseries.
Preparation to Capelin Keynote 2022
Frederic.Cyr@dfo-mpo.gc.ca

'''

import matplotlib.pyplot as plt
#import ruptures as rpt
import pandas as pd
import numpy as np
import os
## Load climate index
nlci = pd.read_csv('/home/cyrf0006/AZMP/state_reports/climate_index/NL_climate_index.csv')
nlci.set_index('Year', inplace=True)

## Load Capelin biomass data USSR
df_cap2 = pd.read_csv('/home/cyrf0006/data/capelin/Fig2_capelin_biomass_1975_2019.csv', index_col='year')
df_cap2_spring = df_cap2[(df_cap2.area=='spring acoustic') | (df_cap2.area=='ussr spring acoustic')] # spring only
df_cap2_spring = df_cap2[(df_cap2.area=='spring acoustic')] # spring only
#df_cap2_fall = df_cap2[(df_cap2.area=='fall acoustic') | (df_cap2.area=='ussr fall acoustic')] # fall only
# ALL
df_cap2 = df_cap2['biomass ktonnes'].sort_index()
df_cap2 = df_cap2.groupby(['year']).mean()
#df_cap2_diff = df_cap2.interpolate().rolling(5, min_periods=3, center=False).mean().diff()
# SPRING
df_cap2_spring = df_cap2_spring['biomass ktonnes'].sort_index()
df_cap2_spring = df_cap2_spring.groupby(['year']).mean()
df_cap2_spring = df_cap2_spring.interpolate().dropna()

## Load new Capelin index (2023)
df_cap3 = pd.read_csv('/home/cyrf0006/data/capelin/abundance_and_biomass_by_year.csv', index_col='year')
df_cap3 = df_cap3['med.bm.fran.kt']
#df_cap3.drop(1982, inplace=True)
#df_cap3 = df_cap3.interpolate().dropna()

## Load cod biomass data (Schijns et al. 2021)
df_cod = pd.read_csv('fig_S11D_data.csv', index_col='year')
Redline = df_cod.RedLine
Redline = Redline.iloc[Redline.index>=1950]*10

## Load crab data (Mullowney)
## df_crab = pd.read_excel('biomasses.xlsx')
## df_crab = df_crab[df_crab.Region=='NL']
## df_crab.set_index('Year', inplace=True)
## df_crab = df_crab.tBIO/100000

df_crab = pd.read_csv('crab_unfished.csv')
# drop 3Ps
#df_crab = df_crab[df_crab.division!='3PS']
df_crab = df_crab[df_crab.age<4]
# average all years
df_crab = df_crab.groupby('year').mean()['Cmnpt']

df_crablobster = pd.read_csv('Crab_lobster_biomass.csv')
df_crablobster.set_index('Year', inplace=True)
df_crab2 = df_crablobster['crabexpbio']
df_lobster = df_crablobster['lobsterlandings']
df_crab2.dropna(inplace=True)
df_lobster.dropna(inplace=True)

# Load Mariano's biomass density (t/km2)
df_bio = pd.read_excel(open('RV_biomass_density.xlsx', 'rb'), sheet_name='data_only')
df_bio.set_index('Year', inplace=True)
df_bio_ave = df_bio['Average-ish biomass density'].dropna()

# Load Mariano's Total Catches (tonnes)
df_catch = pd.read_excel(open('NL_Catch_Data.xlsx', 'rb'), sheet_name='data_only')
df_catch.set_index('Year', inplace=True)
#df_catch_ave = df_catch['Total Result'].dropna()
#df_catch_ave = df_catch[['Piscivore', 'Planktivore']].mean(axis=1).dropna()
df_catch_ave = df_catch[['Piscivore']].mean(axis=1).dropna()
df_catch_ave = df_catch_ave/1000 #(kt)

## df_crab = df_crab[df_crab.Region=='NL']
## df_crab.set_index('Year', inplace=True)
## df_crab = df_crab.tBIO/100000

# Load Groundfish 
df_ncam = pd.read_csv('/home/cyrf0006/data/NCAM/multispic_process_error.csv', index_col='year')
df_cod = df_ncam[df_ncam.species=='Atlantic Cod']
df_had = df_ncam[df_ncam.species=='Haddock']
df_hak = df_ncam[df_ncam.species=='White Hake']
df_pla = df_ncam[df_ncam.species=='American Plaice']
df_red = df_ncam[df_ncam.species=='Redfish spp.']
df_ska = df_ncam[df_ncam.species=='Skate spp.']
df_wit = df_ncam[df_ncam.species=='Witch Flounder']
df_yel = df_ncam[df_ncam.species=='Yellowtail Flounder']
df_tur = df_ncam[df_ncam.species=='Greenland Halibut']
df_wol = df_ncam[df_ncam.species=='Wolffish spp.']
# drop 3Ps
df_ncam = df_ncam[df_ncam.region!='3Ps']
# average all years
#df_ncam = df_ncam.groupby('year').mean()['process_error']
df_ncam = df_ncam.groupby('year').sum()['delta_biomass_kt']

# Load Primary Production
df_PP = pd.read_csv('PP/cmems_PP.csv')
df_PP.set_index('time', inplace=True)

# Breaking points based on the NLCI:
years_list = [
[1948, 1971],
[1971, 1976],
[1976, 1982],
[1982, 1998],
[1998, 2014],
[2014, 2017],
[2017, 2021]
]

nlci.rolling(2).mean().cumsum().to_csv('NLCI_cumsum.csv')  
ax = nlci.rolling(2).mean().cumsum().plot()
Redline.plot(color='red', ax=ax)
df_crab.rolling(5).mean().plot(color='orange', ax=ax)
df_bio_ave.plot(color='green', ax=ax)
ax2 = ax.twinx()
#df_cap2_spring.interpolate().plot(ax=ax2, color='blue')
df_cap3.interpolate().plot(ax=ax2, color='blue')
ax3 = ax2.twinx()
df_PP.plot(ax=ax3, color='magenta')
plt.legend(['DFO Spring survey'], loc='upper left')
ax.set_ylabel('NLCI cumsum')                                               
ax2.set_ylabel('Biomass (ktonnes)')                                               
ax.set_xlim([1950, 2020])                                               


for years in years_list:
    plt.plot([years[0], years[0]], [0, 6000], '--k')

plt.grid()



##  ---- Figures for presentation ---- ##

## PP
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
df_PP.plot(ax=ax2, color='Green', linewidth=3, alpha=.6)
ax2.legend(['Net Primary Production'])
plt.ylabel(r'PP ($\rm mgC\,m^{-3}\,d^{-1}$)', color='green')
plt.title('Global ocean biogeochemistry hindcast')
ax.set_xlim([1975, 2020])
for years in years_list[:-1]:
    PP_tmp = df_PP[(df_PP.index>=years[0]) & (df_PP.index<=years[1])]
    if len(PP_tmp)>0:
        ax2.plot([years[0], years[1]], [PP_tmp.mean().values, PP_tmp.mean().values], linestyle='--', color='green', linewidth=2)
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'PP_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

## Zooplankton
# Anomalies
df_cal = pd.read_csv('/home/cyrf0006/research/PeopleStuff/BelangerStuff/CorrelationData.csv', index_col='Year')
df_cal = df_cal['calfin']
# Abundance
df_cal_ab = pd.read_csv('/home/cyrf0006/research/PeopleStuff/BelangerStuff/CALFIN_abundance.csv', index_col='Year')
df_cal_ab = df_cal_ab['Mean abundance (log10 ind. m-2)']

fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
df_cal_ab.plot(ax=ax2, color='tab:brown', linewidth=3, alpha=.6)
ax2.legend(['Calfin'])
plt.ylabel(r'$\rm log_{10}(ind\,m^{-2})$', color='tab:brown')
plt.title('2J3KLNO Calanus finmarchicus abundance')
ax2.set_ylim([8.5, 9.5])
for years in years_list:
    cal_tmp = df_cal_ab[(df_cal_ab.index>=years[0]) & (df_cal_ab.index<=years[1])]
    if len(cal_tmp)>0:
        ax2.plot([years[0], years[1]], [cal_tmp.mean(), cal_tmp.mean()], linestyle='--', color='tab:brown', linewidth=2)
ax.set_xlim([1975, 2021])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'calanus_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## Biomass density
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
df_bio_ave.plot(ax=ax2, color='red', linewidth=3, alpha=.6)
ax2.legend(['multispecies'])
plt.ylabel(r'biomass density ($\rm t\,km^{-2}$)', color='red')
plt.title('Multispecies scientific trawl survey')
ax2.set_ylim([0, 23])
for years in years_list[3:-1]:
    bio_tmp = df_bio_ave[(df_bio_ave.index>=years[0]) & (df_bio_ave.index<=years[1])]
    if len(bio_tmp)>0:
        pp = np.polyfit(bio_tmp.index, bio_tmp.values, 1)
        yyyy = np.arange(years[0], years[1]+1)
        ax2.plot(yyyy, yyyy*pp[0] + pp[1], linestyle='--', color='red', linewidth=2)
ax.set_xlim([1975, 2020])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'trawl_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

## Total catches
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
df_catch_ave.plot(ax=ax2, color='indianred', linewidth=3, alpha=.6)
ax2.legend(['Statlan21'])
plt.ylabel(r'Catches ($\rm kt$)', color='indianred')
plt.title('NAFO Statlan21 Catches (NL)')
#ax2.set_ylim([0, 23])
for years in years_list[0:-1]:
    catch_tmp = df_catch_ave[(df_catch_ave.index>=years[0]) & (df_catch_ave.index<=years[1])]
    if len(catch_tmp)>0:
        pp = np.polyfit(catch_tmp.index, catch_tmp.values, 1)
        yyyy = np.arange(years[0], years[1]+1)
        ax2.plot(yyyy, yyyy*pp[0] + pp[1], linestyle='--', color='indianred', linewidth=2)
ax.set_xlim([1960, 2022])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'catch_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## Capelin
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
#df_cap2_spring.plot(ax=ax2, color='tab:blue', linewidth=3, alpha=.6)
#df_cap3.interpolate().plot(ax=ax2, color='tab:blue', linewidth=3, marker='.', alpha=.6)
df_cap3.interpolate().plot(ax=ax2, color='tab:blue', linewidth=3, alpha=.6)
ax2.legend(['Capelin'])
plt.ylabel(r'Biomass ($\rm kt$)', color='tab:blue')
plt.title('Capelin Spring Acoustic Survey')
ax2.set_ylim([0, 6000])
for years in years_list[:-1]:
    cap_tmp = df_cap2_spring[(df_cap2_spring.index>=years[0]) & (df_cap2_spring.index<=years[1])]
    if len(cap_tmp)>0:
        pp = np.polyfit(cap_tmp.index, cap_tmp.values, 1)
        yyyy = np.arange(years[0], years[1]+1)
        ax2.plot(yyyy, yyyy*pp[0] + pp[1], linestyle='--', color='tab:blue', linewidth=2)
ax.set_xlim([1975, 2020])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'capelin_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## Groundfish
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
df_ncam.plot(ax=ax2, color='tab:orange', linewidth=3, alpha=.6)
ax2.legend(['Groundfish'])
plt.ylabel(r'Excess biomass ($\rm kt$)', color='tab:orange')
plt.title('Groundfish surplus production model')
#ax2.set_ylim([0, 6000])
for years in years_list[:-1]:
    gf_tmp = df_ncam[(df_ncam.index>=years[0]) & (df_ncam.index<=years[1])]
    if len(gf_tmp)>0:
        pp = np.polyfit(gf_tmp.index, gf_tmp.values, 1)
        yyyy = np.arange(years[0], years[1]+1)
        ax2.plot(yyyy, yyyy*pp[0] + pp[1], linestyle='--', color='tab:orange', linewidth=2)
ax.set_xlim([1975, 2020])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'groundfish_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

## Nutrients
df_monthly = pd.read_pickle('/home/cyrf0006/AZMP/btl_data/monthly_nutrients.pkl') #see cs_extract_SInutrients.py
#df_nut = df[['Nitrate', 'Silicate', 'Phosphate']].mean(axis=1)
df_nuts = df_monthly[['Nitrate', 'Silicate']].resample('As').mean()
df_nuts.index = df_nuts.index.year
df_nut = df_nuts.mean(axis=1)

plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
df_nuts.Nitrate.dropna().plot(ax=ax2, marker='.', linestyle=' ', alpha=.6)    
df_nuts.Silicate.dropna().plot(ax=ax2, marker='.', linestyle=' ', alpha=.6)
df_nuts.Nitrate.dropna().rolling(5, center=True).mean().plot(ax=ax2, color='steelblue', linewidth=3, alpha=.6)
df_nuts.Silicate.dropna().rolling(5, center=True).mean().plot(ax=ax2, color='tab:orange', linewidth=3, alpha=.6) 
ax2.legend(['NO3', 'SiO', 'NO3 5yr', 'SiO 5yr'])
plt.ylabel(r'Concentration ($mmol\,m^{-3}$)')
plt.title('Nutrients - Southern Labrador Shelf (50-150m)')
## for years in years_list:
##     nut_tmp = df_nut[(df_nut.index>=years[0]) & (df_nut.index<=years[1])]
##     if len(nut_tmp)>0:
##         ax2.plot([years[0], years[1]], [nut_tmp.mean(), nut_tmp.mean()], linestyle='--', color='tab:brown', linewidth=2)
ax.set_xlim([1995, 2020])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'nutrients_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## Crab
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
df_crab2.plot(ax=ax2, color='tab:orange', linewidth=3, alpha=.6)
ax2.legend(['Snow Crab'])
plt.ylabel(r'Biomass ($\rm kt$)', color='tab:orange')
plt.title('Snow Crab Exploitable biomass')
#ax2.set_ylim([0, 6000])
for years in years_list[:-1]:
    gf_tmp = df_crab2[(df_crab2.index>=years[0]) & (df_crab2.index<=years[1])]
    if len(gf_tmp)>0:
        pp = np.polyfit(gf_tmp.index, gf_tmp.values, 1)
        yyyy = np.arange(years[0], years[1]+1)
        ax2.plot(yyyy, yyyy*pp[0] + pp[1], linestyle='--', color='tab:orange', linewidth=2)
ax.set_xlim([1975, 2020])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'snowcrab_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## Lobster
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
nlci.rolling(2).mean().cumsum().plot(ax=ax, linewidth=3, alpha=.6, color='magenta')
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI (cumsum)', color='magenta')
for years in years_list:
    plt.plot([years[0], years[0]], [-1, 12], '--k')
ax.set_ylim([-1, 12])
ax2 = ax.twinx()
df_lobster.plot(ax=ax2, color='tab:red', linewidth=3, alpha=.6)
ax2.legend(['Lobster'])
plt.ylabel(r'Landings ($\rm kt$)', color='tab:red')
plt.title('Lobster Landings')
#ax2.set_ylim([0, 6000])
for years in years_list[:-1]:
    gf_tmp = df_lobster[(df_lobster.index>=years[0]) & (df_lobster.index<=years[1])]
    if len(gf_tmp)>0:
        pp = np.polyfit(gf_tmp.index, gf_tmp.values, 1)
        yyyy = np.arange(years[0], years[1]+1)
        ax2.plot(yyyy, yyyy*pp[0] + pp[1], linestyle='--', color='tab:red', linewidth=2)
ax.set_xlim([1975, 2020])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'lobster_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)
