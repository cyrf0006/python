'''

I provide 4 files/timeseries to the IROC:

Newf_Cartwright_Air_Timeseries.csv
Newf_CIL_Area_Timeseries.csv
Newf_SeaIce_Timeseries.csv
Newf_Station27_Annual.csv


'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

clim_years = [1991, 2020]
year = 2022

## -- 1. Air T Cartwright -- ##
airT = pd.read_pickle('~/AZMP/state_reports/airTemp/airT_anom.pkl')
airT = airT[['Cartwright']]
airT = airT[airT.index>=1935]
airT.to_csv('iroq_airT_Cartwright.csv', sep=',', float_format='%.2f')
# ** need to manually paste iroq_airT_Cartwright.csv into Newf_Cartwright_Air_Timeseries.csv
#airT_std = pd.read_pickle('~/AZMP/state_reports/airTemp/airT_std_anom.pkl')


## -- 2. CIL data -- ##
# ** need to run iroc_CIL_area.py for 'BB' and 'SI'
# **** HERE, should load .csv file in ~/AZMP/state_reports/sections_plots/CIL since 2021!! (note from 21 Nov. 2021)
df_BB = pd.read_csv('/home/cyrf0006/AZMP/state_reports/sections_plots/CIL/CIL_area_BB.csv')
df_SI = pd.read_csv('/home/cyrf0006/AZMP/state_reports/sections_plots/CIL/CIL_area_SI.csv')
df_BB.set_index('Unnamed: 0', inplace=True)
df_SI.set_index('Unnamed: 0', inplace=True)
# select interp field
df_BB = df_BB[['interp_field']]
df_SI = df_SI[['interp_field']]
df_BB.rename(columns={"interp_field": "BB"}, inplace=True)
df_SI.rename(columns={"interp_field": "SI"}, inplace=True)
# Concatenate
df = pd.concat([df_BB, df_SI], axis=1)
df.index.name='Year'
df = df.replace({0:np.nan})
# Some manual tweaking to remove very low values
df.loc[df.index==1932] = np.nan
df.loc[df.index==1966] = np.nan
df.loc[df.index==1967] = np.nan
df.BB.loc[df.index<1950]=np.nan
df.BB.loc[df.index==1968]=np.nan
df.BB.loc[df.index==1982]=np.nan
df.SI.loc[df.index==1989]=np.nan
df = df.dropna(how='all')
df.to_csv('iroc_CIL.csv', sep=',', float_format='%.1f')
# ** need to manually paste iroc_CIL.csv into Newf_CIL_Area_Timeseries.csv
# **** Only paste after 1950 (keep historical records before because of mismatch...)
# ***** Need to find a better way by correcting for missing pixels... Maybe by looking at ratio of size vs climatology
## For comparision
df_orig = pd.read_csv('../2017_orig/Newf_CIL_Area_Timeseries.csv', header=12, names=['Year', 'BB', 'SI'])
df_orig.set_index('Year', inplace=True)
df_BB.plot()
df_orig.BB.plot()
plt.title('CIL calculation on BB')
plt.legend(['new calculation', 'previous version'])
plt.grid()

df_SI.plot()
df_orig.SI.plot()
plt.title('CIL calculation on SI')
plt.legend(['new calculation', 'previous version'])
plt.grid()

# To help writing the IROC text:
df_clim = df[(df.index>=clim_years[0]) & (df.index<=clim_years[1])]
df_std_anom = (df - df_clim.mean()) / df_clim.std()
df_std_anom.plot()

## -- 3. Stn 27 data -- ##
#Manually copy-paste iroc_stn27.csv in
# created by azmp_stn27_analysis.py ** check to make sure you have the good climatology!
df_s27 = pd.read_csv('./Newf_Station27_Annual.csv', header=16, names=['Year', 'T', 'Tanom', 'Tstd', 'S', 'Sanom', 'Sstd'])
df_s27.set_index('Year', inplace=True)
df_s27.Tstd.plot()

## -- 4. sea ice data -- ##
# Provided by P.S. Galbraith for 3 regions
df_ice = pd.read_csv('./Newf_SeaIce_Timeseries.csv', header=12, names=['Year', 'NLab', 'SLab', 'NFLD'])
df_ice.set_index('Year', inplace=True)
df_ice_clim = df_ice[(df_ice.index>=clim_years[0]) & (df_ice.index<=clim_years[1])]

df_ice_std_anom = (df_ice - df_ice_clim.mean()) / df_ice_clim.std()
