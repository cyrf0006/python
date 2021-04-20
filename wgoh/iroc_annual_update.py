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


## -- Air T Cartwright -- ##
airT = pd.read_pickle('~/AZMP/state_reports/airTemp/airT_anom81.pkl')
airT = airT[['Cartwright']]
airT = airT[airT.index>=1935]
airT.to_csv('iroq_airT_Cartwright.csv', sep=',', float_format='%.2f')
# ** need to manually paste iroq_airT_Cartwright.csv into Newf_Cartwright_Air_Timeseries.csv

## -- CIL data -- ##
# ** need to run iroc_CIL_area.py for 'BB' and 'SI'
df_BB = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/CIL_area_BB.csv')
df_SI = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/CIL_area_SI.csv')
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
