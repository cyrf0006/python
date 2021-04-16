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
# ** need to manually paste iroq_airT_Cartwright.csn into Newf_Cartwright_Air_Timeseries.csv

## -- CIL data -- ##
# ** need to run iroc_CIL_area.py for 'BB' and 'SI'
df_BB = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/CIL_area_BB.csv')
df_SI = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/CIL_area_SI.csv')
# merge into one
df = pd.DataFrame(index=df_BB['Unnamed: 0'], columns=['BB', 'SI'])
df.index.name='Year'
df.BB = df_BB.interp_field.values
df.SI = df_SI.interp_field.values
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
