'''
CIL area

This script is still in progress...


'''
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np


## One both are generated:
df_BB = pd.read_csv('CIL_area_BB.csv')
df_SI = pd.read_csv('CIL_area_SI.csv')

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

# [some manual work to do here to add to document]

df_orig = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/2019_update/Newf_CIL_Area_Timeseries.csv', header=12)
df_old = pd.read_csv('/home/cyrf0006/research/WGOH/IROC/2018_update/Newf_CIL_Area_Timeseries.csv', header=12)
df_orig.set_index(['Decimal Year'], inplace=True)
df_old.set_index(['Decimal Year'], inplace=True)

# Compare plots
ax = df_orig['Newfoundland CIL Area (km^2)'].plot()
df_old['Newfoundland CIL Area (km^2)'].plot(ax=ax)
ax.set_ylabel(r'$\rm Area_{CIL} (km^2)$', fontWeight = 'bold')
plt.legend(['New way', 'Old way'])
plt.savefig('compare_CILmethod_BB.png', dpi=150)
plt.close()

ax = df_orig['Labrador CIL Area (km^2)'].plot()
df_old['Labrador CIL Area (km^2)'].plot(ax=ax)
ax.set_ylabel(r'$\rm Area_{CIL} (km^2)$', fontWeight = 'bold')
plt.legend(['New way', 'Old way'])
plt.savefig('compare_CILmethod_SI.png', dpi=150)
plt.close()
