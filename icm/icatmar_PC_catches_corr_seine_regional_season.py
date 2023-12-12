'''
[THIS IS NOT THE CODE
NEED TO COMPUTE MONTHLY EOFS
]


Test correlation between CPUEs in different ports of Catalonia and Regional EOFs in the NNW Med.

Frederic.Cyr@dfo-mpo.gc.ca
October 2023

'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import seaborn as sn
import cmocean as cmo
from matplotlib.colors import from_levels_and_colors

## Options:
# Series 1 (zos, temp 0-400, sal 0-400) or Series 2 (temp/sal 75, MLD, EKE, chla)
series1 = True

## ---- Catch data ---- ##
# Load data
df_landings = pd.read_csv('20231017_Landings_PS_00-22.csv')

# Separate by region
North = df_landings[(df_landings.PortName=='BLANES') | (df_landings.PortName=='ARENYS DE MAR') | (df_landings.PortName=='SANT FELIU DE GUÍXOLS')]
Center = df_landings[(df_landings.PortName=='BARCELONA') | (df_landings.PortName=='VILANOVA I LA GELTRÚ') | (df_landings.PortName=='TARRAGONA') | (df_landings.PortName=='MATARÓ') | (df_landings.PortName=='CAMBRILS') | (df_landings.PortName=='ROSES') | (df_landings.PortName=='PALAMÓS') | (df_landings.PortName=="L'ESCALA") | (df_landings.PortName=='PORT DE LA SELVA') | (df_landings.PortName=='LLANÇÀ')]
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
df_value = df_value.resample('As').mean()
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

## Drop some values if a lot of NaNs
# correlation matric and pvalues
df_anom = df_anom.dropna(thresh=80, axis=1)
# Find shape 
L = df_anom.shape[1]


## ---- EOFs / PCs data ---- ##
pc_zos = pd.read_pickle('./EOFs/scores_zos_deseasonNNW.pkl')
pc_zos.index = ['SSH_01','SSH_02', 'SSH_03', 'SSH_04', 'SSH_05']

pc_so = pd.read_pickle('./EOFs/scores_so_deseasonNNW.pkl')
pc_so.index = ['SAL_01','SAL_02', 'SAL_03', 'SAL_04', 'SAL_05']

pc_so075 = pd.read_pickle('./EOFs/scores_so_075_deseasonNNW.pkl')
pc_so075.index = ['SAL75_01','SAL75_02', 'SAL75_03', 'SAL75_04', 'SAL75_05']

pc_so200 = pd.read_pickle('./EOFs/scores_so_200_deseasonNNW.pkl')
pc_so200.index = ['SAL2_01','SAL2_02', 'SAL2_03', 'SAL2_04', 'SAL2_05']

pc_so400 = pd.read_pickle('./EOFs/scores_so_400_deseasonNNW.pkl')
pc_so400.index = ['SAL4_01','SAL4_02', 'SAL4_03', 'SAL4_04', 'SAL4_05']

pc_botT = pd.read_pickle('./EOFs/scores_bottomT_deseasonNNW.pkl')
pc_botT.index = ['BOT_01','BOT_02', 'BOT_03', 'BOT_04', 'BOT_05']

pc_tem = pd.read_pickle('./EOFs/scores_thetao_deseasonNNW.pkl')
pc_tem.index = ['TEM_01','TEM_02', 'TEM_03', 'TEM_04', 'TEM_05']

pc_tem075 = pd.read_pickle('./EOFs/scores_thetao_075_deseasonNNW.pkl')
pc_tem075.index = ['TEM75_01','TEM75_02', 'TEM75_03', 'TEM75_04', 'TEM75_05']

pc_tem200 = pd.read_pickle('./EOFs/scores_thetao_200_deseasonNNW.pkl')
pc_tem200.index = ['TEM2_01','TEM2_02', 'TEM2_03', 'TEM2_04', 'TEM2_05']

pc_tem400 = pd.read_pickle('./EOFs/scores_thetao_400_deseasonNNW.pkl')
pc_tem400.index = ['TEM4_01','TEM4_02', 'TEM4_03', 'TEM4_04', 'TEM4_05']

pc_eke = pd.read_pickle('./EOFs/scores_EKE_deseasonNNW.pkl')
pc_eke.index = ['EKE_01','EKE_02', 'EKE_03', 'EKE_04', 'EKE_05']

pc_mld = pd.read_pickle('./EOFs/scores_MLD_deseasonNNW.pkl')
pc_mld.index = ['MLD_01','MLD_02', 'MLD_03', 'MLD_04', 'MLD_05']

pc_chl = pd.read_pickle('./EOFs/scores_chla_deseasonNNW.pkl')
pc_chl.index = ['CHL_01','CHL_02', 'CHL_03', 'CHL_04', 'CHL_05']

pc_chl075 = pd.read_pickle('./EOFs/scores_chla75_deseasonNNW.pkl')
pc_chl075.index = ['CHL75_01','CHL75_02', 'CHL75_03', 'CHL75_04', 'CHL75_05']

## ---- Concat data ---- ##
if series1:
    PC_CPUE = pd.concat([df_anom, pc_zos.T.resample('Q').mean(), pc_so.T.resample('Q').mean(), pc_so200.T.resample('Q').mean(), pc_so400.T.resample('Q').mean(),pc_botT.T.resample('Q').mean(),pc_tem.T.resample('Q').mean(),pc_tem200.T.resample('Q').mean(),pc_tem400.T.resample('Q').mean()], axis=1)
else:
    PC_CPUE = pd.concat([df_anom, pc_so.T.resample('Q').mean(), pc_so075.T.resample('Q').mean(), pc_botT.T.resample('Q').mean(), pc_tem075.T.resample('Q').mean(), pc_eke.T.resample('Q').mean(),pc_mld.T.resample('Q').mean(),pc_chl.T.resample('Q').mean(), pc_chl075.T.resample('Q').mean()], axis=1)

PC_CPUE = PC_CPUE[PC_CPUE.index>='2000-01-31']
PC_CPUE = PC_CPUE[PC_CPUE.index<='2021-01-31']


## ---- Plot correlation ---- ##
from scipy.stats import pearsonr
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

# Build the colormap (need to color >+-.3)
vmin = -1.0
vmax = 1.0
midpoint = 0.0
levels = np.linspace(vmin, vmax, 21)
midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
colvals = np.interp(midp, [vmin, midpoint, vmax], [-1, 0., 1])
normal = plt.Normalize(-1.0, 1.0)
reds = plt.cm.Reds(np.linspace(0,1, num=6))
blues = plt.cm.Blues_r(np.linspace(0,1, num=6))
whites = [(.95,.95,.95,.95)]*9
colors = np.vstack((blues[0:-1,:], whites, reds[1:,:]))
colors = np.concatenate([[colors[0,:]], colors, [colors[-1,:]]], 0)
cmap, norm = from_levels_and_colors(midp, colors, extend='both')
# By-pass:
cmap = cmo.cm.balance

# correlation matrix and pvalues
corrMatrix = PC_CPUE.corr().round(2)
pvalues = calculate_pvalues(PC_CPUE)
## Restrict correlation matrix
corrMatrix = corrMatrix.iloc[0:L,L:,]
pvalues = pvalues.iloc[0:L,L:,]
# Text
annot_text  = corrMatrix.astype('str')
corrMatrix_text = corrMatrix.copy()

for i in np.arange(pvalues.shape[0]):
    for j in np.arange(pvalues.shape[1]):
        if pvalues.iloc[i,j]>=.05:
            #annot_text.iloc[i,j] = annot_text.iloc[i,j]+'*'
            corrMatrix.iloc[i,j] = 0            
            corrMatrix_text.iloc[i,j] = ' '
            
plt.close('all')
fig = plt.figure(3)
sn.heatmap(corrMatrix, annot=corrMatrix_text.astype('str'), fmt='s', linewidths=.2, cmap=cmap, cbar=None, vmin=-1.05, vmax=1.05)
plt.title('Pearson correlation coefficients')
# tweak yticklabels
LABELS = corrMatrix.keys().to_list()
ax = plt.gca()
LL = corrMatrix.shape[1]
XTICKS = np.arange(0.5, LL+.5, 1)
plt.xticks(XTICKS)
ax.set_xticklabels(LABELS)
fig.set_size_inches(w=20,h=14)
if series1:
    fig_name = 'Correlation_PC-CPUE_PS_regional_season.png'
else:
    fig_name = 'Correlation_PC-CPUE_PS_regional_season_series2.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

