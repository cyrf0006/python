
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import seaborn as sn
import cmocean as cmo
from matplotlib.colors import from_levels_and_colors


## ---- Catch data ---- ##
# Load data
df_landings = pd.read_csv('20230921_landingsCatalunya_00-22.csv')

df_cpue = df_landings.groupby(['Date', 'Subgroup']).mean()['Average weight (Kg / day * vessel)']
df_cpue = df_cpue.unstack()
df_cpue.index = pd.to_datetime(df_cpue.index)
df_cpue = df_cpue.resample('As').mean()
# index to year
df_cpue.index = df_cpue.index.year

## Anomalies
df_anom = (df_cpue - df_cpue.mean()) / df_cpue.std()

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

## ---- EOFs / PCs data ---- ##
pc_zos = pd.read_pickle('./EOFs/scores_zos_annual.pkl')
pc_zos.index = ['SSH_01','SSH_02', 'SSH_03', 'SSH_04', 'SSH_05']

pc_so = pd.read_pickle('./EOFs/scores_so_annual.pkl')
pc_so.index = ['SAL_01','SAL_02', 'SAL_03', 'SAL_04', 'SAL_05']

pc_so200 = pd.read_pickle('./EOFs/scores_so_200_annual.pkl')
pc_so200.index = ['SAL2_01','SAL2_02', 'SAL2_03', 'SAL2_04', 'SAL2_05']

pc_so400 = pd.read_pickle('./EOFs/scores_so_400_annual.pkl')
pc_so400.index = ['SAL4_01','SAL4_02', 'SAL4_03', 'SAL4_04', 'SAL4_05']

pc_botT = pd.read_pickle('./EOFs/scores_bottomT_annual.pkl')
pc_botT.index = ['BOT_01','BOT_02', 'BOT_03', 'BOT_04', 'BOT_05']

pc_tem = pd.read_pickle('./EOFs/scores_thetao_annual.pkl')
pc_tem.index = ['TEM_01','TEM_02', 'TEM_03', 'TEM_04', 'TEM_05']

pc_tem200 = pd.read_pickle('./EOFs/scores_thetao_200_annual.pkl')
pc_tem200.index = ['TEM2_01','TEM2_02', 'TEM2_03', 'TEM2_04', 'TEM2_05']

pc_tem400 = pd.read_pickle('./EOFs/scores_thetao_400_annual.pkl')
pc_tem400.index = ['TEM4_01','TEM4_02', 'TEM4_03', 'TEM4_04', 'TEM4_05']

## ---- Concat data ---- ##
#PC_CPUE = pd.concat([pc_zos.T,pc_so.T,pc_so200.T, pc_so400.T,pc_botT.T,pc_tem.T,pc_tem200.T,pc_tem400.T,df_anom], axis=1)
# No SSH
PC_CPUE = pd.concat([pc_so.T,pc_so200.T, pc_so400.T,pc_botT.T,pc_tem.T,pc_tem200.T,pc_tem400.T,df_class_anom], axis=1)
PC_CPUE = PC_CPUE[PC_CPUE.index>=2000]


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

# Build the colormap
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

## For Values
# correlation matric and pvalues
PC_CPUE = PC_CPUE.dropna(thresh=20, axis=1)
corrMatrix = PC_CPUE.corr().round(2)
pvalues = calculate_pvalues(PC_CPUE)
## Restrict correlation matrix
corrMatrix = corrMatrix.iloc[0:35,35:,]
pvalues = pvalues.iloc[0:35,35:,]
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
#mask = np.zeros_like(corrMatrix)
#mask[np.triu_indices_from(mask)] = False
#np.fill_diagonal(mask, 0)
sn.heatmap(corrMatrix, annot=corrMatrix_text.astype('str'), fmt='s', linewidths=.2, cmap=cmap, cbar=None, vmin=-1.05, vmax=1.05)
plt.title('Pearson correlation coefficients')
# tweak yticklabels
LABELS = corrMatrix.keys().to_list()
ax = plt.gca()
XTICKS = np.arange(0.5, 9.5, 1)
plt.xticks(XTICKS)
ax.set_xticklabels(LABELS)
fig.set_size_inches(w=20,h=14)
fig_name = 'Correlation_PC-CPUE_classes.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

