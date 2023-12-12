'''
Cross-correlation between different EOFs

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

## ---- EOFs / PCs data ---- ##
pc_zos = pd.read_pickle('./EOFs/scores_zos_annual_NNW.pkl')
pc_zos.index = ['SSH_01','SSH_02', 'SSH_03', 'SSH_04', 'SSH_05']

pc_so = pd.read_pickle('./EOFs/scores_so_annual_NNW.pkl')
pc_so.index = ['SAL000_01','SAL000_02', 'SAL000_03', 'SAL000_04', 'SAL000_05']

pc_so075 = pd.read_pickle('./EOFs/scores_so_075_annual_NNW.pkl')
pc_so075.index = ['SAL075_01','SAL075_02', 'SAL075_03', 'SAL075_04', 'SAL075_05']

pc_so200 = pd.read_pickle('./EOFs/scores_so_200_annual_NNW.pkl')
pc_so200.index = ['SAL200_01','SAL200_02', 'SAL200_03', 'SAL200_04', 'SAL200_05']

pc_so400 = pd.read_pickle('./EOFs/scores_so_400_annual_NNW.pkl')
pc_so400.index = ['SAL400_01','SAL400_02', 'SAL400_03', 'SAL400_04', 'SAL400_05']

pc_botT = pd.read_pickle('./EOFs/scores_bottomT_annual_NNW.pkl')
pc_botT.index = ['TEMBOT_01','TEMBOT_02', 'TEMBOT_03', 'TEMBOT_04', 'TEMBOT_05']

pc_tem = pd.read_pickle('./EOFs/scores_thetao_annual_NNW.pkl')
pc_tem.index = ['TEM000_01','TEM000_02', 'TEM000_03', 'TEM000_04', 'TEM000_05']

pc_tem075 = pd.read_pickle('./EOFs/scores_thetao_075_annual_NNW.pkl')
pc_tem075.index = ['TEM075_01','TEM075_02', 'TEM075_03', 'TEM075_04', 'TEM075_05']

pc_tem200 = pd.read_pickle('./EOFs/scores_thetao_200_annual_NNW.pkl')
pc_tem200.index = ['TEM200_01','TEM200_02', 'TEM200_03', 'TEM200_04', 'TEM200_05']

pc_tem400 = pd.read_pickle('./EOFs/scores_thetao_400_annual_NNW.pkl')
pc_tem400.index = ['TEM400_01','TEM400_02', 'TEM400_03', 'TEM400_04', 'TEM400_05']

pc_eke = pd.read_pickle('./EOFs/scores_EKE_annual_NNW.pkl')
pc_eke.index = ['EKE_01','EKE_02', 'EKE_03', 'EKE_04', 'EKE_05']

pc_ke = pd.read_pickle('./EOFs/scores_KE_annual_NNW.pkl')
pc_ke.index = ['KE_01','KE_02', 'KE_03', 'KE_04', 'KE_05']

pc_mld = pd.read_pickle('./EOFs/scores_MLD_annual_NNW.pkl')
pc_mld.index = ['MLD_01','MLD_02', 'MLD_03', 'MLD_04', 'MLD_05']

pc_wmld = pd.read_pickle('./EOFs/scores_wMLD_annual_NNW.pkl')
pc_wmld.index = ['wMLD_01','wMLD_02', 'wMLD_03', 'wMLD_04', 'wMLD_05']

pc_chl = pd.read_pickle('./EOFs/scores_chla_annual_NNW.pkl')
pc_chl.index = ['CHL000_01','CHL000_02', 'CHL000_03', 'CHL000_04', 'CHL000_05']

pc_chl075 = pd.read_pickle('./EOFs/scores_chla75_annual_NNW.pkl')
pc_chl075.index = ['CHL075_01','CHL075_02', 'CHL075_03', 'CHL075_04', 'CHL075_05']

pc_phyc = pd.read_pickle('./EOFs/scores_phyc_annual_NNW.pkl')
pc_phyc.index = ['PHYC_01','PHYC_02', 'PHYC_03', 'PHYC_04', 'PHYC_05']

## ---- Concat data ---- ##
#PC = pd.concat([pc_zos.T, pc_so.T, pc_so075.T, pc_so200.T, pc_so400.T, pc_tem.T, pc_tem075.T, pc_tem200.T,pc_tem400.T, pc_botT.T, pc_eke.T,pc_mld.T,pc_chl.T, pc_chl075.T], axis=1)
PC = pd.concat([pc_zos.T, pc_so.T, pc_so075.T, pc_so200.T, pc_so400.T, pc_tem.T, pc_tem075.T, pc_tem200.T,pc_tem400.T, pc_botT.T, pc_ke.T, pc_eke.T,pc_wmld.T,pc_chl.T, pc_chl075.T, pc_phyc.T], axis=1)


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
#L = df_anom.shape[1]
# correlation matric and pvalues
corrMatrix = PC.corr().round(2)
pvalues = calculate_pvalues(PC)
## Restrict correlation matrix
#corrMatrix = corrMatrix.iloc[0:L,L:,]
#pvalues = pvalues.iloc[0:L,L:,]
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
mask = np.zeros_like(corrMatrix)
mask[np.triu_indices_from(mask)] = True
np.fill_diagonal(mask, 0)
sn.heatmap(corrMatrix, annot=corrMatrix_text.astype('str'), fmt='s', mask=mask, linewidths=.2, cmap=cmap, cbar=None, vmin=-1.05, vmax=1.05)
plt.title('Pearson correlation coefficients')
# tweak yticklabels
LABELS = corrMatrix.keys().to_list()
ax = plt.gca()
LL = corrMatrix.shape[1]
XTICKS = np.arange(0.5, LL+.5, 1)
plt.xticks(XTICKS)
plt.yticks(XTICKS)
ax.set_xticklabels(LABELS)
ax.set_yticklabels(LABELS)
fig.set_size_inches(w=30,h=14)
fig_name = 'Cross_Correlation_PC.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

