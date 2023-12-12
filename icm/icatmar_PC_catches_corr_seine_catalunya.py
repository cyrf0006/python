'''
Test correlation between CPUEs for the entire Catalunya
(unlike the regional where I separated the different regions)
This is the most recent version develop before leaving

Frederic.Cyr@dfo-mpo.gc.ca
December 2023

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
# Series 1 (zos, temp 0-400, sal 0-400) or Series 2 (temp/sal 75, MLD, EKE, chla) or Series 3 (EKE, KE, and wMLD lags)
series1 = False
series2 = False
series3 = False
series4 = False
series5 = True

# 3 or 5 PCs?
threePCs = True

## ---- Catch data ---- ##
# Load data
df_landings = pd.read_csv('20231017_Landings_PS_00-22.csv')

# CPUEs
VAR = 'Average weight (Kg / day * vessel)' # CPUE
#VAR = 'SumWeight_Kg'

cpue_cat = df_landings.groupby(['Date', 'Subgroup', ]).mean()[VAR]
cpue_cat = cpue_cat.unstack()
cpue_cat.index = pd.to_datetime(cpue_cat.index)
cpue_cat = cpue_cat.resample('As').mean()


# Seclect most important species based on value (all regions)
df_value = df_landings.groupby(['Date', 'Subgroup']).mean()['SumAmount_Euros']
df_value = df_landings.groupby(['Date', 'Subgroup']).mean()[VAR]
df_value = df_value.unstack()
df_value.index = pd.to_datetime(df_value.index)
df_value = df_value.resample('As').mean()
value10 = df_value.sum()[df_value.sum()>=df_value.sum().sort_values().iloc[-4]].index
value8 = df_value.sum()[df_value.sum()>=df_value.sum().sort_values().iloc[-8]].index

cpue_cat = cpue_cat[value10.values]

# Concat regions
df_cpue = cpue_cat

## Anomalies
df_anom = (df_cpue - df_cpue.mean()) / df_cpue.std()
df_anom.index = df_anom.index.year

## Drop some values if a lot of NaNs
# correlation matric and pvalues
#df_anom = df_anom.dropna(thresh=21, axis=1)
# Find shape 
L = df_anom.shape[1]

## ---- EOFs / PCs data ---- ##
## ---- EOFs / PCs data ---- ##
pc_zos = pd.read_pickle('./EOFs/scores_zos_annual_NNW.pkl')
pc_zos.index = ['SSH_1','SSH_2', 'SSH_3', 'SSH_4', 'SSH_5']

pc_so = pd.read_pickle('./EOFs/scores_so_annual_NNW.pkl')
pc_so.index = ['SALSUR_1','SALSUR_2', 'SALSUR_3', 'SALSUR_4', 'SALSUR_5']

pc_so075 = pd.read_pickle('./EOFs/scores_so_075_annual_NNW.pkl')
pc_so075.index = ['SAL075_1','SAL075_2', 'SAL075_3', 'SAL075_4', 'SAL075_5']

pc_so200 = pd.read_pickle('./EOFs/scores_so_200_annual_NNW.pkl')
pc_so200.index = ['SAL200_1','SAL200_2', 'SAL200_3', 'SAL200_4', 'SAL200_5']

pc_so400 = pd.read_pickle('./EOFs/scores_so_400_annual_NNW.pkl')
pc_so400.index = ['SAL400_1','SAL400_2', 'SAL400_3', 'SAL400_4', 'SAL400_5']

pc_botT = pd.read_pickle('./EOFs/scores_bottomT_annual_NNW.pkl')
pc_botT.index = ['TEMBOT_1','TEMBOT_2', 'TEMBOT_3', 'TEMBOT_4', 'TEMBOT_5']

pc_tem = pd.read_pickle('./EOFs/scores_thetao_annual_NNW.pkl')
pc_tem.index = ['TEMSUR_1','TEMSUR_2', 'TEMSUR_3', 'TEMSUR_4', 'TEMSUR_5']

pc_tem075 = pd.read_pickle('./EOFs/scores_thetao_075_annual_NNW.pkl')
pc_tem075.index = ['TEM075_1','TEM075_2', 'TEM075_3', 'TEM075_4', 'TEM075_5']

pc_tem200 = pd.read_pickle('./EOFs/scores_thetao_200_annual_NNW.pkl')
pc_tem200.index = ['TEM200_1','TEM200_2', 'TEM200_3', 'TEM200_4', 'TEM200_5']

pc_tem400 = pd.read_pickle('./EOFs/scores_thetao_400_annual_NNW.pkl')
pc_tem400.index = ['TEM400_1','TEM400_2', 'TEM400_3', 'TEM400_4', 'TEM400_5']

pc_eke = pd.read_pickle('./EOFs/scores_EKE_annual_NNW.pkl')
pc_eke.index = ['EKE_1','EKE_2', 'EKE_3', 'EKE_4', 'EKE_5']

# Use monthyl EKE and everage yearly [Not convincing, maybe to further explore...]
pc_ekem = pd.read_pickle('./EOFs/scores_EKE_deseasonNNW.pkl')
pc_ekem = pc_ekem.T
#pc_ekem = pc_ekem[(pc_ekem.index.month>=5) & (pc_ekem.index.month<=9)]
pc_ekem = pc_ekem.resample('As').mean()
pc_ekem.index = pc_ekem.index.year
pc_ekem = pc_ekem.T
pc_ekem.index = ['EKE_1','EKE_2', 'EKE_3', 'EKE_4', 'EKE_5']

pc_ke = pd.read_pickle('./EOFs/scores_KE_annual_NNW.pkl')
pc_ke.index = ['KE_1','KE_2', 'KE_3', 'KE_4', 'KE_5']

pc_mld = pd.read_pickle('./EOFs/scores_MLD_annual_NNW.pkl')
pc_mld.index = ['MLD_1','MLD_2', 'MLD_3', 'MLD_4', 'MLD_5']

pc_wmld = pd.read_pickle('./EOFs/scores_wMLD_annual_NNW.pkl')
pc_wmld.index = ['wMLD_1','wMLD_2', 'wMLD_3', 'wMLD_4', 'wMLD_5']

pc_chl = pd.read_pickle('./EOFs/scores_chla_annual_NNW.pkl')
pc_chl.index = ['CHLSUR_1','CHLSUR_2', 'CHLSUR_3', 'CHLSUR_4', 'CHLSUR_5']

pc_chl075 = pd.read_pickle('./EOFs/scores_chla75_annual_NNW.pkl')
pc_chl075.index = ['CHL075_1','CHL075_2', 'CHL075_3', 'CHL075_4', 'CHL075_5']

pc_phyc = pd.read_pickle('./EOFs/scores_phyc_annual_NNW.pkl')
pc_phyc.index = ['PHYC_1','PHYC_2', 'PHYC_3', 'PHYC_4', 'PHYC_5']

pc_chlt = pd.read_pickle('./EOFs/scores_chlat_annual_NNW.pkl')
pc_chlt.index = ['CHLTOT_1','CHLTOT_2', 'CHLTOT_3', 'CHLTOT_4', 'CHLTOT_5']

pc_phyc = pd.read_pickle('./EOFs/scores_phyc_annual_NNW.pkl')
pc_phyc.index = ['PHYC_1','PHYC_2', 'PHYC_3', 'PHYC_4', 'PHYC_5']

# Calculate wMLD lags
pc_wmld1 = pc_wmld.copy()
pc_wmld2 = pc_wmld.copy()
pc_wmld1.index = ['wMLD1_1', 'wMLD1_2', 'wMLD1_3', 'wMLD1_4', 'wMLD1_5']
pc_wmld2.index = ['wMLD2_1', 'wMLD2_2', 'wMLD2_3', 'wMLD2_4', 'wMLD2_5']
pc_wmld1.columns = pc_wmld.columns+1
pc_wmld2.columns = pc_wmld.columns+2
# Calculate CHLt lags
pc_chlt1 = pc_chlt.copy()
pc_chlt2 = pc_chlt.copy()
pc_chlt1.index = ['CHLt1_1', 'CHLt1_2', 'CHLt1_3', 'CHLt1_4', 'CHLt1_5']
pc_chlt2.index = ['CHLt2_1', 'CHLt2_2', 'CHLt2_3', 'CHLt2_4', 'CHLt2_5']
pc_chlt1.columns = pc_chlt.columns+1
pc_chlt2.columns = pc_chlt.columns+2
# Calculate bottomT lags
pc_botT1 = pc_botT.copy()
pc_botT2 = pc_botT.copy()
pc_botT1.index = ['BOT1_1', 'BOT1_2', 'BOT1_3', 'BOT1_4', 'BOT1_5']
pc_botT2.index = ['BOT2_1', 'BOT2_2', 'BOT2_3', 'BOT2_4', 'BOT2_5']
pc_botT1.columns = pc_botT.columns+1
pc_botT2.columns = pc_botT.columns+2

## ---- Concat data ---- ##
#PC_CPUE = pd.concat([pc_zos.T,pc_so.T,pc_so200.T, pc_so400.T,pc_botT.T,pc_tem.T,pc_tem200.T,pc_tem400.T,df_anom], axis=1)
# No SSH
if series1:
    PC_CPUE = pd.concat([df_anom, pc_zos.T, pc_so.T, pc_so200.T, pc_so400.T,pc_botT.T,pc_tem.T,pc_tem200.T,pc_tem400.T], axis=1)
elif series2:
    PC_CPUE = pd.concat([df_anom, pc_so.T, pc_so075.T, pc_botT.T, pc_tem.T, pc_tem075.T, pc_eke.T, pc_wmld.T, pc_phyc.T, pc_chl075.T], axis=1)
elif series3:
    PC_CPUE = pd.concat([df_anom, pc_so075.T, pc_botT.T, pc_tem075.T, pc_ke.T, pc_eke.T, pc_wmld.T, pc_wmld1.T, pc_wmld2.T, pc_chl075.T], axis=1)
elif series4: 
    PC_CPUE = pd.concat([df_anom, pc_so075.T, pc_botT.T, pc_tem075.T, pc_ke.T, pc_eke.T, pc_wmld.T, pc_chlt.T, pc_chlt1.T, pc_chlt2.T], axis=1)
else: 
    #PC_CPUE = pd.concat([df_anom, pc_so.T, pc_so075.T, pc_so200.T, pc_tem.T, pc_tem400.T, pc_botT.T, pc_eke.T], axis=1)
    PC_CPUE = pd.concat([df_anom, pc_so.T, pc_so075.T, pc_so200.T, pc_tem.T, pc_tem400.T, pc_eke.T], axis=1)
    
PC_CPUE = PC_CPUE[PC_CPUE.index>=2000]

# 3 or 5 PCs?
if threePCs:
    PC_CPUE = PC_CPUE.drop(PC_CPUE.columns[PC_CPUE.columns.str.contains('_4')], axis=1)
    PC_CPUE = PC_CPUE.drop(PC_CPUE.columns[PC_CPUE.columns.str.contains('_5')], axis=1)

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
#PC_CPUE = PC_CPUE.dropna(thresh=22, axis=1)   
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
# add delimitation between PCs
plt.plot([3,3],[0,4], color='gray', linestyle='--')
plt.plot([6,6],[0,4], color='gray', linestyle='--')
plt.plot([9,9],[0,4], color='gray', linestyle='--')
plt.plot([12,12],[0,4], color='gray', linestyle='--')
plt.plot([15,15],[0,4], color='gray', linestyle='--')
plt.plot([18,18],[0,4], color='gray', linestyle='--')
plt.plot([21,21],[0,4], color='gray', linestyle='--')

if threePCs:
    fig.set_size_inches(w=15,h=8)
    if series1:
        fig_name = 'Correlation_3PCs-CPUE_PS_catalunya.png'
    elif series2:
        fig_name = 'Correlation_3PCs-CPUE_PS_catalunya_series2.png'
    elif series3:
        fig_name = 'Correlation_3PCs-CPUE_PS_catalunya_series3.png'
    elif series4:
        fig_name = 'Correlation_3PCs-CPUE_PS_catalunya_series4.png'
    else:
        fig_name = 'Correlation_3PCs-CPUE_PS_catalunya_series5.png'
else:
    fig.set_size_inches(w=20,h=14)
    if series1:
        fig_name = 'Correlation_PC-CPUE_PS_catalunya.png'
    elif series2:
        fig_name = 'Correlation_PC-CPUE_PS_catalunya_series2.png'
    elif series3:
        fig_name = 'Correlation_PC-CPUE_PS_catalunya_series3.png'
    elif series4:
        fig_name = 'Correlation_PC-CPUE_PS_catalunya_series4.png'
    else:
        fig_name = 'Correlation_PC-CPUE_PS_catalunya_series5.png'

fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

