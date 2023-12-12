
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import seaborn as sn
import cmocean as cmo
from matplotlib.colors import from_levels_and_colors


## --- Catch data --- ##
# Load data
df_landings = pd.read_csv('20230921_landingsCatalunya_00-22.csv')

# Subgroups (N=174)
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

# Plot distribution of all sub-species
plt.close('all')
fig, ax = plt.subplots()
df_weight.sum().sort_values().plot.bar(rot=90)
plt.ylabel('Total Weight (kg)')
plt.xlabel('species')
plt.grid()
fig.tight_layout()
#plt.show()

plt.close('all')
fig, ax = plt.subplots()
df_value.sum().sort_values().plot.bar(rot=90)
plt.ylabel('Value (euros)')
plt.xlabel('species')
plt.grid()
#fig.tight_layout(rect=[0,0,.8,1])
fig.tight_layout()
#plt.show()


# Seclect most important species
value30 = df_value.sum()[df_value.sum()>=df_value.sum().sort_values().iloc[-30]].index
weight30 = df_weight.sum()[df_weight.sum()>=df_weight.sum().sort_values().iloc[-30]].index
cpue30 = df_cpue.sum()[df_cpue.sum()>=df_cpue.sum().sort_values().iloc[-30]].index

df_value = df_value[value30.values]
df_weight = df_weight[weight30.values]
df_cpue = df_cpue[cpue30.values]
# index to year
df_value.index = df_value.index.year
df_weight.index = df_weight.index.year
df_cpue.index = df_cpue.index.year

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
corrMatrix = df_value.corr().round(2)
pvalues = calculate_pvalues(df_value)
annot_text  = corrMatrix.astype('str')
corrMatrix_text = corrMatrix.copy()

for i in np.arange(11):
    for j in np.arange(11):
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
LABELS = corrMatrix.index.to_list()
ax = plt.gca()
YTICKS = np.arange(0.5, 30.5, 1)
plt.yticks(YTICKS)
ax.set_yticklabels(LABELS)
fig.set_size_inches(w=13,h=14)
fig_name = 'Correlation_landings_value30.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)



## For Landings
# correlation matric and pvalues
corrMatrix = df_weight.corr().round(2)
pvalues = calculate_pvalues(df_weight)
annot_text  = corrMatrix.astype('str')
corrMatrix_text = corrMatrix.copy()

for i in np.arange(11):
    for j in np.arange(11):
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
LABELS = corrMatrix.index.to_list()
ax = plt.gca()
YTICKS = np.arange(0.5, 30.5, 1)
plt.yticks(YTICKS)
ax.set_yticklabels(LABELS)
fig.set_size_inches(w=13,h=14)
fig_name = 'Correlation_landings_weight30.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)

## For CPUE
# correlation matric and pvalues
df_cpue = df_cpue.dropna(thresh=20, axis=1)
corrMatrix = df_cpue.corr().round(2)
pvalues = calculate_pvalues(df_cpue)
annot_text  = corrMatrix.astype('str')
corrMatrix_text = corrMatrix.copy()

for i in np.arange(11):
    for j in np.arange(11):
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
LABELS = corrMatrix.index.to_list()
ax = plt.gca()
YTICKS = np.arange(0.5, 24.5, 1)
plt.yticks(YTICKS)
ax.set_yticklabels(LABELS)
fig.set_size_inches(w=13,h=14)
fig_name = 'Correlation_landings_cpue30.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)
