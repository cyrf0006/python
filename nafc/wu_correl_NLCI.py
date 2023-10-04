'''
In /home/cyrf0006/research/PeopleStuff/BelangerStuff
'''

import pandas as pd
import matplotlib.pyplot as plt
import os

# Time period
YEAR_MIN = 1998
YEAR_MAX = 2020

## Load bloom param and compute metrics (legacy, initial submission)
df_bloom = pd.read_csv('BloomFitParams_NAFO_15Jun2021.csv')
df_bloom = df_bloom[['Region', 'Sensor', 'Year', 'Mean', 'Median', 't[start]', 't[max]', 't[end]', 't[duration]', 'Magnitude[real]', 'Magnitude[fit]', 'Amplitude[real]', 'Amplitude[fit]']]
# sensor averaging
df_2J = df_bloom[df_bloom.Region=='2J'].groupby('Year').mean()
df_3K = df_bloom[df_bloom.Region=='3K'].groupby('Year').mean()
df_3L = df_bloom[df_bloom.Region=='3L'].groupby('Year').mean()
df_3NO = df_bloom[df_bloom.Region=='3NO'].groupby('Year').mean()
df_3LNO = df_bloom[df_bloom.Region=='3LNO'].groupby('Year').mean()
# Initiation
df_init = pd.concat([df_2J['t[start]'], df_3K['t[start]'], df_3LNO['t[start]']], keys=['2J', '3K','3LNO'], axis=1)
anom_init = (df_init-df_init.mean()) / df_init.std()
anom_init.to_csv('bloom_init_anomaly_for_adamack.csv')
anom_init = anom_init.mean(axis=1)
# Peak
df_max = pd.concat([df_2J['t[max]'], df_3K['t[max]'], df_3LNO['t[max]']], keys=['2J', '3K','3LNO'], axis=1)
anom_max = (df_max-df_max.mean()) / df_max.std()
anom_max = anom_max.mean(axis=1)
anom_max.to_csv('bloom_max_timing_anom.csv')
# End bloom
df_end = pd.concat([df_2J['t[end]'], df_3K['t[end]'], df_3LNO['t[end]']], keys=['2J', '3K','3LNO'], axis=1)
anom_end = (df_end-df_end.mean()) / df_end.std()
anom_end = anom_end.mean(axis=1)
# bloom duration
df_duration = pd.concat([df_2J['t[duration]'], df_3K['t[duration]'], df_3LNO['t[duration]']], keys=['2J', '3K','3LNO'], axis=1)
anom_duration = (df_duration-df_duration.mean()) / df_duration.std()
anom_duration = anom_duration.mean(axis=1)

## Load bloom param and compute metrics (May2023 version)
#df_bloom2 = pd.read_csv('tempIceBloom_clean_avg.csv') # already sensor averaged
#df_2J = df_bloom2[df_bloom2.region=='2J'].groupby('year').mean()
#df_3K = df_bloom2[df_bloom2.region=='3K'].groupby('year').mean()
#df_3L = df_bloom2[df_bloom2.region=='3L'].groupby('year').mean()
#df_3NO = df_bloom2[df_bloom2.region=='3NO'].groupby('year').mean()
#df_3LNO = df_bloom2[df_bloom2.region=='3LNO'].groupby('year').mean()
# Initiation
#df_init = pd.concat([df_2J['t_start'], df_3K['t_start'], df_3LNO['t_start']], keys=['2J', '3K','3LNO'], axis=1)
#anom_init = (df_init-df_init.mean()) / df_init.std()
#anom_init.to_csv('bloom_init_anomaly_for_adamack.csv')
#anom_init = anom_init.mean(axis=1)
# Peak
#df_max = pd.concat([df_2J['t_max'], df_3K['t_max'], df_3LNO['t_max']], keys=['2J', '3K','3LNO'], axis=1)
#anom_max = (df_max-df_max.mean()) / df_max.std()
#anom_max = anom_max.mean(axis=1)
#anom_max.to_csv('bloom_max_timing_anom.csv')
# End bloom
#df_end = pd.concat([df_2J['t_end'], df_3K['t_end'], df_3LNO['t_end']], keys=['2J', '3K','3LNO'], axis=1)
#anom_end = (df_end-df_end.mean()) / df_end.std()
#anom_end = anom_end.mean(axis=1)
# bloom duration
#df_duration = pd.concat([df_2J['t_duration'], df_3K['t_duration'], df_3LNO['t_duration']], keys=['2J', '3K','3LNO'], axis=1)
#anom_duration = (df_duration-df_duration.mean()) / df_duration.std()
#anom_duration = anom_duration.mean(axis=1)


# All together
df_anom = pd.concat([anom_init, anom_max, anom_end, anom_duration], axis=1).mean(axis=1)
# restrict period
df_init = df_init[(df_init.index>=YEAR_MIN) & (df_init.index<=YEAR_MAX)]
df_max = df_max[(df_max.index>=YEAR_MIN) & (df_max.index<=YEAR_MAX)]
df_end = df_end[(df_end.index>=YEAR_MIN) & (df_end.index<=YEAR_MAX)]
df_duration = df_duration[(df_duration.index>=YEAR_MIN) & (df_duration.index<=YEAR_MAX)]
df_anom = df_anom[(df_anom.index>=YEAR_MIN) & (df_anom.index<=YEAR_MAX)]

## Load climate index
nlci = pd.read_csv('/home/cyrf0006/AZMP/state_reports/climate_index/NL_climate_index.csv')
nlci.set_index('Year', inplace=True)
nlci = nlci[(nlci.index>=YEAR_MIN) & (nlci.index<=YEAR_MAX)]


## Load calfin stuff (updated Oct. 2022)
df0 = pd.read_csv('/home/cyrf0006/research/PeopleStuff/BelangerStuff/CorrelationData.csv', index_col='Year')
df = pd.read_csv('Zooplankton_zonal_mean_anomalies.csv', index_col='year')
df = df[(df.index>=YEAR_MIN) & (df.index<=YEAR_MAX)]

df['meanAnomaly_calfin'].to_csv('bloomTiming_calfinAbundance.csv')


## Load stratification
DEPTH_RANGE = '5-150m'
PREDICTION_FILE = 'bloom_predict_gaussian_3LNO_' + DEPTH_RANGE + '.csv'
df_predict = pd.read_csv(PREDICTION_FILE)    
df_predict.set_index('Year', inplace=True)
# keep only after 1998
#df_predict = df_predict[df_predict.index>=1998]
df_predict_clim = df_predict[(df_predict.index>=1991) & (df_predict.index<=2020)]
df_predict = (df_predict - df_predict_clim.mean())/df_predict_clim.std()


## Correlation
anom_init.name='init' 
anom_max.name='max' 
anom_end.name='end' 
anom_duration.name='duration' 
matrix = pd.concat([nlci, anom_init, anom_max, anom_end, anom_duration, df0['calfin'], df['meanAnomaly_Biomass'], df['meanAnomaly_calfin'], df_predict], axis=1) 
from scipy.stats import pearsonr
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

corrMatrix = matrix.corr().round(2)
pvalues = calculate_pvalues(matrix)

anom_init.to_csv('bloomTiming_initiationTiming.csv')
anom_max.to_csv('bloomTiming_peakTiming.csv')

## plot
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2, color='tab:blue')
plt.plot(anom_init, linewidth=2, color='tab:green')
plt.grid()
plt.ylabel('Standardized anomaly')
plt.legend(['NLCI', 'Bloom initiation'])
plt.text(1998, -1.12, ' r = -0.59 (p=0.007)', fontsize=14, fontweight='bold')
fig_name = 'correlation_NLCI_bloom-init.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_bloom-init.png correlation_NLCI_bloom-init.png')

### ------ For manuscript ----- ###

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2, color='tab:blue')
plt.plot(anom_max, linewidth=2, color='tab:green')
plt.grid()
plt.ylabel('Standardized anomaly')
plt.legend(['NLCI', 'Bloom peak timing'], loc='upper right')
#plt.text(1997, -1.4, ' r = -0.73 (p=0.0005)', fontsize=14, fontweight='bold')
plt.text(2021, -1.4, 'r = -0.73 (p=0.001)', fontsize=14, fontweight='bold', horizontalalignment='right')
plt.text(1994, 1.5, 'a)', fontsize=14, fontweight='bold')
plt.ylim([-1.75, 1.75])
# No tick label
plt.gca().axes.get_xaxis().set_ticklabels([])
fig.set_size_inches(w=7, h=4)
fig_name = 'correlation_NLCI_bloom-max.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_bloom-max.png correlation_NLCI_bloom-max.png')

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2, color='tab:blue')
plt.plot(df['meanAnomaly_calfin'], linewidth=2, color='tab:orange')
plt.grid()
plt.ylabel('Standardized anomaly')
plt.legend(['NLCI', r'$C. finmarchicus$'], loc='upper right')
#plt.text(1997, -1.4, ' r = 0.53 (p=0.02)', fontsize=14, fontweight='bold')
plt.text(2021, -1.4, 'r = 0.52 (p=0.02)', fontsize=14, fontweight='bold', horizontalalignment='right')
plt.text(1994, 1.5, 'b)', fontsize=14, fontweight='bold')
plt.ylim([-1.75, 1.75])
fig_name = 'correlation_NLCI_calfin.png'
fig.set_size_inches(w=7, h=4)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_calfin.png correlation_NLCI_calfin.png')

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(anom_max, linewidth=2, color='tab:green')
plt.plot(df['meanAnomaly_calfin'], linewidth=2, color='tab:orange')
plt.grid()
plt.ylabel('Standardized anomaly')
plt.legend(['Bloom peak timing', r'$C. finmarchicus$'], loc='upper right')
#plt.text(1997, -1.4, ' r = -0.42 (p=0.08)', fontsize=14, fontweight='bold')
plt.text(2021, -1.4, 'r = -0.42 (p=0.08)', fontsize=14, fontweight='bold', horizontalalignment='right')
plt.text(1994, 1.5, 'c)', fontsize=14, fontweight='bold')
plt.ylim([-1.75, 1.75])
fig_name = 'correlation_bloom-peak_calfin.png'
fig.set_size_inches(w=7, h=4)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_bloom-peak_calfin.png correlation_bloom-peak_calfin.png')

## ------------------------------------------ ##


fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2, color='tab:blue')
plt.plot(anom_end, linewidth=2, color='tab:orange')
plt.grid()
plt.ylabel('Standardized anomaly')
plt.legend(['NLCI', 'Bloom end timing'])
plt.text(2000, 1.25, ' r = -0.63', fontsize=14, fontweight='bold')
fig.set_size_inches(w=7, h=4)
fig_name = 'correlation_NLCI_bloom-end.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_bloom-end.png correlation_NLCI_bloom-end.png')

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2)
plt.plot(df['meanAnomaly_Biomass'], linewidth=2)
plt.grid()
plt.ylabel('Standardized anomaly')
plt.legend(['NLCI', 'Zooplankton'], loc='upper right')
plt.text(2015, 0.85, ' r = 0.59', fontsize=14, fontweight='bold')
plt.text(1997, 1.25, 'b)', fontsize=14, fontweight='bold')
fig_name = 'correlation_NLCI_zooplankton.png'
fig.set_size_inches(w=7, h=4)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_zooplankton.png correlation_NLCI_zooplankton.png')

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2)
plt.plot(anom_max, linewidth=2)
plt.plot(df['meanAnomaly_calfin'], linewidth=2)
plt.grid()
plt.ylabel('Standardized anomaly', fontsize=12)
plt.legend(['NLCI', 'Bloom peak timing', r'$C. finmarchicus$'], loc='upper right')
plt.text(2015, -1, r' $\rm r_{1,2} = -0.73$', fontsize=14, fontweight='bold')
plt.text(2015, -1.25, r' $\rm r_{2,3} = -0.42$', fontsize=14, fontweight='bold')
plt.text(2015, -1.5, r' $\rm r_{1,3} = 0.53$', fontsize=14, fontweight='bold')
#plt.text(1997, 1.25, 'b)', fontsize=14, fontweight='bold')
plt.ylim([-1.75, 1.75])
fig_name = 'correlation_NLCI_peak_calfin.png'
fig.set_size_inches(w=7, h=4)
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_peak_calfin.png correlation_NLCI_peak_calfin.png')

## fig, ax = plt.subplots(nrows=1, ncols=1)
## plt.plot(df_anom, linewidth=2)
## plt.plot(df['calfin'], linewidth=2)
## plt.grid()
## plt.ylabel('Standardized anomaly')
## plt.legend(['bloom metrics', 'Cal. finmarchicus'])
## plt.text(2015, 0.75, ' r = -0.33', fontsize=14, fontweight='bold')
## fig.set_size_inches(w=7, h=4)
## fig_name = 'correlation_bloom_calfin.png'
## fig.savefig(fig_name, dpi=200)
## os.system('convert -trim correlation_bloom_calfin.png correlation_bloom_calfin.png')


os.system('montage correlation_NLCI_bloom-max.png correlation_NLCI_calfin.png -tile 1x2 -geometry +10+10  -background white  montage_nlci_bloom_calfin.png')

os.system('montage correlation_NLCI_bloom-max.png correlation_NLCI_calfin.png correlation_bloom-peak_calfin.png -tile 1x3 -geometry +10+10  -background white  montage_nlci_bloom_calfin_bloom.png')

os.system('montage Bloom_timing_anom3LNO_5-150m.png correlation_NLCI_peak_calfin.png -tile 1x2 -geometry +10+10  -background white  wu_correl.png')



## comparision with Keith's
#df_K = pd.read_csv('maxbloom_calfin.csv')
#df_K.set_index('year', inplace=True, drop=True)
#df_K.drop('Unnamed: 0', inplace=True)

#anom_max.plot()
#df_K['mean_tmax'].plot()
#df_K['meanAnomaly_calfin'].plot()
#plt.legend(['my max', 'your max', 'calfin'])
#df_K['mymax'] = anom_max
