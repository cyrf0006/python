'''
In /home/cyrf0006/research/PeopleStuff/BelangerStuff
'''

import pandas as pd
import matplotlib.pyplot as plt
import os


## Load bloom param and compute metrics
df_bloom = pd.read_csv('BloomFitParams_NAFO_15Jun2021.csv')
df_bloom = df_bloom[['Region', 'Sensor', 'Year', 'Mean', 'Median', 't[start]', 't[max]', 't[end]', 't[duration]', 'Magnitude[real]', 'Magnitude[fit]', 'Amplitude[real]', 'Amplitude[fit]']]
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

df_anom = pd.concat([anom_init, anom_max, anom_end], axis=1).mean(axis=1)

## Load climate index
nlci = pd.read_csv('/home/cyrf0006/AZMP/state_reports/climate_index/NL_climate_index.csv')
nlci.set_index('Year', inplace=True)
nlci = nlci[nlci.index>=1998]

## Load calfin stuff (need to be updated)
df = pd.read_csv('/home/cyrf0006/research/PeopleStuff/BelangerStuff/CorrelationData.csv', index_col='Year')

## plot

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2)
plt.plot(anom_init, linewidth=2)
plt.grid()
plt.ylabel('Normalized anomaly')
plt.legend(['NLCI', 'Bloom initiation'])
plt.text(2000, 1.25, ' r = -0.59', fontsize=15, fontweight='bold')
fig_name = 'correlation_NLCI_bloom-init.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_bloom-init.png correlation_NLCI_bloom-init.png')


fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2)
plt.plot(anom_max, linewidth=2)
plt.grid()
plt.ylabel('Normalized anomaly')
plt.legend(['NLCI', 'Bloom max timing'])
plt.text(2015, -1.15, ' r = -0.72', fontsize=15, fontweight='bold')
plt.text(1997, 1.3, 'a)', fontsize=14, fontweight='bold')
# No tick label
plt.gca().axes.get_xaxis().set_ticklabels([])
fig_name = 'correlation_NLCI_bloom-max.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_bloom-max.png correlation_NLCI_bloom-max.png')


fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2)
plt.plot(anom_end, linewidth=2)
plt.grid()
plt.ylabel('Normalized anomaly')
plt.legend(['NLCI', 'Bloom end timing'])
plt.text(2000, 1.25, ' r = -0.62', fontsize=15, fontweight='bold')
fig_name = 'correlation_NLCI_bloom-end.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_bloom-end.png correlation_NLCI_bloom-end.png')



fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(nlci, linewidth=2)
plt.plot(df['calfin'], linewidth=2)
plt.grid()
plt.ylabel('Normalized anomaly')
plt.legend(['NLCI', 'Cal. finmarchicus'], loc='upper right')
plt.text(2015, 0.85, ' r = 0.58', fontsize=15, fontweight='bold')
plt.text(1997, 1.25, 'b)', fontsize=14, fontweight='bold')
fig_name = 'correlation_NLCI_calfin.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_NLCI_calfin.png correlation_NLCI_calfin.png')


fig, ax = plt.subplots(nrows=1, ncols=1)
plt.plot(df_anom, linewidth=2)
plt.plot(df['calfin'], linewidth=2)
plt.grid()
plt.ylabel('Normalized anomaly')
plt.legend(['bloom metrics', 'Cal. finmarchicus'])
plt.text(2015, 0.75, ' r = -0.33', fontsize=15, fontweight='bold')
fig_name = 'correlation_bloom_calfin.png'
fig.savefig(fig_name, dpi=200)
os.system('convert -trim correlation_bloom_calfin.png correlation_bloom_calfin.png')


pd.concat([nlci, anom_init], axis=1).corr()
pd.concat([nlci, anom_max], axis=1).corr()
pd.concat([nlci, anom_end], axis=1).corr()
pd.concat([nlci, df['calfin']], axis=1).corr()
pd.concat([df_anom, df['calfin']], axis=1).corr()

pd.concat([anom_max, df['calfin']], axis=1).corr()

os.system('montage correlation_NLCI_bloom-max.png correlation_NLCI_calfin.png -tile 1x2 -geometry +10+10  -background white  montage_nlci_bloom_calfin.png')
