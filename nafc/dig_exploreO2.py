# A first test to read Excel nutrient file and export to Pandas.

# Check in:
#  /home/cyrf0006/research/AZMP_database/biochem

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from seawater import extras as swx

# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)

## ----  Load data from Excel sheet ---- ##
df = pd.read_excel('/home/cyrf0006/github/AZMP-NL/data/AZMP_Nutrients_1999_2016_Good_Flags_Only.xlsx')

## ---- Some cleaning ---- ##
# Set date as index
df = df.set_index('sas_date')

# Drop other time-related columns
df = df.drop(['Day', 'Month', 'Year'], axis=1)

# Compute Saturation O2 and add to dataframe
df['satO2'] = swx.satO2(df['salinity'], df['temp'])
df['satO2%'] = df['oxygen']/df['satO2']*100

# Try to drop Smith Sound data
df = df.drop(df.index[df['section']=='SS'], axis=0)

## ---- Get numpy array ---- ##
PO4 = df['PO4'].values
NO3 = df['NO3'].values
SIO = df['SIO'].values
T = df['temp'].values
S = df['salinity'].values
SIG = df['sigmat'].values
O2 = df['oxygen'].values
Z = df['depth'].values
O2sat = df['satO2'].values


## ---- clean / remove NaNs ---- ##
idx_good = np.argwhere(~np.isnan(O2))
O2sat = np.squeeze(O2sat[idx_good])
O2 = np.squeeze(O2[idx_good])
T = np.squeeze(T[idx_good])
years = np.squeeze(df.index.year[idx_good])
AOU = O2sat - O2
## O2sat = np.squeeze(O2sat[idx_good])
## O2 = np.squeeze(O2[idx_good])
## T = np.squeeze(T[idx_good])


## ---- T-O2 relationship ---- ##
fig = plt.figure(1)
plt.clf()
plt.scatter(T, O2, c=years, alpha=0.5, cmap=plt.cm.RdBu_r)
plt.ylabel(r'$\rm O_2 (mg L^{-1})$', fontsize=20, fontweight='bold')
plt.xlabel(r'$\rm T (^{\circ}C)$', fontsize=20, fontweight='bold')
cb = plt.colorbar()
#cb.ax.set_ylabel('Years', fontsize=20, fontweight='bold')
plt.ylim([2,12])
plt.xlim([-2,19])
plt.text(10,10, 'N = %d\n' % np.size(np.where((~np.isnan(T)) & (~np.isnan(O2)))), fontsize=20, fontweight='bold')
#plt.show()

fig.set_size_inches(w=8, h=6)
fig.set_dpi(300)
fig.savefig('T-O2_scatter.png')

## ---- T-O2 relationship ---- ##
fig = plt.figure(2)
plt.clf()
plt.scatter(O2sat, O2, c=years, alpha=0.5, cmap=plt.cm.RdBu_r)
plt.ylabel(r'$\rm O_2 (mg L^{-1})$', fontsize=20, fontweight='bold')
plt.xlabel(r'$\rm O2_{sat} (mg L^{-1})$', fontsize=20, fontweight='bold')
cb = plt.colorbar()
#cb.ax.set_ylabel('Years', fontsize=20, fontweight='bold')
plt.ylim([2,12])
plt.xlim([2, 12])
plt.text(10,10, 'N = %d\n' % np.size(np.where((~np.isnan(O2sat)) & (~np.isnan(O2)))), fontsize=20, fontweight='bold')
#plt.show()

fig.set_size_inches(w=8, h=6)
fig.set_dpi(300)
fig.savefig('O2sat-O2_scatter.png')

## ---- T-AOU relationship ---- ##
fig = plt.figure(3)
plt.clf()
plt.scatter(T, AOU, c=years, alpha=0.5, cmap=plt.cm.RdBu_r)
plt.ylabel(r'$\rm AOU (mg L^{-1})$', fontsize=20, fontweight='bold')
plt.xlabel(r'$\rm T (^{\circ}C)$', fontsize=20, fontweight='bold')
cb = plt.colorbar()
#cb.ax.set_ylabel('Years', fontsize=20, fontweight='bold')
plt.ylim([-3,4])
plt.xlim([-2,19])
plt.text(12,3, 'N = %d\n' % np.size(np.where((~np.isnan(T)) & (~np.isnan(O2)))), fontsize=20, fontweight='bold')
#plt.show()

fig.set_size_inches(w=8, h=6)
fig.set_dpi(300)
fig.savefig('T-AOU_scatter.png')
