
'''
Fit and correlation between the bloom timing and its prediction (from stratification).

Reads prediction saved in wu_N2_min.py

Frederic.Cyr@dfo-mpo.gc.ca

October 2022

'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gsw
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from lmfit.models import LorentzianModel
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import optimize
from datetime import timedelta

## Parameters
REGION = '3LNO'
#REGION = 'AC'
DEPTH_RANGE = '5-150m'
PREDICTION_FILE = 'bloom_predict_gaussian_3LNO_' + DEPTH_RANGE + '.csv'

## Load prediction
df_predict = pd.read_csv(PREDICTION_FILE)    
df_predict.set_index('Year', inplace=True)
# keep only after 1998
df_predict = df_predict[df_predict.index>=1998]

## Load bloom params
if REGION == 'AC': # No SeaWiFS, but VIIRS
    bloom = pd.read_csv('SpringBloom_parameters.csv') 
    bloom.set_index('year', inplace=True)
    #bloom = bloom.groupby('year').mean()  
    # Select sensors
    df_predict['Modis'] = bloom[bloom.sensor=='MODIS']['t[start]']
    df_predict['SeaWiFS'] = bloom[bloom.sensor=='SeaWiFS']['t[start]']
    both = df_predict.loc[: , "Modis":"SeaWiFS"]
    df_predict['Both'] = both.mean(axis=1)
    
else:     
    bloom = pd.read_csv('BloomFitParams_Wu revisited_15Jun2021.csv')
    bloom.set_index('Year', inplace=True)
    bloom = bloom[bloom.Region==REGION]
    # Select sensors
    df_predict['Modis'] = bloom[bloom.Sensor=='MODIS 4km']['t[start]']
    df_predict['SeaWiFS'] = bloom[bloom.Sensor=='SeaWiFS 4km']['t[start]']
    both = df_predict.loc[: , "Modis":"SeaWiFS"]
    df_predict['Both'] = both.mean(axis=1)
    
## Compute anomaly
anom = (df_predict - df_predict.mean(axis=0)) / df_predict.std(axis=0)
anom = anom[['predict', 'Both']]                                                  
anom2 = anom.copy()


# Drop some years (late occupation)
df_predict.drop(1998, inplace=True)

## Correlation
corr = df_predict.corr()
corr_anom = anom.corr()

from scipy.stats import pearsonr
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

corrMatrix = df_predict.corr().round(2)
pvalues = calculate_pvalues(df_predict)


# plot
plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1)
df_predict.plot(ax=ax)
ax.grid()
ax.text(2004, 105, 'corr with Modis = ' + str(np.round(corr.iloc[0,1],3)))
ax.text(2004, 100, 'corr with SeaWiFS = ' + str(np.round(corr.iloc[0,2],3)))
ax.text(2004, 95, 'corr with both = ' + str(np.round(corr.iloc[0,3],3)))
plt.ylabel('bloom init (DOY)')
# Save Figure
fig.set_size_inches(w=12, h=6)
outfile_year = 'Predict_bloom_gauss' + REGION + '_' + DEPTH_RANGE + '.png'
fig.savefig(outfile_year, dpi=200)
os.system('convert -trim ' + outfile_year + ' ' + outfile_year)

# plot
plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)
anom.plot(ax=ax, linewidth=2)
#plt.legend([r'DOY of $\rm \tilde{N}^2_{min}$', 'DOY of bloom initiation'])
plt.legend([r'$\rm \widetilde{ts}$', r'$\rm \widetilde{tb_i}$'], loc='upper left')
ax.grid()
#ax.text(2012, 2.25, ' r  = ' + str(np.round(corrMatrix.iloc[0,3],3)) + ' [p = ' + str(np.round(pvalues.iloc[0,3],3)) + ']', FontSize=14)
ax.text(2012, 2.25, ' r  = ' + str(np.round(corrMatrix.iloc[0,3],3)), FontSize=14)
plt.ylabel('standardized anomaly', FontSize=14)
plt.xlabel(' ')
# Save Figure
fig.set_size_inches(w=7, h=4)
outfile_year = 'Bloom_timing_anom' + REGION + '_' + DEPTH_RANGE + '.png'
fig.savefig(outfile_year, dpi=200)
os.system('convert -trim ' + outfile_year + ' ' + outfile_year)

# plot add tice
plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)
anom2.plot(ax=ax, linewidth=2)
#plt.legend(['predict. from stratification', 'Modis / SeaWiFS'])
ax.grid()
ax.text(1998, 2.1, ' r  = ' + str(np.round(corrMatrix.iloc[0,3],3)) + ' [p = ' + str(np.round(pvalues.iloc[0,3],3)) + ']', FontSize=14)
plt.ylabel('bloom initiation std anomaly', FontSize=14)
# Save Figure
fig.set_size_inches(w=12, h=6)
outfile_year = 'Bloom_timing_anom_tice' + REGION + '_' + DEPTH_RANGE + '.png'
fig.savefig(outfile_year, dpi=200)
os.system('convert -trim ' + outfile_year + ' ' + outfile_year)

# montage N2_bloom_1998_3LNO_5-150m.png  N2_bloom_2002_3LNO_5-150m.png N2_bloom_2006_3LNO_5-150m.png N2_bloom_2010_3LNO_5-150m.png -tile 2x2 -geometry +10+10  -background white  montage_strat_fits.png


# montage N2_bloom_1998_3LNO_5-150m.png N2_bloom_1999_3LNO_5-150m.png N2_bloom_2000_3LNO_5-150m.png N2_bloom_2001_3LNO_5-150m.png N2_bloom_2002_3LNO_5-150m.png N2_bloom_2003_3LNO_5-150m.png N2_bloom_2004_3LNO_5-150m.png N2_bloom_2005_3LNO_5-150m.png -tile 2x4 -geometry +10+10  -background white  montage_strat_fits1.png

#montage N2_bloom_2006_3LNO_5-150m.png N2_bloom_2007_3LNO_5-150m.png N2_bloom_2008_3LNO_5-150m.png N2_bloom_2009_3LNO_5-150m.png N2_bloom_2010_3LNO_5-150m.png N2_bloom_2011_3LNO_5-150m.png N2_bloom_2012_3LNO_5-150m.png N2_bloom_2013_3LNO_5-150m.png -tile 2x4 -geometry +10+10  -background white  montage_strat_fits2.png

#montage  N2_bloom_2014_3LNO_5-150m.png N2_bloom_2015_3LNO_5-150m.png N2_bloom_2016_3LNO_5-150m.png -tile 1x3 -geometry +10+10  -background white  montage_strat_fits3.png
