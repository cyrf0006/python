'''
Script to develop a stratification index at Station 27.
Would be useful for capelin forecast model and to revisit Wu paper.

Frederic.Cyr@dfo-mpo.gc.ca

'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
## import xarray as xr
## import datetime
## import os
## import matplotlib.dates as mdates
## from matplotlib.ticker import NullFormatter
## from matplotlib.dates import MonthLocator, DateFormatter
## import cmocean
## import gsw
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from lmfit.models import LorentzianModel
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Gaussian function
def gaus(x,a,x0,sigma,off):
    return a*exp(-(x-x0)**2/(2*sigma**2))+off

# Function to calculate the exponential with constants a and b
def exponential(x, a, b):
    return a*np.exp(b*x)

# Logistic curve fitting
def logifunc(x,A,x0,k,off):
    return A / (1 + np.exp(-k*(x-x0)))+off

## load Stn 27 data
df_N2 = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_N2_raw.pkl')
df_MLD = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_MLD_raw.pkl')
df_strat = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_stratif_raw.pkl')

## Load bloom params
bloom = pd.read_csv('SpringBloom_parameters.csv')
bloom.set_index('year', inplace=True)

## Gaussian Fit.
years = np.arange(2003, 2021)

for year in years:
    print(str(year))
    # Get annual values
    df = df_strat[df_strat.index.year==year]
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    df.index = df.index.dayofyear
    df = df.groupby('time',as_index=True).mean()
    x = df.index.values
    y = df.values
    xi = np.arange(x.min(), x.max())
    
    # Fit Gaussian
    #n = len(x)                          #the number of data
    #mean = sum(x*y)/n                   #note this correction
    #sigma = sum(y*(x-mean)**2)/n        #note this correction
    mean=200
    sigma=50
    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma,0.01], maxfev=1000)

    # Spline fitting
    #spline = interp1d(x, y, kind='cubic')
    
    # Fit Lorentzian
    model = LorentzianModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)

    # Logistic function
    x2 = x.copy()
    y2 = y.copy()
    y2[y.argmax():] = y.max() 
    popt2, pcov2 = curve_fit(logifunc, x2, y2, p0=[300,150,0.1,0])
    
    
    #result.plot_fit()
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.plot(x,y,'b+:',label='data')
    #df.rolling(10, center='true').mean().plot()
    plt.plot(xi,gaus(xi,*popt),'r-',label='gaussian')
    plt.plot(xi, logifunc(xi, *popt2), 'c-',label='logistic')

    #    plt.plot(xi,spline(xi),'go:',label='spline')
    B = bloom.loc[year]
    try:
        m = B[B.sensor=='MODIS']['t[max]'].values
        plt.plot([m,m],[0,.07], '--k')
    except:
        print(' -> No MODIS')

    try:
        v = B[B.sensor=='VIIRS']['t[max]'].values
        plt.plot([v,v],[0,.07], '--b')
    except:
        print(' -> No VIIRS')        
        
    plt.legend()
    plt.title('Fit on stratification - ' + str(year))
    plt.xlabel('doy')
    plt.ylabel('drho/dz')
    # Save Figure
    fig.set_size_inches(w=12, h=6)
    outfile_year = 'Stratif_bloom_' + str(year) + '.png'
    fig.savefig(outfile_year, dpi=200)
    os.system('convert -trim ' + outfile_year + ' ' + outfile_year)

    
    
