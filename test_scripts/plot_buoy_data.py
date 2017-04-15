import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
#import scipy.io
from scipy.io import loadmat

# If you don't use squeeze_me = True, then Pandas doesn't like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True fixes that.
wave_dict = loadmat('SMHI_wave_33004_201002_201003.mat',squeeze_me=True)
meteo_dict = loadmat('SMHI_meteo_55570_201002_201003.mat',squeeze_me=True)
Tlowres_dict = loadmat('/media/cyrf0006/Seagate1TB/IOW/Titp_S1_p4std_10minAve.mat',squeeze_me=True)

# make a new dictionary with just dependent variables we want
# (we handle the time variable separately, below)
wave_height = { k: wave_dict[k] for k in ['waveheight_max','waveheight_signif']}
wave_dir = { k: wave_dict[k] for k in ['wavedir_maxenergy']}
wave_period= { k: wave_dict[k] for k in ['waveperiod_mean']}
wind_dir = { k: meteo_dict[k] for k in ['wind_dir']}
wind_mag = { k: meteo_dict[k] for k in ['wind_mag']}
T = { k: Tlowres_dict[k] for k in ['habVec', 'Tbin']}

## ---- Datetime ----- ##
# (http://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python)
import pandas as pd
import datetime as dt
def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

# convert Matlab time into list of python datetime objects and put in dictionary
#wave_datetime = [matlab2datetime(tval) for tval in wave_time]
wave_height['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]
wave_dir['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]
wave_period['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]
wind_dir['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
wind_mag['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
T['date_time'] = [matlab2datetime(tval) for tval in Tlowres_dict['timeVec']]

# plot with Pandas
#DataFrame.plot(x=None, y=None, kind='line', ax=None, subplots=False, sharex=None, sharey=False, layout=None, figsize=None, use_index=True, title=None, grid=None, legend=True, style=None, logx=False, logy=False, loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False, **kwds)[source]

## -- Plot -- ##
fig, axes = plt.subplots(nrows=6, ncols=1)

# S0
plt.axes(axes[0], rowspan=2)
mycontourf = plt.contourf(T['date_time'], T['habVec'], T['Tbin'], 30, cmap=plt.cm.RdBu_r)
mycontour = plt.contour(T['date_time'], T['habVec'], T['Tbin'], 10, colors='k')
cl = plt.colorbar(mycontourf, orientation='horizontal')
plt.plot(grid=True)
axes[0].set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 12:00:00'))
axes[0].xaxis.label.set_visible(False)
axes[0].tick_params(labelbottom='off')


# S1 - Wind/wave direction
df1 = pd.DataFrame(wind_dir)
df2= pd.DataFrame(wave_dir)
df1 = df1.set_index('date_time')
df2 = df2.set_index('date_time')
df1.plot(ax=axes[1], grid='on', label="winddir", legend=True)
df2.plot(secondary_y=False, ax=axes[1], grid='on', label="wavedir", legend=True)
axes[1].set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 12:00:00'))
axes[1].set_ylabel('Angle (deg)')
axes[1].xaxis.label.set_visible(False)
axes[1].tick_params(labelbottom='off')

#S2 - Wind mag.
df = pd.DataFrame(wind_mag)
df = df.set_index('date_time')
df.plot(ax=axes[2], grid='on')
#XTICKS = axes[0].get_xticks()
axes[2].set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 12:00:00 '))
axes[2].tick_params(labelbottom='off')
axes[2].xaxis.label.set_visible(False)
axes[2].set_ylabel('U_wind (m/s)')

# S3 - Wave heights
df = pd.DataFrame(wave_height)
df = df.set_index('date_time')
df.plot(ax=axes[3], grid='on')
#XTICKS = axes[0].get_xticks()
axes[3].set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 12:00:00'))
axes[3].tick_params(labelbottom='off')
axes[3].xaxis.label.set_visible(False)
axes[3].set_ylabel('H (m)')

#S4 - Wave period
df = pd.DataFrame(wave_period)
df = df.set_index('date_time')
df.plot(ax=axes[4], grid='on')
#XTICKS = axes[0].get_xticks()
axes[4].set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 12:00:00'))
axes[4].tick_params(labelbottom='on')
axes[4].set_ylabel('Period (s)')
axes[4].set_xlabel('Time')



#plt.savefig('exercise-T-contour.png')
plt.show()
