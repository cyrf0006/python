import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
#import scipy.io
from scipy.io import loadmat
import pandas as pd
import datetime as dt

# If you don't use squeeze_me = True, then Pandas doesn't like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True fixes that.
wave_dict = loadmat('SMHI_wave_33004_201002_201003.mat',squeeze_me=True)
meteo_dict = loadmat('SMHI_meteo_55570_201002_201003.mat',squeeze_me=True)
#Tlowres_dict = loadmat('/media/cyrf0006/Seagate1TB/IOW/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
Tlowres_dict = loadmat('./data/Titp_S1_p4std_10minAve.mat',squeeze_me=True)
mooring_depth = 83

# make a new dictionary with just dependent variables we want
# (we handle the time variable separately, below)
wave_height = { k: wave_dict[k] for k in ['waveheight_max','waveheight_signif']}
wave_dir = { k: wave_dict[k] for k in ['wavedir_maxenergy']}
wave_period= { k: wave_dict[k] for k in ['waveperiod_mean']}
wind_dir = { k: meteo_dict[k] for k in ['wind_dir']}
wind_mag = { k: meteo_dict[k] for k in ['wind_mag']}
T = { k: Tlowres_dict[k] for k in ['habVec', 'Tbin']}

# Wind data from R/V Alkor
Alkor_dict = loadmat('./data/AL351_DDL_wind.mat',squeeze_me=True, struct_as_record=False)
Ualkor =  Alkor_dict['DDL'].wind.speed.data
yearday = Alkor_dict['DDL'].time.data+1
pdTimeAlkor = pd.to_datetime('2010') + pd.to_timedelta(yearday - 1, unit='d')
alkor = pd.DataFrame(Ualkor, index=pdTimeAlkor)
alkor = alkor.resample('30Min').mean()


## ---- Datetime ----- ##
# (http://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python)
def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

# convert Matlab time into list of python datetime objects and put in dictionary
# wave_datetime = [matlab2datetime(tval) for tval in wave_time]
wave_height['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]
wave_dir['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]
wave_period['date_time'] = [matlab2datetime(tval) for tval in wave_dict['wave_matlabtime']]
wind_dir['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
wind_mag['date_time'] = [matlab2datetime(tval) for tval in meteo_dict['wind_matlabtime']]
T['date_time'] = [matlab2datetime(tval) for tval in Tlowres_dict['timeVec']]
T['zVec'] = [mooring_depth-tval for tval in Tlowres_dict['habVec']]


## -- Plot -- ##

fig = plt.figure()
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
# AX1 - T contours
ctf = plt.contourf(T['date_time'], T['zVec'], T['Tbin'], 30, cmap=plt.cm.RdBu_r)
ct = plt.contour(T['date_time'], T['zVec'], T['Tbin'], 10, colors='k', linewidths=0.5)
cax = plt.axes([0.91,0.67,0.02,0.2])
plt.colorbar(ctf, cax=cax)
plt.plot(grid=True)
ax1.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00'))
ax1.xaxis.label.set_visible(False)
ax1.tick_params(labelbottom='off')
ax1.set_ylabel('Depth (m)')
ax1.invert_yaxis()
ax1.text(pd.Timestamp('2010-02-28 15:00:00'), 80, r'T ($\rm ^{\circ}C$)', horizontalalignment='left', verticalalignment='center', fontsize=14, color=[1,1,1])

# AX2 - Wind mag.
ax2 = plt.subplot2grid((6, 1), (2, 0))
df = pd.DataFrame(wind_mag)
df = df.set_index('date_time')
df.plot(ax=ax2, grid='on', legend=False)
alkor.plot(ax=ax2, grid='on', legend=False)
ax2.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00 '))
ax2.tick_params(labelbottom='off')
ax2.xaxis.label.set_visible(False)
ax2.set_ylabel(r'$\overline{U} (m/s)$')
ax2.legend(['SMHI','ship'], bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
#ax3.legend([r'$\overline{U}$'])

# AX3 - Wind/wave direction
ax3 = plt.subplot2grid((6, 1), (3, 0))
df1 = pd.DataFrame(wind_dir)
df2= pd.DataFrame(wave_dir)
df1 = df1.set_index('date_time')
df2 = df2.set_index('date_time')
df1.plot(ax=ax3, grid='on', label="winddir", legend=True)
df2.plot(secondary_y=False, ax=ax3, grid='on', label="wavedir", legend=True)
ax3.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00'))
ax3.set_ylim(85,365)
ax3.set_ylabel(r'$\rm \theta$')
ax3.xaxis.label.set_visible(False)
ax3.tick_params(labelbottom='off')
ax3.legend([r'$\rm \theta_{wind}$', r'$\rm \theta_{wave}$'], bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.yticks([90, 180, 270, 360], ['E', 'S', 'W', 'N'], rotation='horizontal')

# S3 - Wave heights
ax4 = plt.subplot2grid((6, 1), (4, 0))
df = pd.DataFrame(wave_height)
df = df.set_index('date_time')
df.plot(ax=ax4, grid='on')
#XTICKS = axes[0].get_xticks()
ax4.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00'))
ax4.tick_params(labelbottom='off')
ax4.xaxis.label.set_visible(False)
ax4.set_ylabel('H(m)')
ax4.legend([r'$\rm H_{max}$', r'$\rm H_{s}$'], bbox_to_anchor=(1.0, 1.0), loc=1, borderaxespad=0.)
# bbox_to_anchor is origin; loc=1 is inside; bordaxesoad=0 means directly on the axis

#S4 - Wave period
ax5 = plt.subplot2grid((6, 1), (5, 0))
df = pd.DataFrame(wave_period)
df = df.set_index('date_time')
df.plot(ax=ax5, grid='on', legend=False)
#XTICKS = axes[0].get_xticks()
ax5.set_xlim(pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00'))
ax5.tick_params(labelbottom='on')
ax5.set_ylabel(r'$\rm \overline{T}_s (s)$')
ax5.set_xlabel('Time')
#ax5.legend([r'$\rm \overline{T}_s$'])



fig.set_size_inches(w=6,h=6)
#fig.tight_layout()
fig_name = 'buoy_data.pdf'
fig.savefig(fig_name)
os.system('pdfcrop %s %s &> /dev/null &'%(fig_name, fig_name))


plt.show()
