## This script loads all MSS profiles within a list, clean and store them in a h5 file.
#
# F. Cyr - July 2017

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
import datetime
import pandas as pd
#import seawater as sw
import gsw


# The list of MSS files to process
filelist = np.genfromtxt('MSS_S1.list', dtype=str)

# Empty lists
EPSlist = []
N2list = []
datelist = []

# Binned depth vector
Pbin = np.arange(1, 100, 2)

# Loop on file    
for fname in filelist:
    MSS_dict = loadmat(fname,squeeze_me=True, struct_as_record=False)
    print MSS_dict
    no_cast = MSS_dict['MIX'].size

    # Loop on casts
    for i in range(0, no_cast):
         
        P =  MSS_dict['MIX'][i].P
        EPS =  MSS_dict['MIX'][i].eps
        N2 =  MSS_dict['MIX'][i].N2
        date =  pd.Timestamp(MSS_dict['STA'][i].date)

        digitized = np.digitize(P, Pbin) #<- this is awesome!
        EPSlist.append([EPS[digitized == i].mean() for i in range(0, len(Pbin))])
        N2list.append([N2[digitized == i].mean() for i in range(0, len(Pbin))])
        datelist.append(date)

# List2Array
EPSarray = np.transpose(np.array(EPSlist))
N2array = np.transpose(np.array(N2list))
MSStime = datelist

# Pickle MSS data (linear scale)
df = pd.DataFrame(np.transpose(EPSarray), index=MSStime, columns=Pbin)
df.to_pickle('MSS_S1.pkl')
# df = pd.read_pickle(file_name) to read back

## ---- Output figure ---- ##
# Create LogScale DataFrame
EPSlog = np.log10(EPSarray)
EPS = pd.DataFrame(np.transpose(EPSlog), index=MSStime, columns=Pbin)

# Casts identifyer
casts = np.ones(len(MSStime))
castsID = pd.DataFrame(casts, index=MSStime) 


# resample data 
EPS = EPS.resample('60Min').mean()

XLIM = [pd.Timestamp('2010-02-28 12:00:00'), pd.Timestamp('2010-03-04 06:00:00')]
days = dates.DayLocator()
hours = dates.HourLocator(interval=6)
dfmt = dates.DateFormatter('%d')

fig = plt.figure(2)
ax1 = plt.subplot2grid((1, 1), (0, 0))
#levels = np.linspace(-10,-6, 19)
levels = np.linspace(-9.5,-6.5, 20)
c = plt.contourf(EPS.index, EPS.columns, np.transpose(EPS), levels, cmap=plt.cm.RdBu_r, extend="both")
cax = plt.axes([0.92,0.2,0.02,0.6])
#plt.colorbar(c, cax=cax, ticks=[-10, -9, -8, -7, -6])
plt.colorbar(c, cax=cax, ticks=[np.linspace(-9.5, -6.5, 7)])
ax1.text(pd.Timestamp('2010-03-01 00:00:00'), 78, r'$\rm log_{10}(\epsilon / W Kg^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
ax1.set_ylabel(r'Depth (m)', fontsize=14)
ax1.set_xlabel(r'March 2010', fontsize=14)
ax1.set_xlim(XLIM)
ax1.xaxis.set_major_locator(days)
ax1.xaxis.set_major_formatter(dfmt)
ax1.xaxis.set_minor_locator(hours)
ax1.set_ylim(0, 88)
ax1.invert_yaxis()

# Mark casts
ax1.plot(castsID, '|', color=[.3,.3,.3])


#### --------- Save Figure ------------- ####
fig_name = 'MSS_S1.eps'
fig.set_size_inches(w=9, h=6)
fig.set_dpi(150)
#fig.tight_layout()
fig.savefig(fig_name)
print ' '
print ' --> check ' + fig_name + ' for results'
print ' --> ' + np.str(castsID.size) + ' MSS casts used'
