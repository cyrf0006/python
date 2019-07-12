# A first test to read Excel nutrient file and export to Pandas.

# Check in:
#  /home/cyrf0006/research/SPM/python_processing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
#from seawater import extras as swx
from scipy.io import loadmat
import datetime as dt
from matplotlib import dates
import seaborn as sb
from matplotlib import cm
from scipy import stats
import os

# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)


# ---- AZFP data ---- #

# 1. 125KHz - 2017-09-07
sv_file = '/home/cyrf0006/research/SPM/Max_stuff/sv_125kHz_07_09_2017.csv'
# Set index
df = pd.read_csv(sv_file, header=0)
# Remove column
df = df.drop(df.columns[0], axis=1)
# reshape using pd.pivot (too cool!!)
df = df.pivot(index='Time_S', columns='Layer_depth_min', values='sv_linear')
# index to datetime
df.index = pd.to_datetime('2017-09-07'+df.index)
df_125 = df

# 2. 38KHz - 2017-09-07
sv_file = '/home/cyrf0006/research/SPM/Max_stuff/sv_38kHz_07_09_2017.csv'
# Set index
df = pd.read_csv(sv_file, header=0)
# Remove column
df = df.drop(df.columns[0], axis=1)
# reshape using pd.pivot (too cool!!)
df = df.pivot(index='Time_S', columns='Layer_depth_min', values='sv_linear')
# index to datetime
df.index = pd.to_datetime('2017-09-07'+df.index)
df_38 = df

# 3. 67KHz - 2017-09-07
sv_file = '/home/cyrf0006/research/SPM/Max_stuff/sv_67kHz_07_09_2017.csv'
# Set index
df = pd.read_csv(sv_file, header=0)
# Remove column
df = df.drop(df.columns[0], axis=1)
# reshape using pd.pivot (too cool!!)
df = df.pivot(index='Time_S', columns='Layer_depth_min', values='sv_linear')
# index to datetime
df.index = pd.to_datetime('2017-09-07'+df.index)
df_67 = df

# ---- VMP data ---- #
# The list of files to process
filelist = np.genfromtxt('/home/cyrf0006/research/VMP_dataprocessing/2017-09-07/20170907_eps.list', dtype=str)

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

# Empty lists
eps_list = []
peps_list = []
N2_list = []
time_list = []
fluo_list = []
prof_time = []
date_list = []
eps_list_toSv = []
N2_list_toSv = []
P_list_toSv = []


# Binned depth vector
Pbin = df.columns

# Loop on file    
for fname in filelist:
    vmp_dict = loadmat(fname,squeeze_me=True, struct_as_record=False)

    pdtime = [matlab2datetime(i) for i in vmp_dict['mtime_eps']]
    p_eps1 =  vmp_dict['p_eps1']
    eps1 = vmp_dict['eps1']
    p_eps2 =  vmp_dict['p_eps2']
    eps2 = vmp_dict['eps2']    
    N2 =  vmp_dict['N']**2
    F =  vmp_dict['FLUORO']

    # Check depth vector
    if np.size(p_eps1) == np.size(p_eps2):
        p_eps = p_eps1
    else:
        print 'Propblem with p_eps'
        
    # "Selection average" of eps1 and eps2
    eps1[np.where(eps1>10*eps2)]=np.nan
    eps2[np.where(eps2>10*eps1)]=np.nan
    eps = np.nanmean((eps1, eps2), axis=0)

    # bin to echogram (necessary to have regular grid)
    digitized = np.digitize(p_eps, Pbin) #<- this is awesome!
    eps_list_toSv.append([eps[digitized == i].mean() for i in range(0, len(Pbin))])
    N2_list_toSv.append([N2[digitized == i].mean() for i in range(0, len(Pbin))])
    P_list_toSv.append([p_eps[digitized == i].mean() for i in range(0, len(Pbin))])

    # beginning of the cast (will become df.index)
    prof_time.append(pdtime[0])
    
    # Append all vectors
    date_list.append(pdtime)
    eps_list.append(eps)
    peps_list.append(p_eps)
    N2_list.append(N2)
    fluo_list.append(F)


# Get depth vector
df_peps = pd.DataFrame(peps_list)
zvec_eps = df_peps.mean(axis=0)

# All list to DataFrame
df_eps = pd.DataFrame(eps_list)
df_eps.index = pd.to_datetime(prof_time).round('S')
df_eps.columns = zvec_eps
df_N2 = pd.DataFrame(N2_list)
df_N2.index = pd.to_datetime(prof_time).round('S')
df_N2.columns = zvec_eps
df_fluo = pd.DataFrame(fluo_list)
df_fluo.index = pd.to_datetime(prof_time).round('S')
df_fluo.columns = zvec_eps
# compute diffusivity
df_K = 0.2*df_eps/(df_N2);


#df.to_pickle('MSS_S1.pkl')
# df = pd.read_pickle(file_name) to read back

## -----  get Sv that matches epsilon time -----##
df_eps2Sv = pd.DataFrame(eps_list_toSv)
df_eps2Sv.index = pd.to_datetime(prof_time).round('S')
df_eps2Sv.columns = Pbin
df_N22Sv = pd.DataFrame(N2_list_toSv)
df_N22Sv.index = pd.to_datetime(prof_time).round('S')
df_N22Sv.columns = Pbin
df_P2Sv = pd.DataFrame(P_list_toSv)
df_P2Sv.index = pd.to_datetime(prof_time).round('S')
df_P2Sv.columns = Pbin
df_K2Sv = 0.2*df_eps2Sv/df_N22Sv

idx = df_125.index.searchsorted(df_eps2Sv.index)
df_125_2_eps = df_125.loc[df_125.index[idx]]
idx = df_38.index.searchsorted(df_eps2Sv.index)
df_38_2_eps = df_38.loc[df_38.index[idx]]
idx = df_67.index.searchsorted(df_eps2Sv.index)
df_67_2_eps = df_67.loc[df_67.index[idx]]


df = pd.DataFrame({'eps':np.log10(df_eps2Sv.values.flatten()),
                   'K':np.log10(df_K2Sv.values.flatten()),
                   'N2':np.log10(df_N22Sv.values.flatten()),
                   'P':df_P2Sv.values.flatten(),
                   'Sv125':10*np.log10(df_125_2_eps.values.flatten()),
                   'Sv38':10*np.log10(df_38_2_eps.values.flatten()),
                   'Sv67':10*np.log10(df_67_2_eps.values.flatten())})
df = df.dropna(how='any')
df = df[df['Sv125']>-100] # remove erroneous Sv
df = df[df['Sv38']>-100] # remove erroneous Sv
df = df[df['Sv67']>-100] # remove erroneous Sv
#df = df[df.P>=40]

df_ratio = pd.DataFrame({'eps':df.eps, 'Sv125_38':df.Sv125/df.Sv38})

keyboard


## -------- Sv contours and VMP and Plot -------- ##
fig = plt.figure(2)
days = dates.DayLocator() # AX0 - AX1
hours6 = dates.HourLocator(interval=6)
hours1 = dates.HourLocator(interval=1)
hours2 = dates.HourLocator(interval=2)
dfmt = dates.DateFormatter('%H:%M')

# unstack to vectors
eps_vec = df_eps.unstack()
# get x,y
x = []
y = []
for i in eps_vec.index.values:
    y.append(i[0])
    x.append(i[1])

ax0 = plt.subplot2grid((1, 1), (0, 0))
levels_sv = np.linspace(-10, -5, 21)
c = ax0.contourf(df_125.index, df_125.columns, np.log10(df_125.T), levels_sv, cmap=plt.cm.gray_r)
#c = ax0.contourf(df_sv.index, df_sv.columns, np.log10(df_sv.T), levels_sv, cmap=plt.cm.Paired)
#plt.colorbar(c)
c2 = ax0.scatter(x, y, c=np.log10(eps_vec), cmap=plt.cm.RdBu_r, alpha=0.4)
cb = plt.colorbar(c2)
ax0.set_ylim(0,105)
ax0.invert_yaxis()
ax0.xaxis.set_major_formatter(dfmt)
ax0.xaxis.set_major_locator(hours1)
ax0.set_ylabel(r'Depth (m)')
ax0.set_xlabel(r'7 Sept. 2017')
cb.ax.set_ylabel(r'$\epsilon {\rm (W Kg^{-1})}$', fontsize=20, fontweight='bold')
#plt.show()
fig.set_size_inches(w=18, h=6)
fig_name = 'Sv_eps_toberename.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


## --------------- BASIC SCATTER PLOT ---------------###
m1,b1 = np.polyfit(df.eps, df.Sv125, 1)
m2,b2 = np.polyfit(df.eps, df.Sv38, 1)
fig = plt.figure(1)
ax = plt.subplot2grid((1, 1), (0, 0))
#c2 = ax.scatter(np.log10(df_eps2Sv.values), 10*np.log10(df_125_2_eps.values), c='r', alpha=0.5)
#c2 = ax.scatter(np.log10(df_eps2Sv.values), 10*np.log10(df_38_2_eps.values), c='k', alpha=0.5)
c2 = ax.scatter(df.eps, df.Sv125, c='r', alpha=0.5)
c2 = ax.scatter(df.eps, df.Sv38, c='k', alpha=0.5)
ax.plot(df.eps.sort_values(), df.eps.sort_values()*m1+b1, 'r--', alpha=0.8)
ax.plot(df.eps.sort_values(), df.eps.sort_values()*m2+b2, 'k--', alpha=0.8)
#cb = plt.colorbar(c2)
#cb.ax.set_ylabel(r'Depth (m)', fontsize=20, fontweight='bold')            
plt.legend(['125kHz', '38kHz'])
ax.set_xlabel(r'$\epsilon {\rm  (W Kg^{-1})}$')
ax.set_ylabel('Sv - 125kHz (dB)')
plt.show()

# Ratio 125kHz/38khz
fig = plt.figure(1)
m,b = np.polyfit(df.eps, df.Sv125/df.Sv38, 1)
ax = plt.subplot2grid((1, 1), (0, 0))
c2 = ax.scatter(df.eps, df.Sv125/df.Sv38, c=df.P, alpha=0.5)
ax.plot(df.eps.sort_values(), df.eps.sort_values()*m+b, 'k--', alpha=0.8)
cb = plt.colorbar(c2)
cb.ax.set_ylabel(r'Depth (m)', fontsize=20, fontweight='bold')            
ax.set_xlabel(r'$\epsilon {\rm  (W Kg^{-1})}$')
ax.set_ylabel('Sv_125kHz / Sv_38kHz')
ax.grid()
plt.show()


## ----  KDE using Seaborn ---- ##
fig = plt.figure(1)
ax = plt.subplot2grid((1, 1), (0, 0))
#sb.kdeplot(df.eps, df.Sv38, cmap="Greys", shade=True, shade_lowest=False, alpha=0.7)
sb.kdeplot(df.eps, df.Sv125, cmap="Reds", shade=True, shade_lowest=False, alpha=0.7)
plt.show()

## Correlation
sb.jointplot(x = 'eps',y = 'Sv125',data = df, kind='kde', cmap="Reds", shade_lowest=False)
sb.jointplot(x = 'eps',y = 'Sv38',data = df, kind='kde', cmap="Blues", shade_lowest=False)
plt.show()

## ----  KDE using matplotlib ---- ##
xmin = df.eps.min()
xmax = df.eps.max()
ymin = df.Sv125.min()
ymax = df.Sv125.max()
X1, Y1 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X1.ravel(), Y1.ravel()])
values = np.vstack([df.eps.values, df.Sv125.values])
kernel = stats.gaussian_kde(values)
Z1 = np.reshape(kernel(positions).T, X1.shape)
ymin = df.Sv38.min()
ymax = df.Sv38.max()
X2, Y2 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X2.ravel(), Y2.ravel()])
values = np.vstack([df.eps.values, df.Sv38.values])
kernel = stats.gaussian_kde(values)
Z2 = np.reshape(kernel(positions).T, X2.shape)
ymin = df.Sv67.min()
ymax = df.Sv67.max()
X3, Y3 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X3.ravel(), Y3.ravel()])
values = np.vstack([df.eps.values, df.Sv67.values])
kernel = stats.gaussian_kde(values)
Z3 = np.reshape(kernel(positions).T, X3.shape)

m1,b1 = np.polyfit(df.eps, df.Sv125, 1)
m2,b2 = np.polyfit(df.eps, df.Sv38, 1)

fig = plt.figure(1)
ax = plt.subplot2grid((1, 1), (0, 0))
#ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
ax.contour(X1, Y1, Z1, cmap=plt.cm.Reds, extent=[xmin, xmax, ymin, ymax])
#ax.contour(X3, Y3, Z3, cmap=plt.cm.Oranges, extent=[xmin, xmax, ymin, ymax])
ax.contour(X2, Y2, Z2, cmap=plt.cm.Blues, extent=[xmin, xmax, ymin, ymax])
ax.plot(df.eps, df.Sv125, 'r.', markersize=2, alpha=0.8)
#ax.plot(df.eps, df.Sv67, 'y.', markersize=2, alpha=0.8)
ax.plot(df.eps, df.Sv38, 'b.', markersize=2, alpha=0.8)
ax.plot(df.eps.sort_values(), df.eps.sort_values()*m1+b1, 'r--', alpha=0.8)
ax.plot(df.eps.sort_values(), df.eps.sort_values()*m2+b2, 'b--', alpha=0.8)
ax.set_xlim([-9.5, -4.5])
ax.set_ylim([-105, -50])
#plt.legend(['125kHz', '67kHz', '38kHz'])
plt.legend(['125kHz', '38kHz'])
ax.set_xlabel(r'$log_{10}(\epsilon)\,{\rm  (W\,Kg^{-1})}$')
ax.set_ylabel('Sv - 125kHz (dB)')
plt.show()

## ----  KDE with Diffusivity ---- ##
xmin = df.K.min()
xmax = df.K.max()
ymin = df.Sv125.min()
ymax = df.Sv125.max()
X1, Y1 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X1.ravel(), Y1.ravel()])
values = np.vstack([df.K.values, df.Sv125.values])
kernel = stats.gaussian_kde(values)
Z1 = np.reshape(kernel(positions).T, X1.shape)
ymin = df.Sv38.min()
ymax = df.Sv38.max()
X2, Y2 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X2.ravel(), Y2.ravel()])
values = np.vstack([df.K.values, df.Sv38.values])
kernel = stats.gaussian_kde(values)
Z2 = np.reshape(kernel(positions).T, X2.shape)
ymin = df.Sv67.min()
ymax = df.Sv67.max()
X3, Y3 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X3.ravel(), Y3.ravel()])
values = np.vstack([df.K.values, df.Sv67.values])
kernel = stats.gaussian_kde(values)
Z3 = np.reshape(kernel(positions).T, X3.shape)

m1,b1 = np.polyfit(df.K, df.Sv125, 1)
m2,b2 = np.polyfit(df.K, df.Sv38, 1)

fig = plt.figure(1)
ax = plt.subplot2grid((1, 1), (0, 0))
ax.contour(X1, Y1, Z1, cmap=plt.cm.Reds, extent=[xmin, xmax, ymin, ymax])
#ax.contour(X3, Y3, Z3, cmap=plt.cm.Oranges, extent=[xmin, xmax, ymin, ymax])
ax.contour(X2, Y2, Z2, cmap=plt.cm.Blues, extent=[xmin, xmax, ymin, ymax])
ax.plot(df.K, df.Sv125, 'r.', markersize=2, alpha=0.8)
#ax.plot(df.K, df.Sv67, 'y.', markersize=2, alpha=0.8)
ax.plot(df.K, df.Sv38, 'b.', markersize=2, alpha=0.8)
ax.plot(df.K.sort_values(), df.eps.sort_values()*m1+b1, 'r--', alpha=0.8)
ax.plot(df.K.sort_values(), df.eps.sort_values()*m2+b2, 'b--', alpha=0.8)
#ax.set_xlim([-9.5, -4.5])
#ax.set_ylim([-105, -50])
#plt.legend(['125kHz', '67kHz', '38kHz'])
plt.legend(['125kHz', '38kHz'])
ax.set_xlabel(r'$log_{10}(K)\,{\rm  (m^2\,s^{-1})}$')
ax.set_ylabel('Sv - 125kHz (dB)')
plt.show()

## ----  KDE with N2 ---- ##
xmin = df.N2.min()
xmax = df.N2.max()
ymin = df.Sv125.min()
ymax = df.Sv125.max()
X1, Y1 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X1.ravel(), Y1.ravel()])
values = np.vstack([df.N2.values, df.Sv125.values])
kernel = stats.gaussian_kde(values)
Z1 = np.reshape(kernel(positions).T, X1.shape)
ymin = df.Sv38.min()
ymax = df.Sv38.max()
X2, Y2 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X2.ravel(), Y2.ravel()])
values = np.vstack([df.N2.values, df.Sv38.values])
kernel = stats.gaussian_kde(values)
Z2 = np.reshape(kernel(positions).T, X2.shape)
ymin = df.Sv67.min()
ymax = df.Sv67.max()
X3, Y3 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X3.ravel(), Y3.ravel()])
values = np.vstack([df.N2.values, df.Sv67.values])
kernel = stats.gaussian_kde(values)
Z3 = np.reshape(kernel(positions).T, X3.shape)

m1,b1 = np.polyfit(df.N2, df.Sv125, 1)
m2,b2 = np.polyfit(df.N2, df.Sv38, 1)

fig = plt.figure(1)
ax = plt.subplot2grid((1, 1), (0, 0))
ax.contour(X1, Y1, Z1, cmap=plt.cm.Reds, extent=[xmin, xmax, ymin, ymax])
#ax.contour(X3, Y3, Z3, cmap=plt.cm.Oranges, extent=[xmin, xmax, ymin, ymax])
ax.contour(X2, Y2, Z2, cmap=plt.cm.Blues, extent=[xmin, xmax, ymin, ymax])
ax.plot(df.N2, df.Sv125, 'r.', markersize=2, alpha=0.8)
#ax.plot(df.N2, df.Sv67, 'y.', markersize=2, alpha=0.8)
ax.plot(df.N2, df.Sv38, 'b.', markersize=2, alpha=0.8)
ax.plot(df.N2.sort_values(), df.eps.sort_values()*m1+b1, 'r--', alpha=0.8)
ax.plot(df.N2.sort_values(), df.eps.sort_values()*m2+b2, 'b--', alpha=0.8)
#ax.set_xlim([-9.5, -4.5])
#ax.set_ylim([-105, -50])
#plt.legend(['125kHz', '67kHz', '38kHz'])
plt.legend(['125kHz', '38kHz'])
ax.set_xlabel(r'$log_{10}(N2)\,{\rm  (s^{-2})}$')
ax.set_ylabel('Sv - 125kHz (dB)')
plt.show()

# --------   VMP contours only (works well!) -- ##
dfmt = dates.DateFormatter('%H:%S')
hours1 = dates.HourLocator(interval=1)
fig = plt.figure(1)
# epsilon

# resample data 
EPS = df_eps.resample('20Min').mean()
# Casts identifyer
casts = np.ones(len(EPS.index))
castsID = pd.DataFrame(casts, index=EPS.index) 

# plot
ax4 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
levels = np.linspace(-9,-5, 21)
c = plt.contourf(EPS.index, EPS.columns, np.log10(np.transpose(EPS)), levels, cmap=plt.cm.RdBu_r, extend="both")
cax = plt.axes([0.91,0.15,0.02,0.7])
#plt.colorbar(c, cax=cax, ticks=[-10, -9, -8, -7, -6])
plt.colorbar(c, cax=cax, ticks=[np.linspace(-9, -4, 6)])
ax4.text(pd.Timestamp('2010-03-01 00:00:00'), 10, r'$\rm log_{10}(\epsilon / W Kg^{-1})$', horizontalalignment='left', verticalalignment='center', fontsize=14, color='k')
ax4.set_ylabel(r'Depth (m)')
#ax4.set_xlim(XLIM)
#ax4.xaxis.set_major_locator(days)
ax4.xaxis.set_major_formatter(dfmt)
ax4.xaxis.set_major_locator(hours1)
#ax4.set_ylim(0, 88)
ax4.invert_yaxis()
#ax4.plot([storm2, storm2], [0, 88], '--k')
plt.show()


