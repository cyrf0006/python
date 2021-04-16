# -*- coding: utf-8 -*-
'''
This script produces custom figures from analysis of Stn27 data as part of the capelin spawning project.

** see azmp_stn27.py and azmp_stn27_explore.py azmp_stn27_analysis.py for more options and ways to explore the dataset


Frederic.Cyr@dfo-mpo.gc.ca - July 2020

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import os
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
import cmocean
import gsw


# preamble
year_clim = [1991, 2020]

#### ------------- T-S ---------------- ####
# Load pickled data generated in azmp_stn27.py
df_temp = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_temperature_monthly.pkl')
df_sal = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_salinity_monthly.pkl')
df_temp = df_temp[df_temp.index.year>=1950]
df_sal = df_sal[df_sal.index.year>=1950]

# Temperature
#df_temp = df_temp.resample('As').mean() # *
T_0_btm = df_temp.mean(axis=1)
T_0_50 = df_temp[df_temp.columns[(df_temp.columns<=50)]].mean(axis=1)
T_0_20 = df_temp[df_temp.columns[(df_temp.columns<=20)]].mean(axis=1)
T_150_btm = df_temp[df_temp.columns[(df_temp.columns>=150)]].mean(axis=1)
# Salinity
#df_sal = df_sal.resample('As').mean() # *
S_0_btm = df_sal.mean(axis=1)
S_0_20 = df_sal[df_sal.columns[(df_sal.columns<=20)]].mean(axis=1)
S_0_50 = df_sal[df_sal.columns[(df_sal.columns<=50)]].mean(axis=1)
S_150_btm = df_sal[df_sal.columns[(df_sal.columns>=150)]].mean(axis=1)
# Merge 
df_T = pd.concat([T_0_btm, T_0_20, T_0_50, T_150_btm], axis=1, keys=['Temp 0-176m', 'Temp 0-20m', 'Temp 0-50m', 'Temp 150-176m'])
df_S = pd.concat([S_0_btm, S_0_20, S_0_50, S_150_btm], axis=1, keys=['Sal 0-176m', 'Sal 0-20m', 'Sal 0-50m', r'Sal 150-176m ${~}$'])
# Save monthly to csv
df_T.to_csv('stn27_monthly_T.csv', float_format='%.3f')
df_S.to_csv('stn27_monthly_S.csv', float_format='%.3f')
# Save Feb-June to csv
df_T_spring = df_T[(df_T.index.month>=2) & (df_T.index.month<=6)]
df_S_spring = df_S[(df_S.index.month>=2) & (df_S.index.month<=6)]
df_T_spring.resample('As').mean().to_csv('stn27_Feb-June_T.csv', float_format='%.3f')
df_S_spring.resample('As').mean().to_csv('stn27_Feb-June_S.csv', float_format='%.3f')
# plot
plt.clf()
fig=plt.figure()
df_T_spring['Temp 0-20m'].plot()   
df_T_spring['Temp 0-20m'].resample('As').mean().plot()  
plt.legend(['Feb-June monthly', 'Feb-June annual'])
plt.ylabel(r'T($^{\circ}$C)')
fig_name = 'Feb-June_temp_stn27.png'
fig.savefig(fig_name, dpi=300)

# T-S anomalies
# Temperature anomalies
df_T_stack = df_T_spring.groupby([(df_T_spring.index.year),(df_T_spring.index.month)]).mean()
df_T_stack_ave = df_T_stack['Temp 0-176m']
df_T_stack_surf = df_T_stack['Temp 0-20m']
df_T_stack_bot = df_T_stack['Temp 150-176m']
df_T_unstack_ave = df_T_stack_ave.unstack()
df_T_unstack_surf = df_T_stack_surf.unstack()
df_T_unstack_bot = df_T_stack_bot.unstack()
T_clim_period = df_T_spring[(df_T_spring.index.year>=year_clim[0]) & (df_T_spring.index.year<=year_clim[1])]
T_monthly_stack = T_clim_period.groupby([(T_clim_period.index.year),(T_clim_period.index.month)]).mean()
T_monthly_clim = T_monthly_stack.mean(level=1)
T_monthly_std = T_monthly_stack.std(level=1)
Tave_monthly_anom = df_T_unstack_ave - T_monthly_clim['Temp 0-176m']
Tsurf_monthly_anom = df_T_unstack_surf - T_monthly_clim['Temp 0-20m']
Tbot_monthly_anom = df_T_unstack_bot - T_monthly_clim['Temp 150-176m']
Tave_monthly_stdanom = Tave_monthly_anom / T_monthly_std['Temp 0-176m']
Tsurf_monthly_stdanom = Tsurf_monthly_anom / T_monthly_std['Temp 0-20m']
Tbot_monthly_stdanom = Tbot_monthly_anom / T_monthly_std['Temp 150-176m']
Tave_anom = Tave_monthly_anom.mean(axis=1) 
Tsurf_anom = Tsurf_monthly_anom.mean(axis=1) 
Tbot_anom = Tbot_monthly_anom.mean(axis=1) 
Tave_anom.index = pd.to_datetime(Tave_anom.index, format='%Y')
Tsurf_anom.index = pd.to_datetime(Tsurf_anom.index, format='%Y')
Tbot_anom.index = pd.to_datetime(Tbot_anom.index, format='%Y')
Tave_anom_std = Tave_monthly_stdanom.mean(axis=1) 
Tsurf_anom_std = Tsurf_monthly_stdanom.mean(axis=1) 
Tbot_anom_std = Tbot_monthly_stdanom.mean(axis=1) 
Tave_anom_std.index = pd.to_datetime(Tave_anom_std.index, format='%Y')
Tsurf_anom_std.index = pd.to_datetime(Tsurf_anom_std.index, format='%Y')
Tbot_anom_std.index = pd.to_datetime(Tbot_anom_std.index, format='%Y')
# Salinity anomalies
df_S_stack = df_S_spring.groupby([(df_S_spring.index.year),(df_S_spring.index.month)]).mean()
df_S_stack_ave = df_S_stack['Sal 0-176m']
df_S_stack_surf = df_S_stack['Sal 0-20m']
df_S_stack_bot = df_S_stack['Sal 150-176m ${~}$']
df_S_unstack_ave = df_S_stack_ave.unstack()
df_S_unstack_surf = df_S_stack_surf.unstack()
df_S_unstack_bot = df_S_stack_bot.unstack()
S_clim_period = df_S_spring[(df_S_spring.index.year>=year_clim[0]) & (df_S_spring.index.year<=year_clim[1])]
S_monthly_stack = S_clim_period.groupby([(S_clim_period.index.year),(S_clim_period.index.month)]).mean()
S_monthly_clim = S_monthly_stack.mean(level=1)
S_monthly_std = S_monthly_stack.std(level=1)
Save_monthly_anom = df_S_unstack_ave - S_monthly_clim['Sal 0-176m']
Ssurf_monthly_anom = df_S_unstack_surf - S_monthly_clim['Sal 0-20m']
Sbot_monthly_anom = df_S_unstack_bot - S_monthly_clim['Sal 150-176m ${~}$']
Save_monthly_stdanom = Save_monthly_anom / S_monthly_std['Sal 0-176m']
Ssurf_monthly_stdanom = Ssurf_monthly_anom / S_monthly_std['Sal 0-20m']
Sbot_monthly_stdanom = Sbot_monthly_anom / S_monthly_std['Sal 150-176m ${~}$']
Save_anom = Save_monthly_anom.mean(axis=1) 
Ssurf_anom = Ssurf_monthly_anom.mean(axis=1) 
Sbot_anom = Sbot_monthly_anom.mean(axis=1) 
Save_anom.index = pd.to_datetime(Save_anom.index, format='%Y')
Ssurf_anom.index = pd.to_datetime(Ssurf_anom.index, format='%Y')
Sbot_anom.index = pd.to_datetime(Sbot_anom.index, format='%Y')
Save_anom_std = Save_monthly_stdanom.mean(axis=1) 
Ssurf_anom_std = Ssurf_monthly_stdanom.mean(axis=1) 
Sbot_anom_std = Sbot_monthly_stdanom.mean(axis=1) 
Save_anom_std.index = pd.to_datetime(Save_anom_std.index, format='%Y')
Ssurf_anom_std.index = pd.to_datetime(Ssurf_anom_std.index, format='%Y')
Sbot_anom_std.index = pd.to_datetime(Sbot_anom_std.index, format='%Y')
# merge anomalies
T_anom = pd.concat([Tave_anom,Tsurf_anom,Tbot_anom], axis=1, keys=['Temp 0-176m','Temp 0-20m','Temp 150-176m'])
T_anom_std = pd.concat([Tave_anom_std,Tsurf_anom_std,Tbot_anom_std], axis=1, keys=['Temp 0-176m','Temp 0-20m','Temp 150-176m'])
T_anom = T_anom[T_anom.index.year>1947]
T_anom_std = T_anom_std[T_anom_std.index.year>=1947]
S_anom = pd.concat([Save_anom,Ssurf_anom,Sbot_anom], axis=1, keys=['Sal 0-176m','Sal 0-20m','Sal 150-176m ${~}$'])
S_anom_std = pd.concat([Save_anom_std,Ssurf_anom_std,Sbot_anom_std], axis=1, keys=['Sal 0-176m','Sal 0-20m','Sal 150-176m ${~}$'])
S_anom = S_anom[S_anom.index.year>=1947]
S_anom_std = S_anom_std[S_anom_std.index.year>=1947]
# Save anomalies
T_anom_std.to_csv('stn27_Feb-June_T_std_anomaly.csv', float_format='%.3f')
S_anom_std.to_csv('stn27_Feb-June_S_std_anomaly.csv', float_format='%.3f')

plt.close('all')
fig = plt.figure(1)
ax = T_anom_std['Temp 0-20m'].plot()
T_anom_std['Temp 0-20m'].rolling(5, min_periods=2).mean().plot(ax=ax)
plt.legend(['annual','5 yr rolling mean'])
plt.title('Stn27 - 0-20m temperature normalized anomalies')
fig_name = 'Feb-June_temp_stn27_stn_anom.png'
fig.savefig(fig_name, dpi=300)

#### ------------- Comparision with Carscadden ---------------- ####
df_c = pd.read_csv('/home/cyrf0006/research/capelin/Carscadden1997_temperatures.csv', index_col='year') 
df_c = df_c.stack()
y = df_c.index.get_level_values(0) 
m = df_c.index.get_level_values(1) 
df_c = df_c.reset_index()
df_c.index = pd.to_datetime(y.astype('str') + '0' + m, format='%Y%m')
df_c = df_c[0]
# Load 6NM temperature
df_tmp = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/stn27_pickles_0.1nm_radius/S27_temperature_monthly.pkl')
df_tmp = df_tmp[df_tmp.index.year>=1950]
T_6NM = df_tmp[df_tmp.columns[(df_tmp.columns<=20)]].mean(axis=1)
# plot
plt.close('all')
fig = plt.figure(1)
ax = df_c.plot(style='ro')   
T_6NM.plot(ax=ax, style='.-k', alpha=1)
df_T['Temp 0-20m'].plot(ax=ax, style='.-b', alpha=.6)
plt.ylabel(r'$\rm T_{0-20m} (^{\circ}C)$')
plt.legend(['Carscadden1997', 'stn27_6NM radius', 'stn27_1.5NM radius']) 

#### ------------- MLD ---------------- ####
MLD_monthly = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_MLD_monthly.pkl')
MLD_monthly = MLD_monthly[MLD_monthly.index.year>=1950]
MLD_apr = MLD_monthly[MLD_monthly.index.month==4]
MLD_june = MLD_monthly[MLD_monthly.index.month==6]
MLD_july = MLD_monthly[MLD_monthly.index.month==7]

# Save MLD
MLD_apr.index = MLD_apr.index.year
MLD_june.index = MLD_june.index.year
MLD_july.index = MLD_july.index.year
MLD = pd.concat([MLD_june, MLD_july], axis=1, keys=['MLD June','MLD July'])
MLD.to_csv('stn27_MLD_June-July.csv', float_format='%.1f')

# plot MLD
plt.close('all')
fig = plt.figure(1)
ax = MLD_june.rolling(5, min_periods=2).mean().plot(color='tab:blue',)
MLD_july.rolling(5, min_periods=2).mean().plot(color='tab:orange', ax=ax)
plt.legend(['June','July'])
ax.grid()
MLD_june.plot(color='tab:blue', marker = '.', linestyle = '', ax=ax)
MLD_july.plot(color='tab:orange', marker= '.', linestyle = '', ax=ax)
plt.title('Mixed Layer Depth')
fig_name = 'MLD_stn27_June-July.png'
fig.savefig(fig_name, dpi=300)

# plot MLD
plt.close('all')
fig = plt.figure(1)
ax = MLD_apr.rolling(5, min_periods=2).mean().plot(color='tab:blue',)
MLD_june.rolling(5, min_periods=2).mean().plot(color='tab:orange',)
MLD_july.rolling(5, min_periods=2).mean().plot(color='tab:red', ax=ax)
plt.legend(['April', 'June','July'])
MLD_apr.plot(color='tab:blue', marker = '.', linestyle = '', ax=ax)
MLD_june.plot(color='tab:orange', marker = '.', linestyle = '', ax=ax)
MLD_july.plot(color='tab:red', marker= '.', linestyle = '', ax=ax)
plt.grid()
plt.title('Mixed Layer Depth')
fig_name = 'MLD_stn27_April-June-July.png'
fig.savefig(fig_name, dpi=300)

#### ------------- Stratification ---------------- ####
strat_monthly = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_stratif_monthly.pkl')
strat_monthly = strat_monthly[strat_monthly.index.year>=1950]
strat_feb = strat_monthly[strat_monthly.index.month==2]
strat_mar = strat_monthly[strat_monthly.index.month==3]
strat_apr = strat_monthly[strat_monthly.index.month==4]
strat_may = strat_monthly[strat_monthly.index.month==5]
strat_june = strat_monthly[strat_monthly.index.month==6]
strat_july = strat_monthly[strat_monthly.index.month==7]

# Save stratification
strat_feb.index = strat_feb.index.year
strat_mar.index = strat_mar.index.year
strat_apr.index = strat_apr.index.year
strat_may.index = strat_may.index.year
strat_june.index = strat_june.index.year
strat_july.index = strat_july.index.year
strat = pd.concat([strat_feb, strat_mar, strat_apr, strat_may, strat_june, strat_july], axis=1,
                  keys=['strat February','strat March','strat April','strat May', 'strat June','strat July'])
strat.to_csv('stn27_strat_Feb-July.csv', float_format='%.5f')

strat_std_anom = (strat - strat[(strat.index>=year_clim[0]) & (strat.index<=year_clim[1])].mean()) / strat[(strat.index>=year_clim[0]) & (strat.index<=year_clim[1])].std()
strat_std_anom.to_csv('stn27_strat_anom_Feb-July.csv', float_format='%.5f')

# plot stratification
plt.close('all')
fig = plt.figure(2)
ax = strat_june.rolling(5, min_periods=2).mean().plot(color='tab:blue')
strat_july.rolling(5, min_periods=2).mean().plot(color='tab:orange', ax=ax)
plt.legend(['June','July'])
plt.grid()
strat_june.plot(color='tab:blue', marker = '.', linestyle = '', ax=ax)
strat_july.plot(color='tab:orange', marker= '.', linestyle = '', ax=ax)
plt.title('5-50m stratification')
fig_name = 'strat_stn27_June-July.png'
fig.savefig(fig_name, dpi=300)

# plot stratification
plt.close('all')
fig = plt.figure(2)
ax = strat_apr.rolling(5, min_periods=2).mean().plot(color='tab:blue')
strat_june.rolling(5, min_periods=2).mean().plot(color='tab:orange')
strat_july.rolling(5, min_periods=2).mean().plot(color='tab:red', ax=ax)
plt.legend(['April','June','July'])
strat_apr.plot(color='tab:blue', marker = '.', linestyle = '', ax=ax)
strat_june.plot(color='tab:orange', marker = '.', linestyle = '', ax=ax)
strat_july.plot(color='tab:red', marker= '.', linestyle = '', ax=ax)
plt.title('5-50m stratification')
fig_name = 'strat_stn27_April-June-July.png'
plt.ylabel(r'$\rm \frac{\partial \rho}{\partial z}~[kg\,m^{-4}]$')
plt.grid()
fig.savefig(fig_name, dpi=300)

#### ------------- Check robustness of data ---------------- ####
mld_raw = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_MLD_raw.pkl')
strat_raw = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_stratif_raw.pkl')
rho_raw = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_rho_raw.pkl')
N2_raw = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_N2_raw.pkl')

stn27_occupation = mld_raw.resample('M').count()
stn27_occupation_june = stn27_occupation[stn27_occupation.index.month==6]
stn27_occupation_june.index = stn27_occupation_june.index.year
stn27_occupation_july = stn27_occupation[stn27_occupation.index.month==7]
stn27_occupation_july.index = stn27_occupation_july.index.year
stn27_occu = pd.concat([stn27_occupation_june, stn27_occupation_july], axis=1, keys=['June', 'July'])

# plot occupation
plt.close('all')
fig = plt.figure(2)
stn27_occu.plot(kind= 'bar')
plt.title('Station27 occupations')
fig_name = 'occu_stn27_June-July.png'
fig.savefig(fig_name, dpi=300)

#Some checks
mld_raw.index[(mld_raw.index.month==6) & (mld_raw.index.year==2014)]    
mld_raw.index[(mld_raw.index.month==6) & (mld_raw.index.year==2012)]    
# to plot an example of double N2 peak:
#N2_raw[(N2_raw.index.month==6) & (N2_raw.index.year==2012)].T.plot()  

#### ------------- CIL ---------------- ####
cil_summer = pd.read_pickle('/home/cyrf0006/AZMP/state_reports/stn27/S27_CIL_summer_stats.pkl')
cil_core = cil_summer['CIL core T']

plt.close('all')
fig = plt.figure(1)
ax = cil_core.plot()
cil_core.rolling(5, min_periods=2).mean().plot(ax=ax)
plt.legend(['annual','5-yr rolling'])
plt.title('CIL core Temperature')
fig_name = 'stn27_CIL_core.png'
fig.savefig(fig_name, dpi=300)

# Save CIL core T
cil_core.index = cil_core.index.year
cil_core.to_csv('stn27_cil_core_summer.csv', float_format='%.2f')

