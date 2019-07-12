'''
Script written to process C. Morris temperature data at Gilber Bay 

Frederic.Cyr@dfo-mpo.gc.ca

'''

import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
import os



## ----  Prepare the data ---- ##
# load from Excel sheets
xls = pd.ExcelFile('/home/cyrf0006/research/GilbertBay/Gilbert_Bay_Temperatures.xlsx')
df_GullIs = pd.read_excel(xls, 'Gull Island', parse_dates={'datetime': [1, 2]})
df_Ins1 = pd.read_excel(xls, 'Inside 1 Bottom 353231', parse_dates={'datetime': [1, 2]})
df_Ins2 = pd.read_excel(xls, 'Inside 2 353232', parse_dates={'datetime': [1, 2]})
df_Ins3 = pd.read_excel(xls, 'Inside 3 353224', parse_dates={'datetime': [1, 2]})
df_Ins4 = pd.read_excel(xls, 'Inside 4 353227', parse_dates={'datetime': [1, 2]})
df_FoxCo = pd.read_excel(xls, 'Fox Cove Line', parse_dates={'datetime': [1, 2]})
df_FoxCoW = pd.read_excel(xls, 'Fox Cove Data from Wade B', parse_dates={'datetime': [1, 2]})

# set index to datetime
df_GullIs.set_index(df_GullIs.datetime, inplace=True)
df_Ins1.set_index(df_Ins1.datetime, inplace=True)
df_Ins2.set_index(df_Ins2.datetime, inplace=True)
df_Ins3.set_index(df_Ins3.datetime, inplace=True)
df_Ins4.set_index(df_Ins4.datetime, inplace=True)
df_FoxCo.set_index(df_FoxCo.datetime, inplace=True)
df_FoxCoW.set_index(df_FoxCoW.datetime, inplace=True)

# Clean the dataFrames
df_GullIs.drop(columns={'Position', 'datetime'}, inplace=True)
df_Ins1.drop(columns={'Position', 'datetime'}, inplace=True)
df_Ins2.drop(columns={'Position', 'datetime'}, inplace=True)
df_Ins3.drop(columns={'Position', 'datetime'}, inplace=True)
df_Ins4.drop(columns={'Position', 'datetime'}, inplace=True)
df_FoxCo.drop(columns={'Position', 'datetime'}, inplace=True)
df_FoxCoW.drop(columns={'Position', 'datetime'}, inplace=True)

df_GullIs.rename(columns={df_GullIs.columns[1]: "Temperature" }, inplace=True)
df_Ins1.rename(columns={df_Ins1.columns[1]: "Temperature" }, inplace=True)
df_Ins2.rename(columns={df_Ins2.columns[1]: "Temperature" }, inplace=True)
df_Ins3.rename(columns={df_Ins3.columns[1]: "Temperature" }, inplace=True)
df_Ins4.rename(columns={df_Ins4.columns[1]: "Temperature" }, inplace=True)
df_FoxCo.rename(columns={df_FoxCo.columns[1]: "Temperature" }, inplace=True)
df_FoxCoW.rename(columns={df_FoxCoW.columns[1]: "Temperature" }, inplace=True)
df_FoxCoW.rename(columns={'Depth (m)': "Depth" }, inplace=True)



## ---- Build DataFrames per station ---- ##
# 1. Gull Island
df = df_GullIs.pivot(columns='Depth', values='Temperature')
df.to_pickle('GullIsland.pkl')
# plot
df = df.resample('D').mean()
df.plot()
plt.ylabel(r'$\rm T(^{\circ}C)$')
plt.title('Gull Island')
plt.show()

# 2. Inside
df =  pd.concat([df_Ins4.Temperature, df_Ins3.Temperature, df_Ins2.Temperature, df_Ins1.Temperature], keys = ['10m', '15m', '20m', '25m'], axis=1)
df.to_pickle('Inside.pkl')
# plot
df = df.resample('D').mean()
df.plot()
plt.ylabel(r'$\rm T(^{\circ}C)$')
plt.title('Inside')
plt.show()
# plot 2
plt.contourf(df.index, np.array([10,15,20,25]), df.T, levels=np.arange(-2,12,1), extend='both')
plt.gca().invert_yaxis()
plt.ylabel('Depth (m)')
plt.title('Inside')
c=plt.colorbar()
c.set_label(r'$\rm T(^{\circ}C)$')
plt.show()

# 3 Fox Cove - 2010's
df = df_FoxCo.pivot(columns='Depth', values='Temperature')
df.to_pickle('FoxCo.pkl')
# plot
df = df.resample('D').mean()
plt.contourf(df.index, np.array([0,10,20,30,40,50,60,70]), df.T)
plt.gca().invert_yaxis()
plt.ylabel('Depth (m)')
plt.title('Fox Cove')
c = plt.colorbar()
c.set_label(r'$\rm T(^{\circ}C)$')
plt.show()

# 4 Fox Cove - 1990's
df = df_FoxCoW.Temperature
df.to_pickle('FoxCoW.pkl')
df = df.resample('D').mean()
df.plot()
plt.title('Fox Cove')
plt.ylabel(r'$\rm T(^{\circ}C)$')
plt.show()

