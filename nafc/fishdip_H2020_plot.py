import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd
import gsw


files  = [
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2020/20200204/DAT_001.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2020/20200204/DAT_002.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2020/20200204/DAT_003.csv']
       
files  = [
'/media/cyrf0006/Seagate Backup Plus Drive/VMP_dataprocessing/FISHDIP2020/20200206/DAT_008.csv',
'/media/cyrf0006/Seagate Backup Plus Drive/VMP_dataprocessing/FISHDIP2020/20200206/DAT_009.csv',
'/media/cyrf0006/Seagate Backup Plus Drive/VMP_dataprocessing/FISHDIP2020/20200206/DAT_010.csv']
       
p_bin = np.linspace(1,90,90)
df_temp = pd.DataFrame(columns = p_bin)
df_sal = pd.DataFrame(columns = p_bin)
df_sig0 = pd.DataFrame(columns = p_bin)
df_w = pd.DataFrame(columns = p_bin)

for iidex, i in enumerate(files):
    if iidex == 0:
        df = pd.read_csv(i)
    else:
        df_tmp = pd.read_csv(i)
        df  = df.append(df_tmp)



        plt.pcolorm




        
        
    Ibtm = np.argmax(df.pressure.values)
    digitized = np.digitize(df.pressure[0:Ibtm], p_bin) 
    temp = np.array([df.conserv_temp.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    sal = np.array([df.abs_salinity.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    sig0 = np.array([df.sigma_0.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    w = np.array([df.vert_vel.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    
    df_temp.iloc[iidex] = temp    
    df_sal.iloc[iidex] = sal   
    df_sig0.iloc[iidex] = sig0    
    df_w.iloc[iidex] = w    

df_temp.dropna(how='all', axis=1, inplace=True)
df_sal.dropna(how='all', axis=1, inplace=True)
df_sig0.dropna(how='all', axis=1, inplace=True)
df_sal.dropna(how='all', axis=1, inplace=True)

df_temp.drop(index=['7', '12', '15'], inplace=True)
df_sal.drop(index=['7', '12', '15'], inplace=True)
df_sig0.drop(index=['7', '12', '15'], inplace=True)
df_w.drop(index=['7', '12', '15'], inplace=True)
  
plt.contourf(df_temp.index, df_temp.columns, df_temp.T.values)
#plt.contour(df_sig0.index, df_sig0.columns, df_sig0.T.values, colors='w')
plt.gca().invert_yaxis()
plt.colorbar()    

plt.figure()
plt.contourf(df_sal.index, df_sal.columns, df_sal.T.values)
#plt.contour(df_sig0.index, df_sig0.columns, df_sig0.T.values, colors='w')
plt.gca().invert_yaxis()
plt.colorbar()    


df_temp.to_pickle('temp_W2019.pkl')
df_sal.to_pickle('sal_W2019.pkl')


sal1 = df_sal.values.reshape(1,df_sal.size)
temp1 = df_temp.values.reshape(1,df_temp.size)
temp1 = temp1[sal1>0]
sal1 = sal1[sal1>0]
plt.scatter(sal1, temp1, c=np.repeat(2019, sal1.size)) 

df = pd.DataFrame[sal1, ]
