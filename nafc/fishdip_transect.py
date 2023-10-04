import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files  = [
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_001.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_002.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_003.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_004.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_006.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_007.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_010.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_012.csv',
'/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_015.csv']

           
stations = ['1', '2', '3', '4', '6', '7', '10', '12', '15']
    
## df01 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_001.csv')
## df02 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_002.csv')
## df03 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_003.csv')
## df04 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_004.csv')
## df06 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_006.csv')
## df07 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_007.csv')
## df10 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_010.csv')
## df12 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_012.csv')
## df15 = pd.read_csv('/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_015.csv')


p_bin = np.linspace(1,200, 200)
df_temp = pd.DataFrame(index = stations, columns = p_bin)
df_sal = pd.DataFrame(index = stations, columns = p_bin)
df_sig0 = pd.DataFrame(index = stations, columns = p_bin)
df_w = pd.DataFrame(index = stations, columns = p_bin)
df_turb = pd.DataFrame(index = stations, columns = p_bin)
df_chla = pd.DataFrame(index = stations, columns = p_bin)

for iidex, i in enumerate(files):
    df = pd.read_csv(i)

    # keep last downcast
    dp = np.diff(df.pressure)
    idx = np.argwhere(dp<0)[-1]
    df = df[df.index.values>idx]
        
    Ibtm = np.argmax(df.pressure.values)
    digitized = np.digitize(df.pressure[0:Ibtm], p_bin) 
    temp = np.array([df.conserv_temp.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    sal = np.array([df.abs_salinity.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    sig0 = np.array([df.sigma_0.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    w = np.array([df.vert_vel.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    turb = np.array([df.Turbidity.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])
    chla = np.array([df.Chlorophyll.values[0:Ibtm][digitized == i].mean() for i in range(0, len(p_bin))])

    df_temp.iloc[iidex] = temp    
    df_sal.iloc[iidex] = sal   
    df_sig0.iloc[iidex] = sig0    
    df_w.iloc[iidex] = w    
    df_turb.iloc[iidex] = turb    
    df_chla.iloc[iidex] = chla    

df_temp.dropna(how='all', axis=1, inplace=True)
df_sal.dropna(how='all', axis=1, inplace=True)
df_sig0.dropna(how='all', axis=1, inplace=True)
df_sal.dropna(how='all', axis=1, inplace=True)
df_turb.dropna(how='all', axis=1, inplace=True)
df_chla.dropna(how='all', axis=1, inplace=True)

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

plt.contourf(df_turb.index, df_turb.columns, df_turb.T.values)
#plt.contour(df_sig0.index, df_sig0.columns, df_sig0.T.values, colors='w')
plt.gca().invert_yaxis()
plt.title('Turbidity (FTU)')
plt.ylabel('Depth (m)')
plt.xlabel('Station no.')
plt.colorbar()    



df_temp.to_pickle('temp_W2019.pkl')
df_sal.to_pickle('sal_W2019.pkl')


sal1 = df_sal.values.reshape(1,df_sal.size)
temp1 = df_temp.values.reshape(1,df_temp.size)
temp1 = temp1[sal1>0]
sal1 = sal1[sal1>0]
plt.scatter(sal1, temp1, c=np.repeat(2019, sal1.size)) 

df = pd.DataFrame[sal1, ]
