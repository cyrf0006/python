import dolfyn as dol
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta


adcp = dol.read('FD01_000.000')

v = np.linspace(-.1,.3, 11)

U = adcp.vel[1,:,:]
V = adcp.vel[2,:,:]
df_U = pd.DataFrame(U.T)
df_V = pd.DataFrame(V.T)
dtime = [datetime.fromordinal(int(x)) + timedelta(days=x%1) - timedelta(days = 366) for x in adcp.mpltime]
z = adcp.range
df_U.index = pd.to_datetime(dtime)
df_U.columns = z
df_V.index = pd.to_datetime(dtime)
df_V.columns = z

# select time
df_U = df_U[(df_U.index > pd.to_datetime('2018-02-04 17:37:00')) & (df_U.index < pd.to_datetime('2018-02-04 20:00:00'))]
df_V = df_V[(df_V.index > pd.to_datetime('2018-02-04 17:37:00')) & (df_V.index < pd.to_datetime('2018-02-04 20:00:00'))]

# select depth range
df_U = df_U.loc[:,df_U.columns<15]
df_V = df_V.loc[:,df_V.columns<15]

# resample
df_U = df_U.resample('60s').mean()
df_V = df_V.resample('60s').mean()



plt.plot(df_U.mean(axis=0).rolling(4).mean(), df_U.columns)
plt.plot(df_V.mean(axis=0).rolling(4).mean(), df_V.columns)
plt.gca().invert_yaxis()
plt.legend(['U', 'V'])
plt.ylabel('Z (m)')
plt.xlabel('(m/s)')
plt.grid()
plt.show()





plt.contourf(df_U.index, df_U.columns, df_U.T, v, extend='both')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

