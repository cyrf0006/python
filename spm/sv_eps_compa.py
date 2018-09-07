'''
Comparison between epsilon and AZMP backscatter data

'''

import csv
import numpy as np
import datetime as dt
import pandas as pd
from scipy.io import loadmat

azfp_file1 = '/home/cyrf0006/research/SPM/Max_stuff/sv_125kHz_06_09_2017.csv'
azfp_file2 = '/home/cyrf0006/research/SPM/Max_stuff/sv_125kHz_07_09_2017.csv'


#### ---- Read backscatter data ---- ####
# Day 1
sv_time = []
sv_depth = []
sv_data = []
with open(azfp_file1, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
       
        sv_depth.append(row[1])
        sv_time.append('2017-09-06'+row[2])
        sv_data.append(row[3])

# remove 1st row
sv_depth = sv_depth[1:-1]
sv_time = sv_time[1:-1]
sv_date = sv_data[1:-1]
        
# Day 2
sv2_time = []
sv2_depth = []
sv2_data = []
with open(azfp_file2, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
       
        sv2_depth.append(row[1])
        sv2_time.append('2017-09-07'+row[2])
        sv2_data.append(row[3])   
        
# remove 1st row
sv2_depth = sv2_depth[1:-1]
sv2_time = sv2_time[1:-1]
sv2_date = sv2_data[1:-1]

# Append 2 days
sv_depth.append(sv2_depth)
sv_time.append(sv2_time)
sv_data.append(sv2_date)

# Convert to Pandas time
sv_pdtime = pd.Series([pd.to_datetime(date) for date in sv_time])



#### ---- Read VMP data  ---- ####
filelist = np.genfromtxt('vmp_azfp.list', dtype=str)

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

# Empty lists
EPSlist = []
N2list = []
Fluolist = []
datelist = []

# Binned depth vector
Pbin = np.arange(1, 150, 2)

# Loop on file    
for fname in filelist:
    vmp_dict = loadmat(fname,squeeze_me=True, struct_as_record=False)

    date_list.append(pd.matlab2datetime(vmp_dict['mtime_eps'][0]))
    #HERE!!!!!!

    
    LATlist.append(profile.attributes['LATITUDE'])
    LONlist.append(profile.attributes['LONGITUDE'])

    # Must get profile, remove upcast + 5-m bin average
    P = np.array(profile['PRES'])
    T = np.array(profile['TEMP'])
    S = np.array(profile['PSAL'])
    C = np.array(profile['CNDC'])
    SIG = np.array(profile['sigma_t'])
    F = np.array(profile['flECO-AFL'])
    O2 = np.array(profile['oxigen_ml_L'])
    PH= np.array(profile['ph'])
   
    Ibtm = np.argmax(P)    
    digitized = np.digitize(P[0:Ibtm], Pbin) #<- this is awesome!
    
    Tlist.append([T[0:Ibtm][digitized == i].mean() for i in range(0, len(Pbin))])
    Slist.append([S[0:Ibtm][digitized == i].mean() for i in range(0, len(Pbin))])    

    
