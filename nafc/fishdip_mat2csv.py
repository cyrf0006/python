'''
To convert to csv.
Run in /home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019
'''

import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd
import gsw

# FishDip 2019
files = ['001', '002', '003', '004', '006', '007', '010', '012', '015']

# FishDip 2020
files = ['001', '002', '003']


# for myself:
for i in files:
    infile = '/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_' + i + '.mat'
    outfile = 'DAT_' + i + '.csv'
    print(infile)

    mat = loadmat(infile)  # load mat-file
    print(mat['date'])
    
    P = mat['P_slow']
    JAC_C = mat['JAC_C']
    JAC_T = mat['JAC_T']
    W = mat['W_slow']
    Turb = mat['Turbidity']
    Chla = mat['Chlorophyll']
    lat = 54
    lon = -60

    SP = gsw.SP_from_C(JAC_C, JAC_T, P)
    SA = gsw.SA_from_SP(SP, P, lon, lat)
    CT = gsw.CT_from_pt(SA, JAC_T)
    sig0 = gsw.sigma0(SA,CT);

    # Bring Chla / Turb to resolution of CTD
    bin_size = len(Turb) / len(P)
    Turb = Turb.reshape(-1, int(bin_size)).mean(axis=1)
    Chla = Chla.reshape(-1, int(bin_size)).mean(axis=1)
    Turb = Turb.reshape(len(Turb),1)
    Chla = Chla.reshape(len(Chla),1)
    
    df = pd.DataFrame(np.concatenate((P, CT, SP, sig0, W, Turb, Chla), axis=1), columns=['pressure', 'conserv_temp', 'abs_salinity', 'sigma_0', 'vert_vel', 'Turbidity', 'Chlorophyll'])

    # reduce according to vertical velocity
    df = df[df.vert_vel>.45]
    df.to_csv(outfile, float_format='%.2f', index=False)


# For T90 and PSU
for i in files:
    infile = '/home/cyrf0006/research/VMP_dataprocessing/FISHDIP2019/DAT_' + i + '.mat'
    outfile = 'H2019_CTD_' + i + '.csv'
    print(infile)

    mat = loadmat(infile)  # load mat-file

    P = mat['P_slow']
    JAC_C = mat['JAC_C']
    JAC_T = mat['JAC_T']
    W = mat['W_slow']
    Turb = mat['Turbidity']
    Chla = mat['Chlorophyll']
    lat = 54
    lon = -60

    SP = gsw.SP_from_C(JAC_C, JAC_T, P)
    SA = gsw.SA_from_SP(SP, P, lon, lat)
    CT = gsw.CT_from_pt(SA, JAC_T)
    sig0 = gsw.sigma0(SA,CT);
    Z = gsw.z_from_p(P,lat)
    SS = gsw.sound_speed(SA, CT, P)

    # Bring Chla / Turb to resolution of CTD
    bin_size = len(Turb) / len(P)
    Turb = Turb.reshape(-1, int(bin_size)).mean(axis=1)
    Chla = Chla.reshape(-1, int(bin_size)).mean(axis=1)
    Turb = Turb.reshape(len(Turb),1)
    Chla = Chla.reshape(len(Chla),1)

    df = pd.DataFrame(np.concatenate((JAC_C, JAC_T, P, Z, SP, SS, sig0, W, Turb, Chla), axis=1), columns=['Conductivity', 'Temperature', 'Sea pressure', 'Depth', 'salinity', 'Speed of sound', 'Density anomaly', 'vert_vel', 'Turbidity', 'Chlorophyll'])

    # reduce according to vertical velocity
    #df = df[df.vert_vel>.45]
    df.to_csv(outfile, float_format='%.2f', index=False, sep='\t')
