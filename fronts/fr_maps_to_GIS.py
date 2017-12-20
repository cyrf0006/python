# Initial test to transfer Matlab's front probability maps into GIS shapefiles
# Frederic.Cyr@dfo-mpo.gc.ca - November 2017

#infile = '/media/cyrf0006/Seagate Backup Plus Drive/cyrf0006_2014-03-26/IML/fronts/matlab_workspace/probability/OUTPUT/SST_atlantic_prob.mat'
infile = '../SST_atlantic_prob.mat'
outfile = 'prob_atlantic_1985-2010.asc'

## import os
## import matplotlib
## import matplotlib.pyplot as plt
## from matplotlib import dates
import numpy as np
## import numpy.ma as ma
from scipy.io import loadmat
## import datetime
## import pandas as pd
## #import seawater as sw
## import gsw

# Load matlab file
front_dict = loadmat(infile,squeeze_me=True, struct_as_record=False)
print front_dict


prob = front_dict['probability']
lon = front_dict['lon']
lat = front_dict['lat']

# reshape in column vector
prob = np.reshape(prob, (prob.size,1))
lon = np.reshape(lon, (lon.size,1))
lat = np.reshape(lat, (lon.size,1))

# full data
data = np.concatenate((lon, lat, prob), axis=1)

# Save data
np.savetxtoutfile, np.c_[lon,lat,prob], fmt='%.4f')
