'''
AZFP processing 
1. Convert *.01A files to *.nc
2. Convert *.nc to *Sv_clean.nc
3. Compute *MVBS.nc

see /home/cyrf0006/research/AZFP

file ex:
/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-01/55139/17071115.01A

'''
import numpy as np
import glob
import echopype as ep
from echopype.model import EchoData
import os

paths = [
'/home/cyrf0006/data_orwell/S27_AZFP/STN27_AZFP_01/55139',
'/home/cyrf0006/data_orwell/S27_AZFP/STN27_AZFP_01/55140',
'/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-02/55139',
'/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-02/55140',
'/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-03/55139',
'/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-03/55140',
]

paths = [
'/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-02/55139',
'/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-02/55140'
]

paths = [
'/home/cyrf0006/data_orwell/S27_AZFP/STN27-AZFP-03/55140'
]



def azfp_convert(path):

    # convert the data file-by-file
    xml_path = glob.glob(path + '/*.XML')[0]
    filenames = glob.glob(path + '/*.01A')
    for filename in filenames:
        if os.stat(filename).st_size !=0:
            print('convert ' + filename)
            data_tmp = ep.convert.ConvertAZFP(filename, xml_path)
            data_tmp.raw2nc()

def azfp_calibrate(path):
        
    # Calibrate the data
    nc_filenames = glob.glob(path + '/*.nc')
    for filename in nc_filenames:
        print('calibrate ' + filename)
        data = EchoData(filename)
        data.calibrate()

        # Now we pass coeff for T=5, S=32, D=80 (should be improve in Echopype)
        if path.split('/')[-1] == '55139': 
            abs_coeff = data.calc_range().frequency*0+np.array([.009778, .019828, .030685, .042934])
        elif path.split('/')[-1] == '55140':
            abs_coeff = data.calc_range().frequency*0+np.array([.042934, .108859, .256768])

        
        data.ABS = 2*abs_coeff*data.calc_range()
        data.TVG = 20*np.log10(data.calc_range())

        data.remove_noise(noise_est_range_bin_size=5, noise_est_ping_size=20, save=True)  
        data.get_MVBS(source='Sv_clean',  MVBS_range_bin_size=5, MVBS_ping_size=12, save=True)  

        
## --- Main processing --- ##

for path in paths:
    azfp_convert(path)
    azfp_calibrate(path)
    
