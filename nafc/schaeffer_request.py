import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import from_levels_and_colors
import numpy as np
import h5py
import cmocean
import cmocean.cm as cmo
import cartopy. crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cc_variable_list2 as vl

# Load data
df = pd.read_csv('/home/cyrf0006/github/AZMP-NL/datasets/carbonates/AZMP_carbon_data.csv', delimiter=',') 

# NL only
df = df[df.Region=='NL']

# Select bottom only (see AZMP_zonal_cc)
df = df.loc[df.groupby('Station_Name')['Depth_(dbar)'].idxmax()]
df = df.loc[df['Depth_(dbar)'] >10]

# select variables
df = df[['Timestamp', 'Trip_Name', 'Station_Name', 'Latitude_(degNorth)', 'Longitude_(degEast)', 'Depth_(dbar)', 'Temperature_(degC)', 'Salinity_(psu)', 'Dissolved_Oxygen_(mL/L)', 'pH_tot', 'Omega_Aragonite_(--)']]

df.set_index('Timestamp', inplace=True)

df.to_csv('schaefer_data_prelim.csv', float_format='%.4f', index=True)
