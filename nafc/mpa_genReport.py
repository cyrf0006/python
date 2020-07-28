# Preliminary attempt to build an "all inclusive" function
# that would generate all figures for RVsurvey CSAS

import os
import matplotlib.pyplot as plt
import numpy as np
import azmp_sections_tools as azst
import azmp_report_tools as azrt  
import mpa_tools as mpa


# 5. bottom temperature maps
mpa.bottom_temperature(season='fall', year='2019', closure_scenario='all') 
mpa.bottom_temperature(season='spring', year='2019', closure_scenario='all') 

# bottom stats and scorecards HERE!!!!!!
mpa.bottom_stats(years=np.arange(1980, 2020), season='spring', closure_scenario='B')
mpa.bottom_stats(years=np.arange(1980, 2020), season='fall', closure_scenario='A')
mpa.bottom_stats(years=np.arange(1980, 2020), season='fall', closure_scenario='C')
#azrt.bottom_stats(years=np.arange(1980, 2020), season='fall')
mpa.bottom_scorecards(years=[1990, 2019])

os.system('cp scorecards_botT_spring.png scorecards_botT_spring_FR.png scorecards_botT_fall_FR.png scorecards_botT_fall.png ../2019')
# For NAFO STACFEN and STACFIS input (for azmp_composite_index.py):
azrt.bottom_stats(years=np.arange(1980, 2020), season='summer', climato_file='Tbot_climato_SA4_summer_0.10.h5')

# bottom temperature bar plots
%my_run azmp_bottomT_mean_anomaly.py # same as previous
os.system('cp mean_anomalies_fall.png mean_anomalies_spring.png ../2019')
os.system('cp mean_anomalies_fall_FR.png mean_anomalies_spring_FR.png ../2019')
