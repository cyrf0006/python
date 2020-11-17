# Preliminary attempt to build an "all inclusive" function
# that would generate all figures for RVsurvey CSAS

import os
import matplotlib.pyplot as plt
import numpy as np
import azmp_sections_tools as azst
import azmp_report_tools as azrt  
import mpa_tools as mpa


# Bottom temperature maps
#mpa.bottom_temperature(season='fall', year='2019', closure_scenario='4') 
#mpa.bottom_temperature(season='spring', year='2019', closure_scenario='all') 

# bottom stats
mpa.bottom_stats(years=np.arange(1980, 2020), season='spring', closure_scenario='reference')
mpa.bottom_stats(years=np.arange(1980, 2020), season='spring', closure_scenario='1')
mpa.bottom_stats(years=np.arange(1980, 2020), season='spring', closure_scenario='2')
mpa.bottom_stats(years=np.arange(1980, 2020), season='spring', closure_scenario='3')
mpa.bottom_stats(years=np.arange(1980, 2020), season='spring', closure_scenario='4')
mpa.bottom_stats(years=np.arange(1980, 2020), season='fall', closure_scenario='reference')
mpa.bottom_stats(years=np.arange(1980, 2020), season='fall', closure_scenario='1')
mpa.bottom_stats(years=np.arange(1980, 2020), season='fall', closure_scenario='2')
mpa.bottom_stats(years=np.arange(1980, 2020), season='fall', closure_scenario='3')
mpa.bottom_stats(years=np.arange(1980, 2020), season='fall', closure_scenario='4')

# Scorecards
mpa.bottom_scorecards(years=[1990, 2019])


