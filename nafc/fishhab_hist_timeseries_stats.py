'''
Some stats on fishhab timeseries
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime



df_coord_spring = pd.read_pickle('spring_coords.pkl')
df_coord_fall = pd.read_pickle('fall_coords.pkl')

df_coord_spring = df_coord_spring[df_coord_spring.index.year>=1978]
df_coord_fall = df_coord_fall[df_coord_fall.index.year>=1978]
