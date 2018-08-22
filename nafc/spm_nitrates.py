# A first test to read Excel nutrient file and export to Pandas.

# Check in:
#  /home/cyrf0006/research/SPM/python_processing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from seawater import extras as swx

# Adjust fontsize/weight
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)

## ----  Load data from Excel sheet ---- ##
# Read in multiindex
df = pd.read_excel('/home/cyrf0006/research/SPM/JauzeinC/SPM_chemical.xls', header=1, index_col=[0,1])
df.dropna(how='all', inplace=True)

# reset index
df = df.reset_index(level=[0,1])

# add a color column
color = ['b']*df.shape[0]
for idx, c in enumerate(df.Station):
    if c=='B':
        color[idx]='r'
    elif c=='C':
        color[idx]='g'
    elif c=='D':
        color[idx]='y'       
df['color'] = color


##########  good until here ---------####






import bokeh.plotting as bpl
import bokeh.models as bmo
from bokeh.palettes import d3
#from bokeh.charts import Scatter
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource

# create figure and plot
p = bpl.figure()
Scatter(df, x='nitrate', y='depth', color='color', title="Nitrate concentrations",
            xlabel="Nitrate Concentrations [umol/L]", ylabel="Depth (ml)", legend='Station')
bpl.output_file('nitrate_chemical.html')
bpl.save(p)

keyboard


p = bpl.figure()
source = ColumnDataSource(df)
TOOLS="pan,wheel_zoom,box_zoom,reset,hover"
p = bpl.figure(title="test", tools=TOOLS)
p.circle('nitrate', 'depth', radius=.66, source=source,
          fill_color=color, fill_alpha=0.6, line_color=None)

hover = p.select(dict(type=HoverTool))[0]
hover.tooltips = [
    ("material", "@material")
]

bpl.save(p)

#HERE!!!

# create figure and plot
p = bpl.figure()
source = ColumnDataSource(df)
p.scatter(x='nitrate', y='depth', color='color', source=source, title="Nitrate concentrations", xlabel="Nitrate Concentrations [umol/L]", ylabel="Depth (ml)", legend='Station')
hover = p.select(dict(type=HoverTool))[0]
bpl.output_file('nitrate_chemical.html')
bpl.save(p)

from bokeh.plotting import figure, output_notebook, show
from bokeh.models import HoverTool, BoxSelectTool, BoxZoomTool, PanTool, WheelZoomTool, ResetTool#For enabling tools
TOOLS = [BoxSelectTool(), HoverTool(), BoxZoomTool(), PanTool(), WheelZoomTool(), ResetTool()]
p = figure(title="NAFO interactive plot", plot_width=1200, plot_height=1200, tools=TOOLS)
p = bpl.figure()
p.circle(x='nitrate', y='depth', color='color', source=source)
bpl.output_file('nitrate_chemical.html')
bpl.save(p)

