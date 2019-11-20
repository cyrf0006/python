import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import seawater
import pyglider as pg
import gsw
import cmocean as cmo
import seaborn as sns
import os

df = pd.read_pickle('preswot_all_column_data.pkl') 

# Select depth range
df_plot = df[(df.depth>=25) & (df.depth<=125)]

#sns.jointplot(x="chlorophyll", y="trp_like", data=df, kind="kde");


## ---- 0-150m / 20-60km ---- ##

df_plot = df[df.depth<=150]
df_plot = df_plot[(df_plot.distance>=20) & (df_plot.distance<=60)]
df_plot = df_plot[df_plot.transect==1]

# 1. vs chl
# chl-trp
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.chlorophyll, df_plot.trp_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.1, .16])
ax.set_xlim([-.1,  1.4])
ax.set_xlabel(r'Chl-a concentration ($\rm \mu g\,L^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_chl_trp_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# chl-phe
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.chlorophyll, df_plot.phe_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.03, .055])
ax.set_xlim([-0.02,  1.3])
ax.set_xlabel(r'Chl-a concentration ($\rm \mu g\,L^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak A/M (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_chl_phe_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# chl-flu
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.chlorophyll, df_plot.flu_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.055, .12])
ax.set_xlim([-0.02,  1.3])
ax.set_xlabel(r'Chl-a concentration ($\rm \mu g\,L^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak B (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_chl_flu_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# chl-pyr
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.chlorophyll, df_plot.pyr_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.15, .215])
ax.set_xlim([-0.02,  1.4])
ax.set_xlabel(r'Chl-a concentration ($\rm \mu g\,L^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak N (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_chl_pyr_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# chl-cdom
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.chlorophyll, df_plot.cdom, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.4, 1.4])
ax.set_xlim([-0.02,  1.4])
ax.set_xlabel(r'Chl-a concentration ($\rm \mu g\,L^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak C (FI, QSU)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_chl_cdom_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# 2. vs O2
# o2-trp
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.oxygen, df_plot.trp_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.1, .16])
ax.set_xlim([180,  250])
ax.set_xlabel(r'$\rm O_2$ concentration ($\rm \mu mol\,Kg^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_o2_trp_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# o2-phe
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.oxygen, df_plot.phe_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.03, .055])
ax.set_xlim([180,  250])
ax.set_xlabel(r'$\rm O_2$ concentration ($\rm \mu mol\,Kg^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak A/M (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_o2_phe_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# o2-flu
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.oxygen, df_plot.flu_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.055, .12])
ax.set_xlim([180,  250])
ax.set_xlabel(r'$\rm O_2$ concentration ($\rm \mu mol\,Kg^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak B (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_o2_flu_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# o2-pyr
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.oxygen, df_plot.pyr_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.15, .215])
ax.set_xlim([180,  250])
ax.set_xlabel(r'$\rm O_2$ concentration ($\rm \mu mol\,Kg^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak N (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_o2_pyr_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# o2-cdom
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.oxygen, df_plot.cdom, c=df_plot.depth)
cb = plt.colorbar()
ax.set_ylim([.4, 1.4])
ax.set_xlim([180,  250])
ax.set_xlabel(r'$\rm O_2$ concentration ($\rm \mu mol\,Kg^{-1}$)', fontWeight = 'bold')
ax.set_ylabel(r'peak C (FI, QSU)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_o2_cdom_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# 3. Fluorophores
# trp-phe
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.trp_like, df_plot.phe_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.1, .16])
ax.set_ylim([.03, .055])
ax.set_ylabel(r'peak A/M (FI, relative units)', fontWeight = 'bold')
ax.set_xlabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_trp_phe_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# trp-flu
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.trp_like, df_plot.flu_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.1, .16])
ax.set_ylim([.055, .12])
ax.set_ylabel(r'peak B (FI, relative units)', fontWeight = 'bold')
ax.set_xlabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_trp_flu_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# trp-pyr
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.trp_like, df_plot.pyr_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.1, .16])
ax.set_ylim([.15, .215])
ax.set_ylabel(r'peak N (FI, relative units)', fontWeight = 'bold')
ax.set_xlabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_trp_pyr_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# trp-cdom
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.trp_like, df_plot.cdom, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.1, .16])
ax.set_ylim([.4, 1.4])
ax.set_ylabel(r'peak C (FI, QSU)', fontWeight = 'bold')
ax.set_xlabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_trp_cdom_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# flu-pyr
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.flu_like, df_plot.pyr_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.055, .12])
ax.set_ylim([.15, .215])
ax.set_ylabel(r'peak B (FI, relative units)', fontWeight = 'bold')
ax.set_xlabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_flu_pyr_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# flu-phe
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.flu_like, df_plot.phe_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.055, .12])
ax.set_ylim([.03, .055])
ax.set_ylabel(r'peak A/M (FI, relative units)', fontWeight = 'bold')
ax.set_xlabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_flu_phe_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# flu-cdom
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.flu_like, df_plot.cdom, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.055, .12])
ax.set_ylim([.4, 1.4])
ax.set_ylabel(r'peak C (FI, QSU)', fontWeight = 'bold')
ax.set_xlabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_flu_cdom_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# pyr-phe
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.pyr_like, df_plot.phe_like, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.15, .215])
ax.set_ylim([.03, .055])
ax.set_ylabel(r'peak A/M (FI, relative units)', fontWeight = 'bold')
ax.set_xlabel(r'peak N (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_pyr_phe_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)

# pyr-cdom
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.scatter(df_plot.pyr_like, df_plot.cdom, c=df_plot.depth)
cb = plt.colorbar()
ax.set_xlim([.15, .215])
ax.set_ylim([.4, 1.4])
ax.set_ylabel(r'peak C (FI, QSU)', fontWeight = 'bold')
ax.set_xlabel(r'peak T (FI, relative units)', fontWeight = 'bold')
cb.set_label('Depth (m)')
figname = 'scatter_pyr_cdom_depth.png'
fig.savefig(figname, dpi=150)
os.system('convert -trim ' + figname + ' ' + figname)



os.system('montage scatter_chl_trp_depth.png scatter_chl_phe_depth.png scatter_chl_flu_depth.png scatter_chl_pyr_depth.png scatter_chl_cdom_depth.png -tile 2x3 -geometry +10+10  -background white  FDOM_vs_chl_0-150m.png')
os.system('montage scatter_o2_trp_depth.png scatter_o2_phe_depth.png scatter_o2_flu_depth.png scatter_o2_pyr_depth.png scatter_o2_cdom_depth.png -tile 2x3 -geometry +10+10  -background white  FDOM_vs_o2_0-150m.png')
os.system('montage scatter_trp_phe_depth.png scatter_pyr_phe_depth.png scatter_flu_phe_depth.png scatter_trp_flu_depth.png scatter_pyr_cdom_depth.png scatter_flu_cdom_depth.png scatter_trp_pyr_depth.png scatter_flu_pyr_depth.png scatter_trp_cdom_depth.png -tile 3x3 -geometry +10+10 -background white  FDOM_vs_FDOM_0-150m.png')
