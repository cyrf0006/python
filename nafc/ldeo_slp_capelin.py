# data source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=71800&vid=676

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
#import datetime
os.environ['PROJ_LIB'] = '/home/cyrf0006/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap


# Some parameters
## years = [1948, 1971]
## years = [1971, 1976]
## ### years = [1971, 1981]
## #years = [1976, 1981]
## years = [1981, 1995]
## years = [1981, 1998]
## years = [1995, 2013]
## years = [2013, 2020]
## years = [2013, 2017]
## #years = [2017, 2021]

# based on NLCI:
years_list = [
[1948, 1971],
[1971, 1976],
[1976, 1982],
[1982, 1998],
[1998, 2014],
[2014, 2017],
[2017, 2021]
]

## # based on Capelin Spring:
## years_list = [
## [1975, 1982],
## [1982, 1990],
## [1990, 1995],
## [1995, 2012],
## [2012, 2017],
## [2017, 2019]
## ]
    

months = [1, 12] # months to keep

# For map limits
lllon = -100.
urlon = 10.
lllat = 0.
urlat = 90.

# For study area
lat1 = 65
lat2 = 47
lon1 =  -47
lon2 = -65
lon1 =  360-40.986870
lon2 = 360-60.704367


#v = np.arange(990, 1030) # SLP values
v = np.arange(995, 1025) # SLP values

# Load SLP data from NOAA ESRL
#ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mnmean.nc')
ds = xr.open_dataset('/home/cyrf0006/data/NOAA_ESRL/slp.mon.mean.nc')
#ds = xr.open_mfdataset('/home/cyrf0006/data/NOAA_ESRL/slp.201*.nc')

# Selection of a subset region
#ds = ds.where((ds.lon>=-120) & (ds.lon<=30), drop=True) # original one
#ds = ds.where((ds.lat>=0) & (ds.lat<=90), drop=True)

da = ds['slp']
#p = da.to_pandas() # deprecated
p = da.to_dataframe() # deprecated
# Restrict to 2021
p = p[(p.index.get_level_values('time').year<=2021)]

# Compute climatology
#p_clim = p[(p.items.month>=months[0]) & (p.items.year<=months[1])]
#df_clim = p_clim.mean(axis=0)
#df_clim = p.mean(axis=0)
# Compute climatology
df_clim = p.groupby(level=[1,2]).mean().unstack()


# Load NCAM data
#df_ncam = pd.read_csv('/home/cyrf0006/data/NCAM/multispic_estimates.csv', index_col='year')
## df_cod = df_ncam[df_ncam.species=='Cod']
## df_had = df_ncam[df_ncam.species=='Haddock']
## df_hak = df_ncam[df_ncam.species=='Hake']
## df_pla = df_ncam[df_ncam.species=='Plaice']
## df_red = df_ncam[df_ncam.species=='Redfish']
## df_ska = df_ncam[df_ncam.species=='Skate']
## df_wit = df_ncam[df_ncam.species=='Witch']
## df_yel = df_ncam[df_ncam.species=='Yellowtail']
## # process error
## df_pe = pd.concat([df_cod.pe, df_had.pe, df_hak.pe, df_pla.pe, df_red.pe, df_ska.pe, df_wit.pe, df_yel.pe], axis=1).mean(axis=1)
## # biomass
## df_B = pd.concat([df_cod.B, df_had.B, df_hak.B, df_pla.B, df_red.B, df_ska.B, df_wit.B, df_yel.B], axis=1).mean(axis=1)

# Load Groundfish 
df_ncam = pd.read_csv('/home/cyrf0006/data/NCAM/multispic_process_error.csv', index_col='year')
df_cod = df_ncam[df_ncam.species=='Atlantic Cod']
df_had = df_ncam[df_ncam.species=='Haddock']
df_hak = df_ncam[df_ncam.species=='White Hake']
df_pla = df_ncam[df_ncam.species=='American Plaice']
df_red = df_ncam[df_ncam.species=='Redfish spp.']
df_ska = df_ncam[df_ncam.species=='Skate spp.']
df_wit = df_ncam[df_ncam.species=='Witch Flounder']
df_yel = df_ncam[df_ncam.species=='Yellowtail Flounder']
df_tur = df_ncam[df_ncam.species=='Greenland Halibut']
df_wol = df_ncam[df_ncam.species=='Wolffish spp.']
# drop 3Ps
df_ncam = df_ncam[df_ncam.region!='3Ps']
# average all years
#df_ncam = df_ncam.groupby('year').mean()['process_error']
df_ncam = df_ncam.groupby('year').sum()['delta_biomass_kt']


## Load cod biomass data (Schijns et al. 2021)
df_cod = pd.read_csv('/home/cyrf0006/research/keynote_capelin/fig_S11D_data.csv', index_col='year')
Redline = df_cod.RedLine

## Load crab data (Mullowney)
## df_crab = pd.read_excel('/home/cyrf0006/research/keynote_capelin/biomasses.xlsx')
## df_crab = df_crab[df_crab.Region=='NL']
## df_crab.set_index('Year', inplace=True)
## df_crab = df_crab.tBIO/100000
df_crab = pd.read_csv('crab_unfished.csv')
# drop 3Ps
df_crab = df_crab[df_crab.division!='3PS']
# average all years
df_crab = df_crab.groupby('year').mean()['Cmnpt']

# Load Capelin biomass data
df_cap = pd.read_excel('/home/cyrf0006/data/capelin/age_disaggregated_2019.xlsx', index_col='year')
df_cap = df_cap['total no.billions']
df_cap_anom = (df_cap - df_cap[df_cap.index>=1990].mean()) / df_cap[df_cap.index>=1990].std()
df_cap_diff = df_cap.interpolate().diff()
#df_cap_diff_anom = (df_cap_diff - df_cap_diff[df_cap_diff.index>=1990].mean()) / df_cap_diff[df_cap_diff.index>=1990].std()

# Load Capelin biomass data USSR
df_cap2 = pd.read_csv('/home/cyrf0006/data/capelin/Fig2_capelin_biomass_1975_2019.csv', index_col='year')
#df_cap2_spring = df_cap2[(df_cap2.area=='spring acoustic') | (df_cap2.area=='ussr spring acoustic')] # spring only
df_cap2_spring = df_cap2[(df_cap2.area=='spring acoustic')] # spring only
df_cap2_fall = df_cap2[(df_cap2.area=='fall acoustic') | (df_cap2.area=='ussr fall acoustic')] # fall only

# ALL
df_cap2 = df_cap2['biomass ktonnes'].sort_index()
df_cap2 = df_cap2.groupby(['year']).mean()
#df_cap2_diff = df_cap2.interpolate().rolling(5, min_periods=3, center=False).mean().diff()

# SPRING
df_cap2_spring = df_cap2_spring['biomass ktonnes'].sort_index()
df_cap2_spring = df_cap2_spring.groupby(['year']).mean()
df_cap2_spring = df_cap2_spring.interpolate()

#df_cap2_spring_diff = df_cap2_spring.interpolate().rolling(3, min_periods=3, center=True).mean().diff()
#df_cap2_spring_diff = df_cap2_spring.interpolate().diff()

# FALL
df_cap2_fall = df_cap2_fall['biomass ktonnes'].sort_index()
df_cap2_fall = df_cap2_fall.groupby(['year']).mean()
df_cap2_fall_diff = df_cap2_fall.interpolate().rolling(5, min_periods=3, center=False).mean().diff()

# Load Mariano's biomass density (t/km2)
df_bio = pd.read_excel(open('RV_biomass_density.xlsx', 'rb'), sheet_name='data_only')
df_bio.set_index('Year', inplace=True)
df_bio_ave = df_bio['Average-ish biomass density'].dropna()

# Load Calfin proxy
df_cal = pd.read_csv('calanus_nlci_proxy.csv')
df_cal.set_index('Year', inplace=True)

# Load Primary Production
df_PP = pd.read_csv('PP/cmems_PP.csv')
df_PP.set_index('time', inplace=True)

# Load Calanus
df_cal = pd.read_csv('/home/cyrf0006/research/PeopleStuff/BelangerStuff/CorrelationData.csv', index_col='Year')
df_cal = df_cal['calfin']
# Abundance
df_cal_ab = pd.read_csv('/home/cyrf0006/research/PeopleStuff/BelangerStuff/CALFIN_abundance.csv', index_col='Year')
df_cal_ab = df_cal_ab['Mean abundance (log10 ind. m-2)']

####  ---- Loop on years --- ####
for years in years_list:
    print(years)
    p_year = p[(p.index.get_level_values('time').year>=years[0]) & (p.index.get_level_values('time').year<=years[1])]

    # average all years
    df = p_year.unstack()
    df = df.groupby(level=['lat']).mean() 

    #### ---- SLP ---- ####
    ## fig_name = 'SLP_map_' + np.str(years[0]) + '-' + np.str(years[1]) + '.png'
    ## fig_name2 = 'SLP_map_' + np.str(years[0]) + '-' + np.str(years[1]) + '.svg'
    ## print(fig_name)
    ## plt.clf()
    ## fig, ax = plt.subplots(nrows=1, ncols=1)

    ## m = Basemap(projection='ortho',lon_0=-40,lat_0=30, resolution='l', llcrnrx=-4000000, llcrnry=-2000000, urcrnrx=5000000, urcrnry=7000000)
    ## m.drawcoastlines()
    ## m.fillcontinents(color='tan')
    ## # draw parallels and meridians.
    ## m.drawparallels(np.arange(-90.,120.,30.))
    ## m.drawmeridians(np.arange(0.,420.,60.))
    ## #m.drawmapboundary(fill_color='aqua')
    ## plt.title("Sea Level Pressure - " + np.str(years[0]) + '-' + np.str(years[1]))

    ## #x,y = m(*np.meshgrid(df.columns.values,df.index.values))
    ## x,y = m(*np.meshgrid(df.columns.droplevel(None) ,df.index))
    ## c = m.contourf(x, y, df.values, v, cmap=plt.cm.inferno, extend='both');
    ## ct = m.contour(x, y, df_clim.values, 10, colors='k');
    ## cb = plt.colorbar(c)
    ## cb.set_label('SLP (mb)')
    ## plt.close('all')

    #### ---- Anomaly ---- ####
    anom = df - df_clim
    fig_name = 'anom_SLP_' + np.str(years[0]) + '-' + np.str(years[1]) + '.png'
    fig_name2 = 'anom_SLP_' + np.str(years[0]) + '-' + np.str(years[1]) + '.svg'
    print(fig_name)
    plt.clf()
    fig2, ax = plt.subplots(nrows=1, ncols=1)

    m = Basemap(projection='ortho',lon_0=-40,lat_0=40, resolution='l', llcrnrx=-4000000, llcrnry=-2000000, urcrnrx=5000000, urcrnry=7000000)
    m.drawcoastlines()
    m.fillcontinents(color='tan')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    #m.drawmapboundary(fill_color='aqua')

    #x,y = m(*np.meshgrid(df.columns.values,df.index.values))
    x,y = m(*np.meshgrid(df.columns.droplevel(None) ,df.index))
    #c = m.contourf(x, y, anom.values, np.linspace(-2.2, 2.2, 12), cmap=plt.cm.seismic, extend='both');
    c = m.contourf(x, y, anom.values, np.linspace(-1.8, 1.8, 10), cmap=plt.cm.seismic, extend='both');
    ct = m.contour(x, y, df_clim.values, 10, colors='k');
    cb = plt.colorbar(c)
    cb.set_label('SLP anomaly (mb)', fontsize=15)
    xBox, yBox = m([lon2, lon1, lon1, lon2, lon2], [lat2, lat2, lat1, lat1, lat2])
    m.plot(xBox, yBox, '--k', linewidth=2)
    plt.text(8400000, 12800000, np.str(years[0]) + '-' + np.str(years[1]), fontsize=18, fontweight='bold')

    #### ---- Add ecosystem trends ---- ####
    # Trends Cod (Regular's process error)
    #df_tmp2 = df_pe[(df_pe.index>=years[0]) & (df_pe.index<years[1])].dropna()
    #df_tmp2 = Redline[(Redline.index>=years[0]) & (Redline.index<years[1])].dropna()*1000
    df_tmp2 = df_ncam[(df_ncam.index>=years[0]) & (df_ncam.index<years[1])].dropna()
    if len(df_tmp2)>0:
        pp2 = np.polyfit(df_tmp2.index, df_tmp2.values, 1)
        trend2 = pp2[0]    
        if trend2>0:
            plt.annotate('+' + "{:.1f}".format(np.abs(trend2)) + r'$\rm \,kt\,yr^{-1}$', xy=(1, 0), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")
        elif trend2<0:
            plt.annotate('-' + "{:.1f}".format(np.abs(trend2)) + r'$\rm \,kt\,yr^{-1}$', xy=(1, 0), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")
        
   # Trends biomass density (Mariano)
    df_tmp4 = df_bio_ave[(df_bio_ave.index>=years[0]) & (df_bio_ave.index<years[1])].dropna()
    if len(df_tmp4)>2:
        pp = np.polyfit(df_tmp4.index, df_tmp4.values, 1)
        trend = pp[0]
        if trend>0:
            plt.annotate('+' + "{:.2f}".format(np.abs(trend)) + r'$\rm \,t\,km^{-2}\,yr^{-1}$', xy=(1, .08), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")
        elif trend<0:
            plt.annotate('-' + "{:.2f}".format(np.abs(trend)) + r'$\rm \,t\,km^{-2}\,yr^{-1}$', xy=(1,.08), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")

   ##  # Trends Crab (Mullowney)
   ##  ## df_tmp3 = df_crab[(df_crab.index>=years[0]) & (df_crab.index<years[1])].dropna()
   ##  ## if len(df_tmp3)>0:
   ##  ##     pp = np.polyfit(df_tmp3.index, df_tmp3.values, 1)
   ##  ##     trend3 = pp[0]
   ##  ##     if trend3>0:
   ##  ##         plt.annotate('+' + "{:.1f}".format(np.abs(trend3)), xy=(.97, .76), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right')
   ##  ##     elif trend3<0:
   ##  ##         plt.annotate('-' + "{:.1f}".format(np.abs(trend3)), xy=(.97, .76), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right')

    # Trends Capelin (DFO spring survey)
    df_tmp = df_cap2_spring[(df_cap2_spring.index>=years[0]) & (df_cap2_spring.index<years[1])].dropna()
    if len(df_tmp)>0:
        pp = np.polyfit(df_tmp.index, df_tmp.values, 1)
        trend = pp[0]
        if trend>0:
            plt.annotate('+' + "{:.1f}".format(np.abs(trend)) + r'$\rm \,kt\,yr^{-1}$', xy=(1, .16), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")
        elif trend<0:
            plt.annotate('-' + "{:.1f}".format(np.abs(trend)) + r'$\rm \,kt\,yr^{-1}$', xy=(1, .16), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")

   ##  ## # Trends Calfin (proxy)
   ##  ## df_tmp5 = df_cal[(df_cal.index>=years[0]) & (df_cal.index<years[1])].dropna()
   ##  ## if len(df_tmp5)>0:
   ##  ##     pp = np.polyfit(df_tmp5.index, df_tmp5.values, 1)
   ##  ##     trend = pp[0]*100
   ##  ##     if trend>0:
   ##  ##         plt.annotate('+' + "{:.1f}".format(np.abs(trend.squeeze())), xy=(1, .24), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")
   ##  ##     elif trend<0:
   ##  ##         plt.annotate('-' + "{:.1f}".format(np.abs(trend.squeeze())), xy=(1, .24), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")

    # Trends PP (cmems)
    df_tmp6 = df_PP[(df_PP.index>=years[0]) & (df_PP.index<years[1])].dropna()
    if len(df_tmp6)>0:
        anom = (df_tmp6.mean() - df_PP.mean()).values  #mg m-3 day-1
        if anom>0:
            plt.annotate('+' + "{:.2f}".format(np.abs(anom[0])) + r'$\rm \,mgC\,m^{-3}\,d^{-1}$', xy=(1,.84), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")
        elif anom<0:
            plt.annotate('-' + "{:.2f}".format(np.abs(anom[0])) + r'$\rm \,mgC\,m^{-3}\,d^{-1}$', xy=(1,.84), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")

    # Trends calfin (azmp)
    df_tmp7 = df_cal_ab[(df_cal_ab.index>=years[0]) & (df_cal_ab.index<years[1])].dropna()
    if len(df_tmp7)>0:
        anom = (df_tmp7.mean() - df_cal_ab.mean())  #mg m-3 day-1
        if anom>0:
            plt.annotate('+' + "{:.2f}".format(np.abs(anom)) + r'$\rm\,log_{10}(ind\,m^{-2})$', xy=(1,.76), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")
        elif anom<0:
            plt.annotate('-' + "{:.2f}".format(np.abs(anom)) + r'$\rm\,log_{10}(ind\,m^{-2})$', xy=(1,.76), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w")
                                    
    #### ---- Save Figure ---- ####
    #plt.suptitle('Fall surveys', fontsize=16)
    fig2.set_size_inches(w=8, h=6)
    fig2.savefig(fig_name, dpi=150)
    #fig2.savefig(fig_name2, format='svg')
    os.system('convert -trim ' + fig_name + ' ' + fig_name)
    plt.close()

plt.close('all')

os.system('montage anom_SLP_1948-1971.png anom_SLP_1971-1976.png anom_SLP_1976-1982.png anom_SLP_1982-1998.png anom_SLP_1998-2014.png anom_SLP_2014-2017.png -tile 2x3 -geometry +10+10  -background white  SLP_anom_capelin.png')


