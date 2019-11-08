'''
A script wrote to correct timestamp problems related to GPS week roll over. See for example:
https://kb.meinbergglobal.com/kb/time_sync/gnss_systems/gps_week_number_rollover <https://kb.meinbergglobal.com/kb/time_sync/gnss_systems/gps_week_number_rollover

DIAGNOSTIC:
Everytime the glider connects to to Irridium, the time reset to 1970-01-01 00:00:00.
One the glider dives, the time increments are correct.

SOLUTION:
Get rid of all states where glider is at surface (remove all timestamps before glider dive, i.e., NavState=110).
Then assume that the time of the beginning of the dive is the time of the last email received from server for this specific yo.


Frederic.Cyr@dfo-mpo.gc.ca
November 2019


'''


import imaplib
import credentials
import re
import numpy as np
import pandas as pd
import os
import glob

## ---- preamble [some info to edit] ---- ##
imap_ssl_host = 'imap.gmail.com'
imap_ssl_port = 993
username = credentials.username()  
password = credentials.password()
mission_id = 'M333'
paths = ['/home/cyrf0006/data/gliders_data/SEA003/M333/Nav',
            '/home/cyrf0006/data/gliders_data/SEA003/M333/science']

## ---- PART 1 - get time from gmail ---- ##
print('PART 1 - get time from gmail')
# initialize fields to fill
lat_dec = []
lat_min = []
lon_dec = []
lon_min = []
time_str = []
cycle_no = []
date_str = []

# Login and retrieve email
M = imaplib.IMAP4_SSL(imap_ssl_host, imap_ssl_port)
M.login(username, password)
M.select()
status, messages = M.search(None, 'FROM', "bhairy.n@osupytheas.fr", 'SUBJECT', mission_id)

# Now extract Lat/Lon and time
for num in messages[0].split():
    typ, msg = M.fetch(num, '(RFC822)')
    # convert msg to a list of lines
    my_str = str(msg)
    lines = my_str.split("\\r\\n")    
    
    for line in lines:
        lat = re.findall(r'^.*>Lat<.*$', line)
        lon = re.findall(r'^.*>Lon<.*$', line)
        time = re.findall(r'^.*>Time<.*$', line)
        cycle = re.findall(r'^.*>Cycle Number<.*$', line)
        date = re.findall(r'^.*Date:.*$', line)

        if lat:
            lat = lat[0].split('"')[8]
            lat = re.findall(r"[-+]?\d*\.\d+|\d+", lat)
            lat_dec.append(float(lat[0]));
            lat_min.append(float(lat[1]));
        if lon:
            lon = lon[0].split('"')[8]
            lon = re.findall(r"[-+]?\d*\.\d+|\d+", lon)
            lon_dec.append(float(lon[0]));
            lon_min.append(float(lon[1]));
        if time:
            time = time[0].split('>')[4]
            time = time.strip('</td')
            time_str.append(time)
        if cycle:
            cycle = cycle[0].split('>')[4]
            cycle = cycle.strip('</td')
            cycle_no.append(np.int(cycle))
        if date:
            dd = date[0].split(' ')[2]
            mm = date[0].split(' ')[3]
            yyyy = date[0].split(' ')[4]                
            date_str.append(dd + ' ' + mm + ' ' + yyyy)
            
M.close()
M.logout()

# some cleaning
lat_dec = np.array(lat_dec)
lat_min = np.array(lat_min)
lon_dec = np.array(lon_dec)
lon_min = np.array(lon_min)
lat = lat_dec + lat_min/60
lon = lon_dec + lon_min/60
cycle_no = np.array(cycle_no)

# convert to dataFrame & drop duplicates
d = {'cycle':cycle_no, 'date':date_str, 'time':time_str, 'latitude':lat, 'longitude':lon}
df = pd.DataFrame(data=d)
df.drop_duplicates('cycle', keep='last', inplace=True) 
df.reset_index(inplace=True)
df.index = pd.to_datetime(df.date + ' ' + df.time)
df.drop(columns=['index', 'date', 'time'], inplace=True)
df_yos = df.copy()
del df


## ---- PART 2 - Load all ascii files and replace time ---- ##
print('PART 2 - Load all ascii files and replace time')

## global corrected_index
corrected_index=None

for path in paths:
    # Rename files with padding zeros
    print(' Rename files in in ' + path)
    list_files = os.listdir(path)
    
    
    for filename in list_files:
        new_file = filename.split('.')
        new_file[-1] = new_file[-1].zfill(3)  
        new_file = '.'.join(new_file)  
        os.system('cp ' + path + '/' + filename + ' ' + path + '/padded_' + new_file)
        
    list_files = glob.glob(path + '/padded_*')
    list_files.sort()
    for filename in list_files:
        filename = filename.split('/')[-1]
        new_filename = 'corrected_' + filename.split('_')[-1]
        # find yo number and time correction
        cycle = int(filename.split('.')[-1])
        time_corr = df_yos[df_yos.cycle==cycle].index
        if time_corr.empty: # in case of multiple yos (just continue from previous)
            time_corr = (pd.Series(corrected_index[-1]) + pd.Timedelta('10 seconds')).values
        
        # read ascii file
        df = pd.read_csv(path + '/' + filename, sep=';')

        # find when glider dives (get rid of everything before)
        if 'pld' in filename: # a science file
            idx_inflex = df.index[df['NAV_RESOURCE']==110].tolist()
        else: # a Nav file
            idx_inflex = df.index[df['NavState']==110].tolist()
            
        if len(idx_inflex)==0:
            continue
        df = df[df.index>=idx_inflex[0]]
        # remove last column
        df = df.iloc[:,0:-1] 

        # Set datetime index
        if 'pld' in filename: # a science file
            time_column = 'PLD_REALTIMECLOCK'
        else: # a navigation file
            time_column = 'Timestamp'
        df.set_index(time_column, inplace=True)

        df.index = pd.to_datetime(df.index)

        # correct time (I save corrected index in case I need to use it for next yo; case of multiple-yos)
        if corrected_index is not None:
            del corrected_index
        corrected_index = (df.index-df.index[0]) + time_corr.repeat(len(df.index))  
        df.index = corrected_index
        # reset index and rename column
        df.reset_index(inplace=True)
        df.rename(columns={'index':time_column}, inplace=True)

        # save corrected ascii files
        new_file = path + '/' + new_filename
        df.to_csv(new_file, sep=';', float_format='%.3f', date_format='%d/%m/%Y %H:%M:%S', index=False)
    # remove temporary padded files
    os.system('rm ' + path + '/padded_*')
