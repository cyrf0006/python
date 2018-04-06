import glob
import os
import pandas as pd
import numpy as np

host = 'cyrf@nl-bpo-dev.ent.dfo-mpo.ca' 
path = '/data/seabird'
list_files = 'pfiles_and_time.list'
list_files = 'test_allfiles.list'
my_password = 'orwe123!'

# DO NOT CALL NOW
#get_list = "ssh -q cyrf@nl-bpo-dev.ent.dfo-mpo.ca 'find /data/seabird/* -type f -name *.p[1-2][0-9][0-9][0-9] -exec ls -l {} \;' > pfiles_and_time.list"
get_list_command = "ssh -q " + host + " 'find " + path + "/* -type f -name *.p[1-2][0-9][0-9][0-9] -exec ls -l {} \;' > " + list_files
os.system(get_list_command)

d = []
with open(list_files) as fp:  
   for idx, line in enumerate(fp):
        line = ' '.join(line.split())
        #print line
        n = line.strip().split(' ')[5:8]
        d.append({'file_name':line.strip().split('/')[-1],\
        'last_update': pd.to_datetime(n[1]+'-'+n[0]+'-'+n[2]),\
        'full_path':line.strip().split(' ')[-1]})
                   
df = pd.DataFrame(d)


df = df.set_index('last_update')
df.drop_duplicates(subset=['file_name'], keep='last', inplace=False)


for file in df['full_path'].values:
        command = 'sshpass -p ' + my_password + ' scp ' + host + ':' + path + file + ' .' 
        print command
        os.system(command)
