import glob
import os


HOST=cyrf@nl-bpo-dev.ent.dfo-mpo.ca
ORIG=/data/seabird
DEST=/home/cyrf0006/data/dev_database

get_list = "ssh -q cyrf@nl-bpo-dev.ent.dfo-mpo.ca 'find /data/seabird/* -type f -name *.p[1-2][0-9][0-9][0-9] -exec ls -l {} \;' > pfiles_and_time.list"
os.system(get_list)

#with open('pfiles_and_time.list') as f:
with open('alist.list') as f:
   lines = f.readlines()

   
with open('alist.list') as fp:  
    line = fp.readline()
    cnt = 1
    while line:
        print("Line {}: {}".format(cnt, line.strip()))
        line = fp.readline()
        cnt += 1   

with open('alist.list') as fp:  
   for line in fp:
        file = line[1].strip().split('/')[-1]
        
        
       print("Line {}: {}".format(cnt, line))
        
for yearfile in lists:
    outfile = os.path.splitext(yearfile)[0] + '.nc'
    p.pfiles_to_netcdf(yearfile, outfile, zbin=5, zmax=6000)
    print ' -> ' + outfile + ' done!'
    expr = 'mv ' + yearfile + ' ./list_done'
    os.system(expr)
