from osgeo import gdal, osr, ogr
import numpy as np
from scipy.io import loadmat

# Files info
infile = '../SST_atlantic_prob.mat'
outfile = 'prob_atlantic_1985-2010.tif'

# Load matlab file
front_dict = loadmat(infile,squeeze_me=True, struct_as_record=False)
#print front_dict

prob = front_dict['probability']
lon = front_dict['lon']
lat = front_dict['lat']
lons = lon[1,:]
lats = lat[:,1]

xres = lons[1] - lons[0]
yres = lats[1] - lats[0]
ysize = len(lats)
xsize = len(lons)
ulx = lons[0] - (xres / 2.)
uly = lats[-1] - (yres / 2.)



driver = gdal.GetDriverByName('GTiff')
ds = driver.Create(outfile,
               xsize, ysize, 1, gdal.GDT_Byte, )
##################################
#pixel type of gdal.GDT_Float32 results in transparent image
##################################

# this assumes the projection is Geographic lat/lon WGS 84
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326) # WGS?
#srs.ImportFromEPSG(3395) # Mercator?
#srs.ImportFromEPSG(3857) # Web Mercator?
ds.SetProjection(srs.ExportToWkt())
gt = [ulx, xres, 0, uly, 0, yres ]
ds.SetGeoTransform(gt)
outband=ds.GetRasterBand(1)
outband.SetStatistics(np.min(prob), np.max(prob), np.average(prob), np.std(prob))
outband.WriteArray(prob)
ds = None
