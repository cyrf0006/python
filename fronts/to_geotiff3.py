
from osgeo import gdal
import numpy as np
from scipy.io import loadmat

def array_to_raster(array, x, y):
    """Array > Raster
    Save a raster from a C order array.

    :param array: ndarray
    """

    # Files info
    dst_filename = 'atiff.tiff'
    
    # Load matlab file
    front_dict = loadmat(infile,squeeze_me=True, struct_as_record=False)
    #print front_dict
    
    # You need to get those values like you did.
    x_pixels = len(x)  # number of pixels in x
    y_pixels = len(y)  # number of pixels in y
    PIXEL_SIZE = 1000  # size of the pixel...(in m?)         
    x_min = np.min(x)
    y_max = np.min(y)  # x_min & y_max are like the "top left" corner.
    wkt_projection = 'a projection in wkt that you got from other file'

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32, )

    dataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))  

    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
    return dataset, dataset.GetRasterBand(1)  #If you need to ret


# Load matlab file
infile = '../SST_atlantic_prob.mat'
front_dict = loadmat(infile,squeeze_me=True, struct_as_record=False)
#print front_dict

data = front_dict['probability']
lon = front_dict['lon']
lat = front_dict['lat']
lons = lon[1,:]
lats = lat[:,1]

array_to_raster(data, lons, lats)
