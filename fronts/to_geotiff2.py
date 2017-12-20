import numpy as np
import gdal
from gdalconst import *
from osgeo import osr
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


def array_to_raster(array, lons, lats):
    """Array > Raster
    Save a raster from a C order array.

    :param array: ndarray
    """
    dst_filename = '/a_file/name.tiff'


    # You need to get those values like you did.
    x_pixels = 16  # number of pixels in x
    y_pixels = 16  # number of pixels in y
    PIXEL_SIZE = 3  # size of the pixel...        
    x_min = 553648  
    y_max = 7784555  # x_min & y_max are like the "top left" corner.
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
    return dataset, dataset.GetRasterBand(1)



def GetGeoInfo(FileName):
    SourceDS = gdal.Open(FileName, GA_ReadOnly)
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())    
    return GeoT, Projection

def CreateGeoTiff(Name, Array, driver, 
                  xsize, ysize, GeoT, Projection):
    DataType = gdal.GDT_Float32
    NewFileName = Name+'.tif'
    # Set up the dataset
    DataSet = driver.Create( NewFileName, xsize, ysize, 1, DataType )
            # the '1' is for band 1.
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection( Projection.ExportToWkt() )
    # Write the array
    DataSet.GetRasterBand(1).WriteArray( Array )
    return NewFileName

def ReprojectCoords(x, y,src_srs,tgt_srs):
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    x,y,z = transform.TransformPoint(x, y)
    return x, y

# Some Data (see beginning)
Data = prob
Lats = lats
Lons = lons

# A raster file that exists in the same approximate aregion.
RASTER_FN = 'some_raster.tif'

# Open the raster file and get the projection, that's the
# projection I'd like my new raster to have, it's 'projected',
# i.e. x, y values are numbers of pixels.
GeoT, TargetProjection, DataType = GetGeoInfo(RASTER_FN)
# Meanwhile my raster is currently in geographic coordinates.
SourceProjection = TargetProjection.CloneGeogCS()

# Get the corner coordinates of my array
LatSize, LonSize = len(Lats), len(Lons)
LatLow, LatHigh = Lats[0], Lats[-1]
LonLow, LonHigh = Lons[0], Lons[-1]
# Reproject the corner coordinates from geographic
# to projected...
TopLeft = ReprojectCoords(LonLow, LatHigh, SourceProjection, TargetProjection)
BottomLeft = ReprojectCoords(LonLow, LatLow, SourceProjection, TargetProjection)
TopRight = ReprojectCoords(LonHigh, LatHigh, SourceProjection, TargetProjection)
# And define my Geotransform
GeoTNew = [TopLeft[0],  (TopLeft[0]-TopRight[0])/(LonSize-1), 0,
           TopLeft[1], 0, (TopLeft[1]-BottomLeft[1])/(LatSize-1)]

# I want a GTiff
driver = gdal.GetDriverByName('GTiff')
# Create the new file...
NewFileName = CreateGeoTiff('Output', Data, driver, LatSize, LonSize, GeoTNew, TargetProjection)
