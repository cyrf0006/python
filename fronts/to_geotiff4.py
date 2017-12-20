from osgeo import gdal
from scipy.io import loadmat


def ascii_to_tiff(infile, outfile):
    """
    Transform an XYZ ascii file without a header to a projected GeoTiff

    :param infile (str): path to infile ascii location
    :param outfile (str): path to final GTiff
    :param refIm (str): path to a reference image made from the lat lon pair centriods

    """

    ## im = gdal.Open(refIm)
    ## ima = gdal.Open(refIm).ReadAsArray()
    ## row = ima.shape[0];
    ## col = ima.shape[1]

    
    # Load matlab file
    front_dict = loadmat(infile,squeeze_me=True, struct_as_record=False)
    prob = front_dict['probability']
    lon = front_dict['lon']
    lat = front_dict['lat']
    lons = lon[1,:]
    lats = lat[:,1]

    # create grid    
    ## xmin, xmax, ymin, ymax = [min(lon), max(lon), min(lat), max(lat)]
    ## xi = np.linspace(xmin, xmax, col)
    ## yi = np.linspace(ymin, ymax, row)
    ## xi, yi = np.meshgrid(xi, yi)

    ## # linear interpolation
    ## zi = ml.griddata(lon, lat, data, xi, yi, interp='linear')
    ## final_array = np.asarray(np.rot90(np.transpose(zi)))

    # translation:
    final_array = prob
    row = prob.shape[0]
    col = prob.shape[1]
    
    
    # projection
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(outfile, col, row, 1, gdal.GDT_Float32)
    dst_ds.GetRasterBand(1).WriteArray(final_array)
    prj = im.GetProjection()
    dst_ds.SetProjection(prj)

    gt = im.GetGeoTransform()
    dst_ds.SetGeoTransform(gt)
    dst_ds = None

    final_tif = gdal.Open(outfile, GA_ReadOnly).ReadAsArray()
    return final_tif

ascii_to_tiff('../SST_atlantic_prob.mat', 'atiff.tiff')
