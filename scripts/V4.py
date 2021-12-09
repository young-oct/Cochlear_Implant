# -*- coding: utf-8 -*-
# @Time    : 2021-12-09 3:34 p.m.
# @Author  : young wang
# @FileName: V4.py
# @Software: PyCharm



import pydicom
import numpy as np
from OssiviewBufferReader import OssiviewBufferReader
from os.path import join, isfile
from pydicom.uid import generate_uid
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
import math
import scipy.spatial.qhull as qhull
import matplotlib.pyplot as plt
import time
from functools import partial
from multiprocessing import cpu_count, Pool, set_executable
from numba import jit, njit


def arrTolist(volume, Yflag=False):
    '''
    convert volume array into list for parallel processing
    :param volume: complex array
    :return:
    volume_list with 512 elements long, each element is Z x X = 330 x 512
    '''

    volume_list = []

    if not Yflag:
        for i in range(volume.shape[0]):
            volume_list.append(volume[i, :, :])
    else:
        for i in range(volume.shape[1]):
            volume_list.append(volume[:, i, :])

    return volume_list


def listtoarr(volume_list, Yflag=False):
    '''convert the volume list back to array format
    :param volume_list: complex array
    :return:
    volume with 512 elements long, each element is Z x X = 330 x 512
    '''

    if not Yflag:
        volume = np.empty((len(volume_list), 512, 330))
        for i in range(len(volume_list)):
            volume[i, :, :] = volume_list[i]
    else:
        volume = np.empty((512, len(volume_list), 330))
        for i in range(len(volume_list)):
            volume[:, i, :] = volume_list[i]

    return volume


def load_from_oct_file(oct_file):
    """
    read .oct file uising OssiviewBufferReader
    export an array in the shape of [512,512,330]
    the aarry contains pixel intensity data(20log) in float16 format
    """
    obr = OssiviewBufferReader(oct_file)
    data_fp16 = np.squeeze(obr.data)

    data = imag2uint(data_fp16)

    clean_data = clean(data)
    return clean_data


def imag2uint(data):
    """
    convert pixel data from the 255 range to unit16 range(0-65535)
    """

    # remove the low and high bounds of the pixel intensity data points
    # 45 and 130dB are heuristically chosen
    data = np.clip(data, 50, np.max(data))
    # pixel intensity normalization
    # for detail, please see wiki page
    # https://en.wikipedia.org/wiki/Normalization_(image_processing)
    # 446 is the maximum pixel in the MRI template
    data = (data - np.min(data)) * 446 / (np.max(data) - np.min(data))

    return np.uint16(np.around(data, 0))


def clean(data):
    #
    top = 30
    data[:, :, 0:top] = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.sqrt((i - 256) ** 2 + (j - 256) ** 2) >= 230:
                data[i, j, :] = 0

    return data


def oct_to_dicom(data, resolutionx, resolutiony, PatientName, seriesdescription,
                 dicom_folder, dicom_prefix):
    """
    convert pixel array [512,512,330] to DICOM format
    using MRI template, this template will be deprecated
    in the future once OCT template is received
    """
    # data = load_from_oct_file(oct_file)

    dss = []

    template_file = '../template data/template.dcm'
    ds = pydicom.dcmread(template_file)

    # SeriesInstanceUID refers to each series, and should be
    # unqie for each sesssion, and generate_uid() provides an unique
    # identifier
    ds.SeriesInstanceUID = generate_uid()

    all_files_exist = False

    # looping through all 330 slice of images with [512(row) x 512(column)]
    for i in range(data.shape[2]):
        # UID used for indexing slices
        ds.SOPInstanceUID = generate_uid()

        # update row and column numbers to be 512
        ds.Rows = data.shape[0]
        ds.Columns = data.shape[1]

        # define the bottom(assuming the middle plane to be zero,
        # that -165 * 30um(axial resolution) = -4.95 mm)
        # DICOM assumes physical dimension to be in mm unit
        bottom = -4.95

        # elevate the z by its axial resolution at a time
        z = bottom + (i * 0.03)

        # update meta properties

        # 1cm / 512 = 0.02 mm, needs to check with rob
        # this spacing should be calculated as radiant/pixel then mm to pixel
        #
        ds.PixelSpacing = [resolutionx, resolutiony]  # pixel spacing in x, y planes [mm]
        ds.SliceThickness = 0.03  # slice thickness in axial(z) direction [mm]
        ds.SpacingBetweenSlices = 0.03  # slice spacing in axial(z) direction [mm]
        ds.SliceLocation = '%0.2f' % z  # slice location in axial(z) direction
        ds.InstanceNumber = '%0d' % (i + 1,)  # instance number, 330 in total
        ds.ImagePositionPatient = [z, 0, 0]  # patient physical location
        ds.Manufacturer = 'Audioptics Medical Inc'
        ds.InstitutionName = 'Audioptics Medical'
        ds.InstitutionAddress = '1344 Summer St., #55, Halifax, NS, Canada'
        ds.StudyDescription = 'Example DICOM export'
        ds.StationName = 'Unit 1'
        ds.SeriesDescription = seriesdescription
        ds.PhysiciansOfRecord = ''
        ds.PerformingPhysicianName = ''
        ds.InstitutionalDepartmentName = ''
        ds.ManufacturerModelName = 'Mark II'
        ds.PatientName = PatientName
        ds.PatientBirthDate = '20201123'
        ds.PatientAddress = ''

        # setting the dynamic range with WindowCenter and WindowWidth
        # lowest_visible_value = window_center — window_width / 2
        # highest_visible_value = window_center + window_width / 2

        ds.WindowCenter = '248'
        ds.WindowWidth = '396'

        # # set highest and lowest pixel values
        ds.LargestImagePixelValue = 446
        ds.SmallestImagePixelValue = 50
        dicom_file = join(dicom_folder, "%s%05d.dcm" % (dicom_prefix, i))

        pixel_data = data[:, :, i]
        pixel_data[pixel_data <= 50] = 0
        ds.PixelData = pixel_data.tobytes()
        ds.save_as(dicom_file)
        dss.append(ds)
        all_files_exist = all_files_exist and isfile(dicom_file)
    return all_files_exist


# @jit(nopython=True)
def getPolarco():
    '''obtian correct polar coordinates from the distorted image

    since X and Y correction can be done independently with respect to Z,
    here we replace X_dim, Y_dim mentioend in Josh's proposal as i_dim
    for detailed math, see johs's proposal
    we can do this because
    (1) i_dim = X_dim = Y_dim = 512
    (2) azimuth and elevation are roughly the same 10 degrees (according to Dan)
    (3) 3D geometirc correction can be decomposed into two independent 2D correction
    please see "Real-time correction of geometric distortion artifact
     in large-volume optical coherence tomography paper'''

    i_dim, zdim, zmax = 512,330,int(330 * 4)
    degree = 10

    _iz = np.zeros((i_dim, zdim, 2))  # construct iz plane
    i0, z0 = int(i_dim / 2), zmax  # i0 is half of the i dimension

    i_phi = math.radians(degree)  # converting from degree to radiant

    ki = i_dim / (2 * i_phi)  # calculate pixel scaling factor for i dimension
    kz = 0.8  # calculate pixel scaling factor for z dimension, it should be Zmax/D, this is
    # a magic number kind works,

    for i in range(i_dim):
        for z in range(zdim):  # pixel coordinates conversion
            _iz[i, z, :] = [
                (z + kz * z0) * math.sin((i - i0) / ki) * math.cos((i - i0) / ki) + i0,
                (z + kz * z0) * math.cos((i - i0) / ki) * math.cos((i - i0) / ki) - kz * z0]

        # _iz.reshape(i_dim * zdim, 2): numpy stores arrays in row-major order
        # This means that the resulting two-column array will first contain all the x values,
        # then all the y values rather than containing pairs of (x,y) in each row
    # _iz = _iz.reshape(i_dim * zdim, 2)
    return _iz

# @jit(nopython=True)
def valueRemap(dis_image):
    """remap the data to match with the correct orientation"""

    _v = np.zeros(dis_image.shape)
    for i in range(dis_image.shape[0]):
        for z in range(dis_image.shape[1]):  # pixel coordinates conversion

            _v[i, z] = dis_image[i, -z]  # store the pixel date temporally and flip along the colume
            # axis
    return np.ravel(_v)
#
# def polar2cart(polarcords, uv,values):
#     x = polarcords[:,0]
#     y = polarcords[:,1]
#
#     """interpolate values from the target grid points"""
#
#     # initilize interpolator
#
#     interpolator = RegularGridInterpolator((x, y), values)
#     # interpolator = LinearNDInterpolator(tri, values)
#
#     # interpolate values from from with respect to the targeted
#     # cartisan coordinates
#     valueUpdate = interpolator(uv)
#
#     return np.fliplr(valueUpdate)
from numba import int32, float32,float64    # import the types

# from numba import jitclass
# spec = [
#             # a simple scalar field
#     ('polrcoordinate', float64[:,:])        # an array field
# ]
# def iniTri(polrcoordinate):
#     '''initialize triangulation'''
#     return qhull.Delaunay(polrcoordinate)


if __name__ == '__main__':

    oct_files = []
    directory = '/Users/youngwang/Desktop/GeoCorrection/phantom'
    import glob

    for filepath in glob.iglob(r'/Users/youngwang/Desktop/GeoCorrection/phantom/*.oct'):
        oct_files.append(filepath)

    oct_files.sort()

    raw_data = load_from_oct_file(oct_files[0])


    #get polarcords & initialize triangularization
    polarcords = getPolarco()

    # tri = iniTri(polarcords)

    # construct Cartesian grids
    xq, zq = np.mgrid[0:int(512), 0:int(330)]

    #
    # # x_list = arrTolist(raw_data, Yflag=False)
    # import matplotlib.pyplot as plt
    slice = raw_data[256,:,:]
    #
    uv = np.mgrid[0:int(512), 0:int(330)]
    #
    uv = np.zeros([xq.shape[0] * xq.shape[1], 2])
    #
    uv[:,0] = np.ravel(xq)
    uv[:,1] = np.ravel(zq)
    #
    # start = time.time()
    #
    # values = valueRemap(slice)

    # my_interpolating_function = RegularGridInterpolator((polarcords[:,0], polarcords[:,1], z), data)
    #
    # # values_interp = polar2cart(polarcords,uv,slice)
    # # values_interp =  partial(polar2cart,polarcords, xq, zq)
    # # values_interp_Qhull = np.fliplr(values_interp_Qhull)
    # # end = time.time()
    # # print(end -start)
    # #
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(slice)
    #
    # ax[1].imshow(values_interp)
    # plt.show()


    from scipy.interpolate import RegularGridInterpolator
    # def f(x, y):
    #     return 2 * x**3 + 3 * y**2
    x = np.linspace(1, 4, 11)
    y = np.linspace(4, 7, 22)
    xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
    # data = f(xg, yg)

    my_interpolating_function = RegularGridInterpolator((x, y), data)

    from scipy import interpolate

    for i in range(polarcords[0].shape[0]):
        for j in range(polarcords[1].shape[0]):
            for k in range(polarcords[2].shape[0]):
                vals[i, j, k] = func(np.array([pts[0][i], pts[1][j], pts[2][k]]))
    # f = interpolate.interp2d(polarcords[:,0], polarcords[:,1], slice, kind='linear')