# -*- coding: utf-8 -*-
# @Time    : 2021-12-05 4:30 p.m.
# @Author  : young wang
# @FileName: GPU.py
# @Software: PyCharm
import numba
import pydicom
import numpy as np
from scipy import interpolate
from OssiviewBufferReader import OssiviewBufferReader
from os.path import join, isfile
from pydicom.uid import generate_uid
from matplotlib import pyplot as plt
import naturalneighbor

import os
# from fast_interp import interp2d
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
import math
import time
from multiprocessing import cpu_count, Pool, set_executable
from numba import jit, njit
import numba_scipy as ns
from numba import vectorize,guvectorize

from interpolation.splines import LinearSpline, CubicSpline

from interpolation.splines import UCGrid, CGrid, nodes


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


def oct_to_dicom(data, PatientName, seriesdescription,
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
        ds.PixelSpacing = [0.02, 0.02]  # pixel spacing in x, y planes [mm]
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

# @njit
def cooridCO(test_slice):

    '''since X and Y correction can be done independently with respect to Z,
    here we replace X_dim, Y_dim mentioend in Josh's proposal as i_dim
    for detailed math, see johs's proposal

    we can do this because
    (1) X_dim = Y_dim = 512
    (2) azimuth and elevation are roughly the same 10 degrees (according to Dan)
    (3) 3D geometirc correction can be decomposed into two independent 2D correction
    please see "Real-time correction of geometric distortion artifacts
     in large-volume optical coherence tomography" paper


    zdim = 330, zmax = 4*330 (works well but a bit magic number)
    '''

    i_dim, zdim, zmax = 512, 330, 330 * 4
    _iz = np.zeros((i_dim, zdim, 2),dtype=np.float64)  # contruct iz plane
    _v = np.zeros((i_dim, zdim),dtype=np.float64)

    i0, z0 = int(i_dim / 2), zmax  # i0 is half of the i dimension

    i_phi = math.radians(10)  # converting from degree to radiant

    ki = i_dim / (2 * i_phi)  # calculate pixel scaling factor for i dimension
    kz = 0.8  # calculate pixel scaling factor for z dimension, it should be Zmax/D, this is
    # a magic number kind works,

    for i in range(i_dim):
        for z in range(zdim):  # pixel coordinates conversion
            _iz[i, z, :] = [
                (z + kz * z0) * math.sin((i - i0) / ki) * math.cos((i - i0) / ki) + i0,
                (z + kz * z0) * math.cos((i - i0) / ki) * math.cos((i - i0) / ki) - kz * z0]

            _v[i, z] = test_slice[i, -z] # store the pixel date temporally and flip along the colume
                                        #axis

    step_x, step_z = complex(0,1)*i_dim, complex(0,1)*zdim
    #
    # xvals = np.arange(0, int(i_dim), 1)
    # yvals = np.arange(0, int(zdim), 1)
    #
    # _iz = _iz.reshape(i_dim * zdim, 2)
    xq, zq = np.mgrid[0:int(i_dim):step_x, 0:int(zdim):step_z] # create a rectangular grid out of an array of x values
    #
    # f = interpolate.interp2d(xq, zq,  _v.flatten(), kind='cubic')
    grid_ranges = [[0, i_dim, complex(0,1)*i_dim], [0, zdim, complex(0,1)*zdim]]
    v = naturalneighbor.griddata(_iz.reshape(i_dim * zdim, 2), _v.flatten(), grid_ranges)
    return v
    # return np.fliplr(v) # flip it from left to right aka horizontal flip
#
# @njit
# def meshgrid(i_dim, zdim):
#     xvals = np.arange(0, int(i_dim), 1)
#     yvals = np.arange(0, int(zdim), 1)
#     xx = np.empty((len(xvals), len(yvals)))
#     for j, y in enumerate(yvals):
#         for i, x in enumerate(xvals):
#             xx[i, j] = x
#
#     yy = np.empty((len(xvals), len(yvals)))
#     for i, x in enumerate(xvals):
#         yy[i, :] = yvals
#
#     return xx,yy

# @njit
# @vectorize
# @guvectorize([float64[:,:,:],int64,int64, float64[:,:],int64[:,:],int64[:,:]], )
# @njit
# def Interdata(_iz,i_dim,zdim,_v,xq,zq):
#     v = griddata(_iz.reshape(i_dim * zdim, 2), _v.flatten(), (xq, zq), method='linear')
#     return v

if __name__ == '__main__':
    import numpy as np

    oct_files = []
    directory = '/Users/youngwang/Desktop/GeoCorrection'
    import glob

    for filepath in glob.iglob(r'/Users/youngwang/Desktop/GeoCorrection/*.oct'):
        oct_files.append(filepath)

    oct_files.sort()

    raw_data = load_from_oct_file(oct_files[0])

    # start = time.time()

    # x_list = arrTolist(raw_data[0:2,:,:], Yflag=False)
    stack = raw_data[256,:,:]
    a = cooridCO(stack)

    a = a.reshape(a.shape[0]*a.shape[1],2)
    func = interpolate.Rbf(a[0], a[1], stack.flatten(), kind='cubic')

    plt.imshow(stack)
    plt.show()

    # x, y = np.mgrid[0:512, 0:330]
    #
    # # x, y = np.mgrid[0:511, -1:1:20j]  # 生成(-1,1)间15*15的网格坐标点
    #
    # f = interp2d(x, y, stack.flatten(), kind='cubic')  # 一阶linear,三阶cubic,五阶quintic
    #
    # xnew = np.linspace(-1, 1, 51)  # 规定使用一维数组，而不能使用二维坐标点
    # ynew = np.linspace(-1, 1, 33)
    # znew = f(xnew, ynew)
    # plt.imshow(znew)
    # plt.show()

    # start = time.time()
    # a = cooridCO(stack)
    # end = time.time()
    # print(end-start)


    #
    # with Pool(processes=cpu_count()) as p:
    #     results_list = p.map(cooridCO, x_list)
    #
    #     p.close()
    #     p.join()
    #
    # data_x = listtoarr(results_list, Yflag=False)
    # data_xc = np.nan_to_num(data_x).astype(np.uint16)
    #
    # y_list = arrTolist(data_xc, Yflag=True)
    #
    # with Pool(processes=cpu_count()) as p:
    #     results_list = p.map(cooridCO, y_list)
    #
    #     p.close()
    #     p.join()
    #
    # data_y = listtoarr(results_list, Yflag=True)
    # data = np.nan_to_num(data_y).astype(np.uint16)
    #
    # end = time.time()
    # print(end - start)
    #
    # dicom_prefix = 'Phantom'
    # seriesdescription = ['GeoCorrection']
    #
    # export_path = '/Users/youngwang/Desktop/GeoCorrection/After'
    # PatientName = 'Phantom'
    # oct_to_dicom(data, PatientName=PatientName,
    #              seriesdescription=seriesdescription[0],
    #              dicom_folder=export_path,
    #              dicom_prefix=dicom_prefix)
    #
    # with open('/Users/youngwang/Desktop/p3D.npy', 'wb') as f:
    #     np.save(f, data)

    from scipy import interpolate

    from interpolation.splines import UCGrid, CGrid, nodes
    from interpolation.splines import LinearSpline, CubicSpline




    # start = time.time()
    # # x = np.arange(-165,165,2)
    # # y = np.arange(-256,256,2)
    # # grid = UCGrid((-165,165, 10), (-256,256, 10))
    # # gp = nodes(grid)
    #
    # a = np.array([-165, -256])  # lower boundaries
    # b = np.array([165, 256])  # upper boundaries
    # orders = np.array([100, 100])  # 50 points along each dimension
    # values = np.sin(gp[:,0]**2+gp[:,1]**2)
    # x = UCGrid((-165, 165, 100), (-256, 256, 100))
    # newgrid = nodes(x)
    #
    # lin = LinearSpline(a, b, orders, values)
    #
    # V = lin(newgrid)

    # S = np.random.random((10 ** 6, 3))  # coordinates at which to evaluate the splines

    # multilinear
    # lin = LinearSpline(a, b, orders, values)
    # V = lin(S)
    # xx,yy = np.meshgrid(x,y)

    # z = np.sin(gp[:,0]**2+gp[:,1]**2)
    # newgrid = UCGrid((-165,165,0.5), (-256,256,0.5))

    # f = interp2d(x,y,z)
    # f = interp2d([-165,-256],[165,256],[0.5,0.5],z)
    #
    # xnew = np.arange(-165,165,0.5)
    # ynew = np.arange(-256,256,0.5)
    # znew = f(xnew,ynew)
    # end = time.time()
    # print(end- start )
    # plt.imshow(V)
    # plt.show()

    # nx = 50
    # ny = 37
    # xv, xh = np.linspace(0, 1, nx, endpoint=True, retstep=True)
    # yv, yh = np.linspace(0, 2 * np.pi, ny, endpoint=False, retstep=True)
    # x, y = np.meshgrid(xv, yv, indexing='ij')
    #
    # test_function = lambda x, y: np.exp(x) * np.exp(np.sin(y))
    # f = test_function(x, y)
    # test_x = -xh / 2.0
    # test_y = 271.43
    # fa = test_function(test_x, test_y)
    #
    # ax = np.arange(-165, 165, 2)
    # ay = np.arange(-165, 165, 2)
    # interpolater = interp2d([0, 0], [1, 2 * np.pi], [xh, yh], f, k=5, p=[False, True], e=[1, 0])
    # fe = interpolater(ax, ay)

    # from fast_interp import interp2d
    # import numpy as np
    #
    # # nx = 50
    # # ny = 37
    # # xv, xh = np.linspace(0, 1,       nx, endpoint=True,  retstep=True)
    # # yv, yh = np.linspace(0, 2*np.pi, ny, endpoint=False, retstep=True)
    # x = np.arange(-165, 165, 2)
    # y = np.arange(-256,256,2)
    #
    # xx, yy = np.meshgrid(x,y)
    #
    # test_function = lambda x, y: 2*np.exp(x)*np.exp(np.sin(y))
    # f = np.sin(xx**2+yy**2)
    # # plt.imshow(f)
    # # plt.show()
    # # test_x = 0.05
    # # test_y = 271.43
    # # fa = test_function(test_x, test_y)
    #
    # interpolater = interp2d([-165,-256], [165,256], [2,2], f, k=3)
    # # fe = interpolater(test_x, test_y)
    # xnew = np.arange(-165, 165, 2)
    # ynew = np.arange(-256,256,2)
    # xxn, yyn = np.meshgrid(x, y)
    #
    # # xv = np.linspace(0, 1,       3*nx, endpoint=True,  retstep=True)
    # # yv = np.linspace(0, 2*np.pi, 3*ny, endpoint=False, retstep=True)
    # fe = interpolater(xxn, yyn)
    # # plt.imshow(f)
    # # plt.show()
    # fig,ax = plt.subplots(1,2)
    # ax[0].imshow(f)
    # ax[1].imshow(fe)
    # plt.show()
    #


