# -*- coding: utf-8 -*-
# @Time    : 2021-12-07 12:45 a.m.
# @Author  : young wang
# @FileName: cv2opt.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2021-12-07 12:34 a.m.
# @Author  : young wang
# @FileName: qhull_fast.py
# @Software: PyCharm

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
import matplotlib.pyplot as plt
import time

"https://coderedirect.com/questions/285029/very-slow-interpolation-using-scipy-interpolate-griddata"
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

    # step_x, step_z = complex(0,1)*i_dim, complex(0,1)*zdim
    #
    # xvals = np.arange(0, int(i_dim), 1)
    # yvals = np.arange(0, int(zdim), 1)
    #
    # _iz = _iz.reshape(i_dim * zdim, 2)
    # xq, zq = np.mgrid[0:int(i_dim):step_x, 0:int(zdim):step_z] # create a rectangular grid out of an array of x values
    # #
    # # f = interpolate.interp2d(xq, zq,  _v.flatten(), kind='cubic')
    # grid_ranges = [[0, i_dim, complex(0,1)*i_dim], [0, zdim, complex(0,1)*zdim]]
    # v = naturalneighbor.griddata(_iz.reshape(i_dim * zdim, 2), _v.flatten(), grid_ranges)
    return _iz

# Definition of the fast  interpolation process. May be the Tirangulation process can be removed !!
def interp_tri(xy):
    tri = qhull.Delaunay(xy)
    return tri


def interpolate(values, tri,uv,d=2):
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv- temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return np.einsum('nj,nj->n', np.take(values, vertices),  np.hstack((bary, 1.0 - bary.sum(axis=1, keepdims=True))))


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

    import cv2

    new_data = stack.T[::-1]

    # calculate the pixel coordinates of the
    # computational domain corners in the data array
    w, e, s, n = map_extent
    dx = float(e - w) / new_data.shape[1]
    dy = float(n - s) / new_data.shape[0]
    x = (lon.ravel() - w) / dx
    y = (n - lat.ravel()) / dy

    computational_domain_corners = np.float32(zip(x, y))

    data_array_corners = np.float32([[0, new_data.shape[0]],
                                     [0, 0],
                                     [new_data.shape[1], new_data.shape[0]],
                                     [new_data.shape[1], 0]])

    # Compute the transformation matrix which places
    # the corners of the data array at the corners of
    # the computational domain in data array pixel coordinates
    tranformation_matrix = cv2.getPerspectiveTransform(data_array_corners,
                                                       computational_domain_corners)

    # Make the transformation making the final array the same shape
    # as the data array, cubic interpolate the data placing NaN's
    # outside the new array geometry
    mapped_data = cv2.warpPerspective(new_data, tranformation_matrix,
                                      (new_data.shape[1], new_data.shape[0]),
                                      flags=2,
                                      borderMode=0,
                                      borderValue=np.nan)