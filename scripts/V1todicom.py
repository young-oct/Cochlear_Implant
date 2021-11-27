# -*- coding: utf-8 -*-
# @Time    : 2021-11-26 7:35 p.m.
# @Author  : young wang
# @FileName: V1todicom.py
# @Software: PyCharm

import pydicom
import numpy as np
import concurrent
import glob
from functools import partial


from OssiviewBufferReader import OssiviewBufferReader
from os.path import join, isfile
from pydicom.uid import generate_uid
from skimage.filters import median
from skimage.morphology import cube
import matplotlib.pyplot as plt
import string
import os


def loadV1(input_file):
    obr = OssiviewBufferReader(input_file)
    data = obr.data['3D Buffer'].squeeze()

    data = 20 * np.log10(abs(data))

    data = imag2uint(data)

    data = clean(data)

    return median(data, cube(2))


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

    top = 30
    data[:, :, 0:top] = 0
    # data[:,:,256] = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.sqrt((i - 256) ** 2 + (j - 256) ** 2) >= 240:
                data[i, j, :] = 0

    return data


def oct_to_dicom(oct_file, seriesdescription,
                 dicom_folder, dicom_prefix):
    """
    convert pixel array [512,512,330] to DICOM format
    using MRI template, this template will be deprecated
    in the future once OCT template is received
    """
    data = loadV1(oct_file)

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
        ds.PixelSpacing = [0.05, 0.05]  # pixel spacing in x, y planes [mm]
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
        ds.PatientName = 'cadaver temporal bone'
        ds.PatientBirthDate = '20201123'
        ds.PatientAddress = ''

        # setting the dynamic range with WindowCenter and WindowWidth
        # lowest_visible_value = window_center — window_width / 2
        # highest_visible_value = window_center + window_width / 2

        ds.WindowCenter = '238'
        ds.WindowWidth = '416'

        # # set highest and lowest pixel values
        ds.LargestImagePixelValue = 446
        ds.SmallestImagePixelValue = 30
        dicom_file = join(dicom_folder, "%s%05d.dcm" % (dicom_prefix, i))

        pixel_data = data[:, :, i]
        
        ds.PixelData = pixel_data.tobytes()
        ds.save_as(dicom_file)
        dss.append(ds)
        all_files_exist = all_files_exist and isfile(dicom_file)
    return all_files_exist


if __name__ == '__main__':

    oct_files = []
    directory = '/Users/youngwang/Desktop/CI_P'
    import glob

    for filepath in glob.iglob(r'/Users/youngwang/Desktop/CI_P/*.bin'):
        oct_files.append(filepath)

    prefix_path = '/Users/youngwang/Desktop/DICOM Export'
    dis_path = list(string.ascii_lowercase)[0:len(oct_files)]
    export_paths= []
    for i in range(len(dis_path)):

        path = join(prefix_path,dis_path[i])
        export_paths.append(path)

        if not os.path.exists(path):
            os.mkdir(path)
        else:
            pass

    dicom_prefix = 'CI-cadaver'
    seriesdescription = dis_path

    for i in range(len(oct_files)):
        export_path = join(prefix_path, dis_path[i])

        if not os.path.exists(export_path):
            os.mkdir(export_path)
        else:
            pass

        oct_to_dicom(oct_file = oct_files[i],
                     seriesdescription = seriesdescription[i],
                     dicom_folder =export_path,
                     dicom_prefix = dicom_prefix)