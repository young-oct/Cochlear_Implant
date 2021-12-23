# -*- coding: utf-8 -*-
# @Time    : 2021-12-16 1:03 p.m.
# @Author  : young wang
# @FileName: Zeiss.py
# @Software: PyCharm

import pydicom
import numpy as np

import glob

from OssiviewBufferReader import OssiviewBufferReader
from os.path import join, isfile
from pydicom.uid import generate_uid
from skimage.filters import median
from skimage.morphology import cube
import matplotlib.pyplot as plt
import string
import os
from pydicom.uid import ExplicitVRLittleEndian

def load_from_oct_file(oct_file):
    """
    read .oct file uising OssiviewBufferReader
    export an array in the shape of [512,512,330]
    the aarry contains pixel intensity data(20log) in float16 format
    """
    obr = OssiviewBufferReader(oct_file)
    data_fp16 = np.squeeze(obr.data)

    data_int = imag2uint(data_fp16)

    data = clean(data_int)

    return data

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
    data = (data - np.min(data)) * 255 / (np.max(data) - np.min(data))

    return np.uint8(np.around(data, 0))

def clean(data):
    #
    # top = 40

    top_remove = 5
    bottom_remove = 50

    index = int(330-top_remove)

    data[:, :, index::] = np.zeros(data[:, :, index::].shape )
    data[:, :, 0::index] = np.zeros(data[:, :, 0::index].shape)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.sqrt((i - 256) ** 2 + (j - 256) ** 2) >= 200:
                data[i, j, :] = 0

    return data


def oct_to_Zeissdicom(oct_file,PatientName,seriesdescription, dicom_folder, dicom_prefix):
    """
    convert pixel array [512,512,330] to DICOM format
    using MRI template, this template will be deprecated
    in the future once OCT template is received
    """
    data = load_from_oct_file(oct_file)

    data = np.uint8(data)

    # data[:,:,-10::] = 0

    dss = []

    template_file = '../template data/zeiss_enface.dcm'
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
        ds.PixelSpacing = [0.03, 0.03]  # pixel spacing in x, y planes [mm]
        ds.SliceThickness = 0.03  # slice thickness in axial(z) direction [mm]
        ds.SpacingBetweenSlices = 0.03  # slice spacing in axial(z) direction [mm]
        ds.SliceLocation = '%0.2f' % z  # slice location in axial(z) direction
        ds.InstanceNumber = '%0d' % (i + 1,)  # instance number, 330 in total
        ds.ImagePositionPatient = [z, 0, 0]  # patient physical location
        # ds.Manufacturer = 'Audioptics Medical Inc'
        # ds.InstitutionName = 'Audioptics Medical'
        # ds.InstitutionAddress = '1344 Summer St., #55, Halifax, NS, Canada'
        # ds.StudyDescription = 'Example DICOM export(20)'
        # ds.StationName = 'Unit 1'
        ds.SeriesDescription = seriesdescription
        # ds.PhysiciansOfRecord = ''
        # ds.PerformingPhysicianName = ''
        # ds.InstitutionalDepartmentName = ''
        # ds.ManufacturerModelName = 'Mark II'
        ds.PatientName = PatientName
        # ds.PatientBirthDate = '20201123'
        # ds.PatientAddress = ''


        # setting the dynamic range with WindowCenter and WindowWidth
        # lowest_visible_value = window_center — window_width / 2
        # highest_visible_value = window_center + window_width / 2

        # ds.WindowCenter = '248'
        # ds.WindowWidth = '396'

        # # set highest and lowest pixel values
        # ds.LargestImagePixelValue = 255
        # ds.SmallestImagePixelValue = 0
        dicom_file = join(dicom_folder, "%s%05d.dcm" % (dicom_prefix, i))

        pixel_data = data[:, :, i]
        ds.PixelData = pixel_data.tobytes()
        ds['PixelData'].is_undefined_length = False

        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        # ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        ds.save_as(dicom_file)
        dss.append(ds)
        all_files_exist = all_files_exist and isfile(dicom_file)
    return all_files_exist

if __name__ == '__main__':

    #
    oct_files = []
    desktop = os.path.expanduser('~/Desktop')
    dst = desktop+'/*.oct'

    for filepath in glob.iglob(dst):
        oct_files.append(filepath)
    oct_files.sort()

    oct_file = oct_files[0]
    PatientName = 'OCT ear'
    seriesdescription = 'sample ear'

    dicom_folder = join(desktop,'ear dicom')

    try:
        os.makedirs(dicom_folder)
    except FileExistsError:
        # directory already exists
        pass

    # dicom_prefix = 'ear'
    # oct_to_Zeissdicom(oct_file,PatientName,
    #                   seriesdescription, dicom_folder, dicom_prefix)
