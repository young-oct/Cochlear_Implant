# -*- coding: utf-8 -*-
# @Time    : 2021-12-10 8:47 p.m.
# @Author  : young wang
# @FileName: tympanotomy.py
# @Software: PyCharm

import os
from scipy import ndimage
from skimage import exposure
import numpy as np
from pydicom.uid import generate_uid
import pydicom
from os.path import join, isfile
import matplotlib.pyplot as plt
import glob


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
        # pixel_data[pixel_data <= 50] = 0
        ds.PixelData = pixel_data.tobytes()
        ds.save_as(dicom_file)
        dss.append(ds)
        all_files_exist = all_files_exist and isfile(dicom_file)
    return all_files_exist


''
if __name__ == '__main__':

    desktop_loc = os.path.expanduser('~/Desktop/GeoCorrection')
    patient_uid = 'patient'
    # study_uid = 'patient'
    # input_dir = join(desktop_loc, patient_uid, study_uid)

    study_uid = 'patient'
    input_dir = join(desktop_loc, patient_uid)
    file_extension = '*.npy'

    data_path = []
    for filename in glob.glob(join(input_dir, file_extension)):
        data_path.append(filename)

    volume_data = np.load(data_path[-1])


    umbom  = []
    tip = [ ]
    from scipy.signal import find_peaks, medfilt
    from scipy.ndimage import gaussian_filter1d,gaussian_filter

    slice = np.flip(volume_data[256,:,:].T)
    slice = gaussian_filter(slice, sigma=0.25)

    for i in range(slice.shape[0]):

        line = gaussian_filter1d(slice[i],sigma = 5 )

        peak_umbom, _ = find_peaks(line, height= 45 * np.mean(slice),distance = 20)

        if len(peak_umbom) >= 1:
            umboml = all(number > 200 for number in list(peak_umbom))
            umbomh = all(number <400 for number in list(peak_umbom))


            if umboml and umbomh:
                x = peak_umbom[0]
                y = line[x]
                umbom.append((x,y))

            else:
                pass

        else:
            pass

    for i in range(slice.shape[0]):

        line = gaussian_filter1d(slice[i], sigma=5)

        peak_tip, _ = find_peaks(line, height=25 * np.mean(slice), distance=20)

        if len(peak_tip) >= 1:

            tipl = all(number < 150 for number in list(peak_umbom))
            tiph = all(number > 450 for number in list(peak_umbom))

            if tipl and tiph:
                x = peak_tip[0]
                y = line[x]
                tip.append((i, x, y))
            else:
                pass

        else:
            pass

    print(len(umbom))
    plt.imshow(slice,'gray')
    pint = np.asarray(umbom)
    tip_array = np.asarray(tip)
    # x_min = min(tip_array[:,0])
    # ymin = slice[x_min]
    tip_cor = tip_array[0]
    plt.scatter(np.median(pint[:,0]),np.median(pint[:,1]))
    plt.scatter(tip_cor[0],tip_cor[1])
    plt.show()


    x1 = tip_cor[0]
    y1 = tip_cor[1]

    x2 = np.median(pint[:,0])
    y2 = np.median(pint[:,1])

    dy = y2 -y1
    dx = x2 - x1

    import math
    rad = math.atan2(dy,dx)
    degrees = math.degrees(rad)

    points = []

    portion = int(y2) - int(y1)
    for i in range (int(y1),int(y2), 1):

        scale = (i-int(y1))/portion
        y = y1 + scale/math.cos(rad)
        points.append((i,y))












