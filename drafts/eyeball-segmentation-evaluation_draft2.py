################################################################################
##                      Eyeball Segmentation Evaluation                       ##
##  Compute the performance metrics (dice coefficient, intersection over      ##
##  union, Matthew's correlation coefficient, and accuracy, Hausdorff         ##
##  ) between the computer segmentation results and radiologist segmentation  ##
##  results. Visualize the results on the original CT scan and export both    ##
##  numerical result and overlaid image file to a customized location.        ##
################################################################################

import SimpleITK as sitk
import numpy as np
import os
import cv2
import pandas as pd

import matplotlib.pyplot as plt

from enum import Enum


# Use enumerations to represent the various evaluation measures
class PerformanceMetrics(Enum):
    dice, iou, mcc, acc, hausdorff_distance = range(5)

def performance_evaluation_file(computer_img, radiologist_img):
    """
    Compute the performance metrics for two image files.
    """

    results = np.zeros((1, len(PerformanceMetrics.__members__.items())))

    # Compute the performance metrics
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    overlap_measures_filter.Execute(computer_img, radiologist_img)
    results[0, PerformanceMetrics.dice.value] = overlap_measures_filter.GetDiceCoefficient()
    results[0, PerformanceMetrics.iou.value] = overlap_measures_filter.GetJaccardCoefficient()

    # Tp-true positive, Tn-true negative, Fp-false positive, Fn-false negative
    Fp = overlap_measures_filter.GetFalsePositiveError()
    Tn = 1 - Fp
    Fn = overlap_measures_filter.GetFalseNegativeError()
    Tp = 1 - Fn
    results[0, PerformanceMetrics.mcc.value] = (
    Tp * Tn - Fp * Fn) / np.sqrt((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn))
    results[0, PerformanceMetrics.acc.value] = (Tp + Tn) / (Tp + Fp + Tn + Fn)

    hausdorff_distance_filter.Execute(computer_img, radiologist_img)
    results[0, PerformanceMetrics.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()

    return results


def performance_evaluation_folder(images, export_path):
    """
    Compute performance metrics for corresponding image files in two folders.
    """

    results = np.zeros((1, len(PerformanceMetrics.__members__.items())))
    files = []

    for computer_img, radiologist_img, CT_img, file in images:
        files.append(file)
        results = np.concatenate((results, performance_evaluation_file(
            computer_img, radiologist_img)), axis=0)
        
    results = np.delete(results, 0, 0)

    visualize_performance_metrics(results, files, export_path)


def visualize_performance_metrics(results, files, export_path):
    # Graft result matrix into pandas data frames
    results_df = pd.DataFrame(data=results, index=files, 
                              columns=[name for name, _ in 
                              PerformanceMetrics.__members__.items()])
    results_df.plot(kind='bar').legend(bbox_to_anchor=(1.6, 0.9))
    results_df.to_csv(export_path + '/result.csv')


def visualize_segmentation(file, export_path, slice_number, CT_image, computer_seg, radiologist_seg, window_min, window_max):
    """
    Export a CT slice (in png) with both computerized (green) and radiologist 
    (red) segmented contours overlaid onto it. The contours are the edges of 
    the labeled regions.
    """

    CT_img = CT_image[:, :, slice_number]
    computer_img = computer_seg[:, :, slice_number]
    radiologist_img = radiologist_seg[:, :, slice_number]

    # Impose computerized contour on the CT slice
    computer_overlay = sitk.LabelMapContourOverlay(
        sitk.Cast(computer_img, sitk.sitkLabelUInt8),
        sitk.Cast(sitk.IntensityWindowing(CT_img, windowMinimum=window_min, 
        windowMaximum=window_max), sitk.sitkUInt8),
        opacity=1, contourThickness=[2, 2])
    outputName = export_path + "/" + file + \
        "_slice" + str(slice_number) + "_result.png"
    output_radiologist = export_path + file + "_radio.png"
    sitk.WriteImage(computer_overlay, outputName)
    sitk.WriteImage(radiologist_img, output_radiologist)

    # Impose radiologist contour on the CT slice
    radiologist_img = cv2.imread(output_radiologist, cv2.IMREAD_GRAYSCALE)
    result_overlay = cv2.imread(outputName)
    contours, _ = cv2.findContours(radiologist_img, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        mask = np.zeros(radiologist_img.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean, _, _, _ = cv2.mean(radiologist_img, mask=mask)
        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(result_overlay, [c], -1, (0, 0, 255), 1)

    cv2.imwrite(outputName, result_overlay)
    os.remove(output_radiologist)


def get_filename(file1, file2):
    """
    Extract the longest common prefix of two file names as the file name of the 
    new exported image.
    """

    n1 = len(file1)
    n2 = len(file2)
    result = ""
    j = 0
    i = 0

    while(i <= n1 - 1 and j <= n2 - 1):
        if (file1[i] != file2[j]):
            break
        result += (file1[i])
        i += 1
        j += 1

    return (result)


def read_folder(computer_files, radiologist_files, CT_files):

    images = []

    while (len(computer_files) != 1 and len(radiologist_files) != 1 and len(CT_files) != 1):
        computer_file = computer_files.pop()
        radiologist_file = radiologist_files.pop()
        CT_file = CT_files.pop()

        computer_image = sitk.ReadImage(computer_file)
        radiologist_image = sitk.ReadImage(radiologist_file)
        CT_image = sitk.ReadImage(CT_file)
        file = get_filename(os.path.split(computer_file)[
                            1], os.path.split(radiologist_file)[1])

        images.append(
            (computer_image, radiologist_image, CT_image, file))

    return images


def file_comparison(computer_image, radiologist_image, CT_image, file, export_path, slice_number, mode):

    visualize_segmentation(file, export_path, slice_number, CT_image=CT_image,  
            computer_seg=computer_image, radiologist_seg=radiologist_image,
            window_min=-1024, window_max=976)

    if (mode == 1):
        results = performance_evaluation_file(computer_image, radiologist_image)
        visualize_performance_metrics(results, [file], export_path)


def folder_comparison(computer_folder, radiologist_folder, CT_folder, export_path, mode):
    
    computer_files = []
    for file in os.listdir(computer_folder):
        computer_file = os.path.join(computer_folder, file)
        if (os.path.isfile(computer_file)):
            computer_files.append(computer_file)
    list.sort(computer_files)

    radiologist_files = []
    for file in os.listdir(radiologist_folder):
        radiologist_file = os.path.join(radiologist_folder, file)
        if (os.path.isfile(radiologist_file)):
            radiologist_files.append(radiologist_file)
    list.sort(radiologist_files)

    CT_files = []
    for file in os.listdir(CT_folder):
        CT_file = os.path.join(CT_folder, file)
        if (os.path.isfile(CT_file)):
            CT_files.append(CT_file)
    list.sort(CT_files)

    images = read_folder(computer_files, radiologist_files, CT_files)

    performance_evaluation_folder(images, export_path)
    
    for computer_image, radiologist_image, CT_image, file in images:
        slice_input = "Enter the slice number for " + computer_file + ": "
        slice_number = int(input(slice_input))
        if (slice_number >= CT_image.GetSize()[2]):
            print("Slice number out of bound!")

        file_comparison(computer_image, radiologist_image,
                        CT_image, file, export_path, slice_number, mode)
        

def main():
    mode = int(input(
        "Enter 1 to compare two image files, 2 to compare image files in two folders (File names must match): "))

    if (mode == 1):
        computer_file = input("Enter the path for computer image file: ")
        computer_image = sitk.ReadImage(computer_file)
        radiologist_file = input("Enter the path for radiologist image file: ")
        radiologist_image = sitk.ReadImage(radiologist_file)
        CT_file = input("Enter the path for CT image file: ")
        CT_image = sitk.ReadImage(CT_file)
        export_path = input("Enter the path for storing the export image: ")
        slice_number = int(input("Enter the slice number: "))
        file = get_filename(os.path.split(computer_file)[
            1], os.path.split(radiologist_file)[1])
        file_comparison(computer_image, radiologist_image,
                        CT_image, file, export_path, slice_number, mode)
    else: 
        computer_folder = input(
            "Enter the path for the folder of computer images: ")
        radiologist_folder = input(
            "Enter the path for the folder of radiologist images: ")
        CT_folder = input("Enter the path for the folder of CT images: ")
        export_path = input("Enter the path for storing the export image: ")
        folder_comparison(computer_folder, radiologist_folder,
                         CT_folder, export_path, mode)


if __name__ == '__main__':
    main()
