################################################################################
##                      Eyeball Segmentation Evaluation                       ##
##  Compute specified performance metrics (only support dice coefficient,     ##
##  intersection over union, Matthew's correlation coefficient,accuracy,      ##
##  and Hausdorff distance) between the computer segmentation results and     ##
##  radiologist segmentation results. Visualize the results on the original   ##
##  CT scan and export both numerical result and overlaid image file to a     ##
##  customized location.                                                      ##
################################################################################


import SimpleITK as sitk
import numpy as np
import os
import cv2
import pandas as pd


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

    while (i <= n1 - 1 and j <= n2 - 1):
        if (file1[i] != file2[j]):
            break
        result += (file1[i])
        i += 1
        j += 1

    return result


def read_folder(folder_path):
    file_paths = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if (os.path.isfile(file_path)):
            file_paths.append(file_path)
    list.sort(file_paths)

    images = []
    for path in file_paths:
        images.append(sitk.ReadImage(path))

    return (file_paths, images)


def get_images_filename(mode, comp_path, radio_path):
    filename = []
    if (mode == 0):
        if (os.path.isfile(comp_path) and os.path.isfile(radio_path)):
            comp_img = [sitk.ReadImage(comp_path)]
            radio_img = [sitk.ReadImage(radio_path)]
            filename.append(get_filename(os.path.split(
                comp_path)[1], os.path.split(radio_path)[1]))
        else:
            exit("Invalid file!")
    else:
        if (os.path.isdir(comp_path) and os.path.isdir(radio_path)):
            (comp_file_paths, comp_img) = read_folder(comp_path)
            (radio_file_paths, radio_img) = read_folder(radio_path)
            if (len(comp_img) != len(radio_img)):
                exit("Numbers of files in folders are unmatched!")
            for i in range(len(comp_file_paths)):
                filename.append(get_filename(os.path.split(comp_file_paths[i])[
                                1], os.path.split(radio_file_paths[i])[1]))
        else:
            exit("Invalid directory!")

    return (comp_img, radio_img, filename)


def valid_input_perform_eval(metrics_input):
    """
    Check the validity of the input for computing performance metrics.
    """

    if (len(metrics_input) == 0):
        return None
    else:
        metrics = {"acc": 0, "dice":1, "hausdorff":2, "iou":3, "mcc":4}
        metrics_indicator = [0, 0, 0, 0, 0]
        for metric in metrics_input:
            if (metric in metrics):
                metrics_indicator[metrics[metric]] = 1
            else:
                return None
        return metrics_indicator


def valid_input_visualize_seg(mode, file_number, CT_path):
    """
    Check the validity of the input for visualizing segmentation.
    """

    CT_image = []

    if (mode == 0):
        if (os.path.isfile(CT_path)):
            CT_image = [sitk.ReadImage(CT_path)]
        else:
            print("Invalid file!")
            return None
    else:
        if (os.path.isdir(CT_path)):
            (_, CT_image) = read_folder(CT_path)
            if (len(CT_image) != file_number):
                print("Unmatched number of CT images!")
                return None
        else:
            print("Invalid directory!")
            return None
    return CT_image
    

def get_slice(filename):
    slice_number = []
    for file in filename:
        slice_number.append(int(input(
            "Enter the slice number for {}: ".format(file))))
    return slice_number


class EyeballSegEval(object):
    def __init__(self, mode, comp_path, radio_path, export_path):
        self.mode = mode
        self.comp_path = comp_path
        self.radio_path = radio_path
        self.export_path = export_path
        (self.comp_img, self.radio_img, self.filename) = get_images_filename(
            self.mode, self.comp_path, self.radio_path)

            
    def performance_evaluation(self, metrics):
        """
        Given a list of performance metrics, compute for the image files.
        """

        metrics_indicator = valid_input_perform_eval(metrics)
        if (metrics_indicator == None):
            print("Invalid metrics!")
            return
        else:
            results = np.zeros((len(self.comp_img), len(metrics)))
            overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

            for i in range(len(self.comp_img)):
                # Cannot compute for unmatched image files
                if (self.comp_img[i].GetSpacing() != self.radio_img[i].GetSpacing()):
                    print(
                        "Spacing of corresponding image files ({}) don't match!".format(self.filename[i]))
                    results[i] = [float("NaN")] * len(metrics)
                else: 
                    overlap_measures_filter.Execute(
                        self.comp_img[i], self.radio_img[i])
                    Fp = overlap_measures_filter.GetFalsePositiveError()
                    Tn = 1 - Fp
                    Fn = overlap_measures_filter.GetFalseNegativeError()
                    Tp = 1 - Fn
                    if (metrics_indicator[1]):
                        results[i, 1] = overlap_measures_filter.GetDiceCoefficient()
                    if (metrics_indicator[3]):
                        results[i, 3] = overlap_measures_filter.GetJaccardCoefficient()
                    if (metrics_indicator[4]):
                        results[i, 4] = (
                            Tp * Tn - Fp * Fn) / np.sqrt((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn))
                    if (metrics_indicator[0]):
                        results[i, 0] = (Tp + Tn) / (Tp + Fp + Tn + Fn)
                    hausdorff_distance_filter.Execute(
                        self.comp_img[i], self.radio_img[i])
                    if (metrics_indicator[2]):
                        results[i, 2] = hausdorff_distance_filter.GetHausdorffDistance()
            
            results_df = pd.DataFrame(
                data=results, index=self.filename, columns=sorted(metrics))
            results_df.to_csv(self.export_path + '/results.csv')

    
    def visualize_segmentation(self, CT_path, window_min=-1024, window_max=976):
        """
        Given a slice number for each image comparison, export a CT slice (in 
        png) with both computerized (green) and radiologist (red) segmented 
        contours overlaid. The contours are the edges of the labeled regions.
        """

        CT_image = valid_input_visualize_seg(
            self.mode, len(self.comp_img), CT_path)
        if (CT_image == None):
            return
        else:
            slice_number = get_slice(self.filename)
            for i in range(len(CT_image)):
                computer = self.comp_img[i][:, :, slice_number[i]]
                radiologist = self.radio_img[i][:, :, slice_number[i]]
                CT = CT_image[i][:, :, slice_number[i]]

                # Impose computerized contour on the CT slice
                computer_overlay = sitk.LabelMapContourOverlay(sitk.Cast(computer, sitk.sitkLabelUInt8), sitk.Cast(sitk.IntensityWindowing(
                    CT, windowMinimum=window_min, windowMaximum=window_max), sitk.sitkUInt8), opacity=1, contourThickness=[2, 2])
                outputName = self.export_path + "/" + \
                    self.filename[i] + "_slice" + \
                    str(slice_number[i]) + "_result.png"
                output_radio = self.export_path + \
                    self.filename[i] + "_radio.png"
                sitk.WriteImage(computer_overlay, outputName)
                sitk.WriteImage(radiologist, output_radio)

                # Impose radiologist contour on the CT slice
                radiologist = cv2.imread(output_radio, cv2.IMREAD_GRAYSCALE)
                result_overlay = cv2.imread(outputName)
                contours, _ = cv2.findContours(
                    radiologist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for i, c in enumerate(contours):
                    mask = np.zeros(radiologist.shape, np.uint8)
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mean, _, _, _ = cv2.mean(radiologist, mask=mask)
                    cv2.drawContours(result_overlay, [c], -1, (0, 0, 255), 1)
                cv2.imwrite(outputName, result_overlay)
                os.remove(output_radio)
            

if __name__ == "__main__":
    test1 = EyeballSegEval(
        0, "/Users/catherine/Desktop/Research/eyeball-segmentation-evaluation/eyeball-computer-results/HB039124OAV_00351_2015-07-13_4_img.nii", "/Users/catherine/Desktop/Research/eyeball-segmentation-evaluation/msk-eyeball-radiologist-results/HB039124OAV_00351_2015-07-13_4_msk.nii", "/Users/catherine/Desktop/Research/eyeball-segmentation-evaluation/results")
    test1.performance_evaluation(["dice", "acc", "hausdorff"])
    test1.visualize_segmentation(
        "/Users/catherine/Desktop/Research/eyeball-segmentation-evaluation/img-eyeball/HB039124OAV_00351_2015-07-13_4_img.nii")

    test2 = EyeballSegEval(
        1, "/Users/catherine/Desktop/Research/eyeball-segmentation-evaluation/eyeball-computer-results", "/Users/catherine/Desktop/Research/eyeball-segmentation-evaluation/msk-eyeball-radiologist-results", "/Users/catherine/Desktop/Research/eyeball-segmentation-evaluation/results")
    test2.performance_evaluation(["dice", "acc", "hausdorff", "mcc", "iou"])
    test2.visualize_segmentation(
        "/Users/catherine/Desktop/Research/eyeball-segmentation-evaluation/img-eyeball")

