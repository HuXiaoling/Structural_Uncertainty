import numpy as np
import os, glob
import cc3d
import SimpleITK as sitk

def getLargestCC(segmentation):
    labels_out, N = cc3d.largest_k(segmentation, k=1, connectivity=26, delta=0, return_N=True) # Get k largest CC
    print("Largest CC size: {}".format(np.sum(labels_out)))
    return labels_out

def generate_final(skeleton, mask_binary):
    union = np.logical_or(skeleton, mask_binary).astype(np.uint8)
    largestCC = getLargestCC(union)
    return largestCC

def load_vol(filepath):
    sitkimg = sitk.ReadImage(filepath)
    arrimg = sitk.GetArrayFromImage(sitkimg)
    return arrimg

def save_vol(arr, dstdir, sitkrefpath):
    sitkimg = sitk.GetImageFromArray(arr)

    sitkrefimg = sitk.ReadImage(sitkrefpath)
    sitkimg.SetSpacing(sitkrefimg.GetSpacing())
    sitkimg.SetOrigin(sitkrefimg.GetOrigin())
    sitkimg.SetDirection(sitkrefimg.GetDirection())

    sitk.WriteImage(sitkimg, os.path.join(dstdir, sitkrefpath.split('/')[-1].replace(".nii.gz", "_reconstructed.nii.gz")))



if __name__ == "__main__":
    
    binary_pred_path = "/data/saumgupta/slicer-tool/datasets/vessel-data/nnunet/nnUNet_results/nnUNet/3d_fullres/Task101_VESSEL/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/PA_000100.nii.gz"

    binary_skeleton_path = "/data/saumgupta/slicer-tool/datasets/vessel-data/outputs/dmt/PA_000100_DHW.nii.gz"

    binary_pred = load_vol(binary_pred_path)
    binary_skeleton = load_vol(binary_skeleton_path)

    binary_reconstructed = generate_final(binary_skeleton, binary_pred)

    save_vol(binary_reconstructed, "/data/saumgupta/slicer-tool/datasets/vessel-data/outputs/dmt/", binary_skeleton_path)