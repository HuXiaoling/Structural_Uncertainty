# convert png to nii.gz
import SimpleITK as sitk
import os, sys, glob
import numpy as np
from PIL import Image

src_file = "/home/saumya/uncertainty/Structural_Uncertainty/heat_map/img10_1_2_ori.png"
dst_dir = "/home/saumya/uncertainty/Structural_Uncertainty/output-for-tool"

def image_only():
    ### read png; converting input image to grayscale (0-255)
    arrayimage = np.expand_dims(np.asarray(Image.open(src_file)),0)

    ### covert to sitk-format
    sitkimage = sitk.GetImageFromArray(arrayimage)

    #sitkimage.SetSpacing(sitkimage_gt.GetSpacing())
    #sitkimage.SetOrigin(sitkimage_gt.GetOrigin())
    #sitkimage.SetDirection(sitkimage_gt.GetDirection())

    ### Save volume to file
    sitk.WriteImage(sitkimage, os.path.join(dst_dir, src_file.split('/')[-1].replace('.png', '.nii.gz')))

if __name__ == "__main__":
    image_only()