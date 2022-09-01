import subprocess

import sys
from matplotlib import image as mpimg
import numpy as np
import os
import time
import cv2
import SimpleITK as sitk

t0 = time.time()

DIPHA_CONST = 8067171840
DIPHA_IMAGE_TYPE_CONST = 1
DIM = 3

dipha_output_filename = 'inputs/complex.bin'
vert_filename = 'inputs/vert.txt'
dipha_edge_filename = 'inputs/dipha.edges';
dipha_edge_txt = 'inputs/dipha_edges.txt';

def dmt_2d(patch, Th):
    nx, ny = patch.shape
    nz = 1
    im_cube = np.zeros([nx, ny, nz])
    im_cube[:, :, 0] = patch

    with open(dipha_output_filename, 'wb') as output_file:
        np.int64(DIPHA_CONST).tofile(output_file)
        np.int64(DIPHA_IMAGE_TYPE_CONST).tofile(output_file)
        np.int64(nx * ny * nz).tofile(output_file)
        np.int64(DIM).tofile(output_file)
        np.int64(nx).tofile(output_file)
        np.int64(ny).tofile(output_file)
        np.int64(nz).tofile(output_file)
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    val = int(-im_cube[i, j, k]*255)
                    np.float64(val).tofile(output_file)
        output_file.close()

    with open(vert_filename, 'w') as vert_file:
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    vert_file.write(str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(int(-im_cube[i, j, k] * 255)) + '\n')
        vert_file.close()

    subprocess.call(["mpiexec", "-n", "1", "dipha-graph-recon/build/dipha", "inputs/complex.bin", "inputs/diagram.bin", "inputs/dipha.edges", str(nx), str(ny), str(nz)])

    def fread(fid, nelements, dtype):

        data_array = np.fromfile(fid, dtype, nelements)
        data_array.shape = (nelements, 1)
        return data_array

    fid = open(dipha_edge_filename, 'r' );

    dipha_identifier = fread(fid, 1, 'int64' );
    diagram_identifier = fread(fid, 1, 'int64' );
    num_pairs = fread( fid, 1, 'int64');

    file1 = open(dipha_edge_txt, "w")
    for i in range(num_pairs[0][0]):
        bverts = fread(fid, 1, 'int64');
        file1.write(str(bverts[0][0])+"\t")
        everts = fread(fid, 1, 'double');
        file1.write(str(int(everts[0][0]))+"\t")
        pers = fread(fid, 1, 'double');
        file1.write(str(int(pers[0][0]))+"\n")
        
    file1.close() 

    subprocess.call(["src/a.out", "inputs/vert.txt", "inputs/dipha_edges.txt", str(Th), "output/"])

    vert_out = np.loadtxt('output/dimo_vert.txt');

    mask = np.zeros([nx, ny])
    for i in range(len(vert_out)):
        mask[int(vert_out[i,0]), int(vert_out[i,1])] = 1;

    return mask


def dmt_3d(patch, Th):
    nx, ny, nz = patch.shape
    im_cube = np.zeros([nx, ny, nz])
    im_cube[:, :, :] = patch

    with open(dipha_output_filename, 'wb') as output_file:
        np.int64(DIPHA_CONST).tofile(output_file)
        np.int64(DIPHA_IMAGE_TYPE_CONST).tofile(output_file)
        np.int64(nx * ny * nz).tofile(output_file)
        np.int64(DIM).tofile(output_file)
        np.int64(nx).tofile(output_file)
        np.int64(ny).tofile(output_file)
        np.int64(nz).tofile(output_file)
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    val = int(-im_cube[i, j, k]*255)
                    np.float64(val).tofile(output_file)
        output_file.close()

    with open(vert_filename, 'w') as vert_file:
        for k in range(nz):
            sys.stdout.flush()
            for j in range(ny):
                for i in range(nx):
                    vert_file.write(str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(int(-im_cube[i, j, k] * 255)) + '\n')
        vert_file.close()

    subprocess.call(["mpiexec", "-n", "1", "dipha-graph-recon/build/dipha", "inputs/complex.bin", "inputs/diagram.bin", "inputs/dipha.edges", str(nx), str(ny), str(nz)])

    def fread(fid, nelements, dtype):

        data_array = np.fromfile(fid, dtype, nelements) # has shape (1,)
        #pdb.set_trace()
        data_array.shape = (nelements, 1) # forcing shape (1,1)
        return data_array

    fid = open(dipha_edge_filename, 'r' );

    dipha_identifier = fread(fid, 1, 'int64' );
    diagram_identifier = fread(fid, 1, 'int64' );
    num_pairs = fread( fid, 1, 'int64');

    print("num_pairs", num_pairs.shape)
    file1 = open(dipha_edge_txt, "w")
    for i in range(num_pairs[0][0]):
        bverts = fread(fid, 1, 'int64');
        
        file1.write(str(bverts[0][0])+"\t")
        everts = fread(fid, 1, 'double');
        file1.write(str(int(everts[0][0]))+"\t")
        pers = fread(fid, 1, 'double');
        file1.write(str(int(pers[0][0]))+"\n")
        
    file1.close() 

    subprocess.call(["src/a.out", "inputs/vert.txt", "inputs/dipha_edges.txt", str(Th), "output/"])

    vert_out = np.loadtxt('output/dimo_vert.txt');
    mask = np.zeros([nx, ny, nz])
    for i in range(len(vert_out)):
        mask[int(vert_out[i,0]), int(vert_out[i,1]), int(vert_out[i,2])] = 1;

    return mask


def dmt(patch, Th):

	B, C, H, W = patch.shape
	patch = np.array(patch.detach().cpu())
	Th = np.array(Th.detach().cpu())
	dmt_map = np.zeros((B, C, H, W))
    
	for i in range(B):
		dmt_map[i,0,:,:]= dmt_2d(patch[i,0,:,:], Th[i] * 255)

	return dmt_map



def load_npz(filepath):
    multidata = np.load(filepath) # returns dictionary
    #print(list(multidata.keys())) # to print the keys
    myarr = multidata['softmax']
    smallarr = myarr[1]
    print(smallarr.shape)
    #sys.exit()
    #print(myarr.shape) # eg: (2, 291, 512, 512)
    return smallarr # only foreground class # values in [0.0,1.0] range


def save_img(arr, dstdir, sitkrefpath):
    print(arr.shape)
    sitkimg = sitk.GetImageFromArray(arr)

    sitkrefimg = sitk.ReadImage(sitkrefpath)
    sitkimg.SetSpacing(sitkrefimg.GetSpacing())
    sitkimg.SetOrigin(sitkrefimg.GetOrigin())
    sitkimg.SetDirection(sitkrefimg.GetDirection())
    sitk.WriteImage(sitkimg, os.path.join(dstdir, sitkrefpath.split('/')[-1]))


if __name__ == "__main__":

    # Input should be likelihood maps, not binary seg map
    srcpath = "/data/saumgupta/slicer-tool/datasets/vessel-data/nnunet/nnUNet_results/nnUNet/3d_fullres/Task101_VESSEL/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/PA_000100.npz"

    gtrefpath = "/data/saumgupta/slicer-tool/datasets/vessel-data/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task101_VESSEL/labelsTr/PA_000100.nii.gz"

    dstdir = "/data/saumgupta/slicer-tool/datasets/vessel-data/outputs/dmt"

    img_vol = load_npz(srcpath)

    mask = dmt_3d(img_vol, 250)

    save_img(mask, dstdir, gtrefpath)

    print(time.time() - t0)